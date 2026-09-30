"""Signed-distance fields of posed dentitions, and meshes extracted from them.

Every fabricated solid is a field in the arch frame (negative inside). There are no mesh
booleans: union is min, intersection is max, offsets are level sets.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.measure import marching_cubes

from aligners.anatomy import GumField
from aligners.geometry import Array, MeshSDF, decimate, transform_points

Field = Callable[[Array], Array]
BIG = 1e3


def edt(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Euclidean distance (in voxels) of each True voxel to the nearest False voxel."""
    return np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64)


@dataclass
class Grid:
    origin: Array
    res: float
    shape: tuple[int, int, int]

    @classmethod
    def around(cls, lo: Array, hi: Array, res: float) -> "Grid":
        shape = tuple(int(v) for v in np.ceil((hi - lo) / res).astype(int) + 1)
        return cls(np.asarray(lo, float), res, (shape[0], shape[1], shape[2]))

    def axes(self) -> list[Array]:
        return [self.origin[i] + self.res * np.arange(self.shape[i]) for i in range(3)]

    def points(self, xs: slice = slice(None)) -> Array:
        ax = self.axes()
        g = np.meshgrid(ax[0][xs], ax[1], ax[2], indexing="ij")
        return np.stack([c.ravel() for c in g], axis=1)


@dataclass
class GridSDF:
    """Signed distance of a solid sampled on a regular grid (trilinear lookup), clamped to
    +-bound. Built once per crown; far cheaper to evaluate than exact mesh queries."""

    origin: Array
    res: float
    values: NDArray[np.float32]
    bound: float

    @property
    def lo(self) -> Array:
        return self.origin

    @property
    def hi(self) -> Array:
        return self.origin + self.res * (np.asarray(self.values.shape) - 1)

    @classmethod
    def build(cls, mesh: trimesh.Trimesh, res: float = 0.15, bound: float = 3.0) -> "GridSDF":
        exact = MeshSDF.of(mesh)
        grid = Grid.around(exact.lo - bound, exact.hi + bound, res)
        values = np.empty(grid.shape, dtype=np.float32)
        for i0 in range(0, grid.shape[0], 16):
            i1 = min(grid.shape[0], i0 + 16)
            pts = grid.points(slice(i0, i1))
            values[i0:i1] = exact(pts, bound=bound).reshape(i1 - i0, grid.shape[1], grid.shape[2])
        return cls(grid.origin, res, values, bound)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            origin=self.origin,
            res=self.res,
            values=self.values.astype(np.float16),
            bound=self.bound,
        )

    @classmethod
    def load(cls, path: Path) -> "GridSDF":
        d = np.load(path)
        return cls(d["origin"], float(d["res"]), d["values"].astype(np.float32), float(d["bound"]))

    def __call__(self, pts: Array, bound: float = np.inf) -> Array:
        ijk = ((pts - self.origin) / self.res).T
        d = ndimage.map_coordinates(self.values, ijk, order=1, mode="constant", cval=self.bound)
        return np.clip(d, -min(bound, self.bound), min(bound, self.bound))


@dataclass
class ToothSolid:
    """A crown solid (plus attachments) in its initial arch-frame position, and its pose."""

    sdf: GridSDF
    attachments: list[tuple[Array, Array, Array]]  # (center, axes as columns, half extents)
    pose: Array
    vertices: Array  # crown surface samples (initial frame), for footprints
    rounding: float = 0.3

    @property
    def surface(self) -> Array:
        """Posed surface samples projected to the occlusal plane (xy)."""
        return transform_points(self.pose, self.vertices)[:, :2]

    @property
    def surface_z(self) -> Array:
        return transform_points(self.pose, self.vertices)[:, 2]

    def local_bounds(self, margin: float) -> tuple[Array, Array]:
        lo, hi = self.sdf.lo.copy(), self.sdf.hi.copy()
        for c, _, h in self.attachments:
            r = np.linalg.norm(h)
            lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
        return lo - margin, hi + margin

    def world_bounds(self, margin: float) -> tuple[Array, Array]:
        lo, hi = self.local_bounds(margin)
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        w = transform_points(self.pose, corners)
        return w.min(axis=0), w.max(axis=0)

    def __call__(self, pts: Array, bound: float) -> Array:
        local = transform_points(np.linalg.inv(self.pose), pts)
        d = self.sdf(local, bound=bound)
        for c, R, h in self.attachments:
            q = np.abs((local - c) @ R) - (h - self.rounding)
            box = (
                np.linalg.norm(np.maximum(q, 0), axis=1)
                + np.minimum(q.max(axis=1), 0)
                - self.rounding
            )
            d = np.minimum(d, box)
        return d


class Dentition:
    """d_teeth, d_gum and d_model of one posed jaw."""

    def __init__(self, teeth: list[ToothSolid], gum: GumField | None):
        self.teeth = teeth
        self.gum = gum

    def d_teeth(self, pts: Array, bound: float = 4.0) -> Array:
        out = np.full(len(pts), bound)
        for tooth in self.teeth:
            lo, hi = tooth.world_bounds(bound)
            near = np.all((pts >= lo) & (pts <= hi), axis=1)
            if near.any():
                out[near] = np.minimum(out[near], tooth(pts[near], bound))
        return out

    def d_gum(self, pts: Array) -> Array:
        return self.gum.sdf(pts) if self.gum is not None else np.full(len(pts), BIG)

    def bounds(self, margin: float) -> tuple[Array, Array]:
        bounds = [t.world_bounds(margin) for t in self.teeth]
        lo = np.min(np.array([b[0] for b in bounds]), axis=0)
        hi = np.max(np.array([b[1] for b in bounds]), axis=0)
        return lo, hi


def banded(
    grid: Grid, fn: Field, band: tuple[float, float], coarse_factor: int = 6
) -> NDArray[np.float32]:
    """Evaluate fn on grid exactly where values fall in [band[0], band[1]]; interpolate elsewhere."""
    nx, ny, nz = (int(np.ceil(s / coarse_factor)) + 1 for s in grid.shape)
    cgrid = Grid(grid.origin, grid.res * coarse_factor, (nx, ny, nz))
    coarse = fn(cgrid.points()).reshape(cgrid.shape)
    diag = cgrid.res * np.sqrt(3)
    lo, hi = band
    active_c = (coarse >= lo - diag) & (coarse <= hi + diag)
    active_c = ndimage.binary_dilation(active_c)
    idx = [np.arange(s) / coarse_factor for s in grid.shape]
    out = np.empty(grid.shape, dtype=np.float32)
    for i0 in range(0, grid.shape[0], 32):
        i1 = min(grid.shape[0], i0 + 32)
        mesh_idx = np.meshgrid(idx[0][i0:i1], idx[1], idx[2], indexing="ij")
        block = ndimage.map_coordinates(coarse, mesh_idx, order=1, mode="nearest")
        ci = [
            np.clip(np.round(m), 0, s - 1).astype(np.intp)
            for m, s in zip(mesh_idx, cgrid.shape, strict=True)
        ]
        act = np.asarray(active_c[ci[0], ci[1], ci[2]], dtype=bool)
        if act.any():
            pts = grid.points(slice(i0, i1)).reshape(block.shape + (3,))[act]
            block[act] = fn(pts)
        out[i0:i1] = block
    return out


def glyph_sdf(
    text: str, height_mm: float, px_mm: float = 0.04
) -> tuple[NDArray[np.float64], float, float]:
    """2D signed distance (mm, negative inside) of rendered text; returns (field, width, height)."""
    size_px = int(round(height_mm / px_mm))
    font = ImageFont.load_default(size=size_px)
    left, top, right, bottom = font.getbbox(text)
    pad = 10
    img = Image.new("L", (int(right - left) + 2 * pad, int(bottom - top) + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - left, pad - top), text, fill=255, font=font)
    mask = np.asarray(img) > 127
    inside = edt(mask) * px_mm
    outside = edt(~mask) * px_mm
    field = np.where(mask, -inside, outside).astype(np.float64)
    return field, img.width * px_mm, img.height * px_mm


def label_field(text: str, height: float, center: Array, right: Array, up: Array) -> Field:
    """Field of text lying in the plane spanned by right/up through center (2D, no depth)."""
    sdf2d, w, h = glyph_sdf(text, height)
    px = w / sdf2d.shape[1]

    def fn(pts: Array) -> Array:
        rel = pts - center
        u = rel @ right + w / 2
        v = h / 2 - rel @ up
        return ndimage.map_coordinates(sdf2d, [v / px, u / px], order=1, mode="constant", cval=5.0)

    return fn


def extract(grid: Grid, values: NDArray[np.float32], target_faces: int) -> trimesh.Trimesh:
    """Mesh of the region values < 0: marching cubes, largest part, topology-safe decimation."""
    # values exactly on the iso-level make zero-area triangles that open holes when merged
    values = np.where(np.abs(values) < 1e-4, np.float32(1e-4), values)
    padded = np.pad(values, 1, constant_values=BIG)
    verts, faces, _, _ = marching_cubes(-padded, 0.0)
    mesh = trimesh.Trimesh(grid.origin + (verts - 1) * grid.res, faces, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda p: len(p.faces))
    mesh = decimate(mesh, target_faces)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


@dataclass
class Curtain:
    """Undercut blockout of one posed crown, seen along the insertion direction (+z).

    The solid is the crown's occlusal-view footprint extruded down from its top surface: the
    region an aligner cannot reach when seated straight down. It is shrunk by the retention
    depth, so undercuts up to that depth stay for grip."""

    lo: Array
    res: float
    sdf2d: NDArray[np.float64]  # footprint signed distance (mm), negative inside
    top: NDArray[np.float64]  # crown top height per pixel (-BIG outside)

    def __call__(self, pts: Array, retention: float) -> Array:
        ij = ((pts[:, :2] - self.lo) / self.res).T
        d2 = ndimage.map_coordinates(self.sdf2d, ij, order=1, mode="constant", cval=BIG)
        top = ndimage.map_coordinates(self.top, ij, order=1, mode="nearest")
        return np.maximum(d2 + retention, pts[:, 2] - top)


def curtains(dent: Dentition, res: float = 0.08) -> list[Curtain]:
    out = []
    for tooth in dent.teeth:
        # dense surface samples of the posed crown (attachments excluded: aligners must grip them)
        v = transform_points(tooth.pose, tooth.vertices)
        lo = v[:, :2].min(axis=0) - 1.0
        hi = v[:, :2].max(axis=0) + 1.0
        shape = tuple(int(x) for x in np.ceil((hi - lo) / res).astype(int) + 1)
        top = np.full(shape, -BIG)
        ij = ((tooth.surface - lo) / res).astype(int)
        np.maximum.at(top, (ij[:, 0], ij[:, 1]), tooth.surface_z)
        mask = np.asarray(ndimage.binary_closing(top > -BIG, iterations=3), dtype=bool)
        mask = np.asarray(ndimage.binary_fill_holes(mask), dtype=bool)
        # Pixels without samples take the LOWEST neighboring top: a maximum would lift the
        # footprint rim to the occlusal height and grow thin fins beyond the crown surface.
        for _ in range(6):
            missing = mask & (top <= -BIG)
            if not missing.any():
                break
            low = ndimage.minimum_filter(np.where(top > -BIG, top, BIG), size=3)
            top = np.where(missing & (low < BIG), low, top)
        top = np.where(mask & (top <= -BIG), -BIG, top)
        # close single-pixel dips (columns whose highest point was missed by the samples)
        closed = ndimage.grey_closing(np.where(mask, top, -BIG), size=3)
        top = np.where(
            mask, np.maximum(top, np.minimum(closed, ndimage.maximum_filter(top, size=3))), top
        )
        sdf2d = np.where(mask, -edt(mask), edt(~mask)) * res
        out.append(Curtain(lo, res, ndimage.gaussian_filter(sdf2d, 1.0), top))
    return out


def blocked(dent: Dentition, retention: float) -> Callable[[Array], Array]:
    """Signed distance to the union of all crowns' undercut curtains."""
    parts = curtains(dent)

    boxes = [(c.lo, c.lo + c.res * (np.asarray(c.sdf2d.shape) - 1)) for c in parts]

    def fn(pts: Array) -> Array:
        d = np.full(len(pts), BIG)
        for c, (lo, hi) in zip(parts, boxes, strict=True):
            near = np.all((pts[:, :2] >= lo) & (pts[:, :2] <= hi), axis=1)
            if near.any():
                d[near] = np.minimum(d[near], c(pts[near], retention))
        return d

    return fn
