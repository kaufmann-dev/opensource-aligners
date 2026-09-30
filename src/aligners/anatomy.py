"""Tooth solids, anatomical frames, landmarks and the gingiva heightfield (arch frame)."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import ndimage, sparse
from scipy.interpolate import splev, splprep
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree

from aligners import fdi
from aligners.case import CaseError, read_json, read_mesh, write_json, write_mesh
from aligners.fdi import Jaw
from aligners.geometry import Array, boundary_loops, sample_surface
from aligners.scan import patient_right_sign

STUB_DEPTH_MM = 3.0
STUB_TAPER = 0.15
CR_ROOT_FRACTION = 0.4
GUM_RES_MM = 0.1
GUM_SOCKET_DROP_MM = 1.0
CONTACT_HEIGHT_FRACTION = 0.4  # contacts are searched in the top 60 % of the crown


class ArchCurve:
    """Smooth dental arch curve in the occlusal (xy) plane, parametrized by arc length s.

    s = 0 lies where the curve crosses the sagittal plane (x = 0); s increases toward +x.
    The curve is a smoothing spline through ordered control points, extended straight by
    30 mm past both ends so distal teeth can be placed beyond the last control point.
    """

    def __init__(self, points: Array, smoothing_mm: float = 0.3):
        self.points = np.asarray(points, dtype=np.float64)
        self.smoothing_mm = smoothing_mm
        pts = self.points if self.points[0, 0] <= self.points[-1, 0] else self.points[::-1]
        chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        k = min(3, len(pts) - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], u=chord, k=k, s=len(pts) * smoothing_mm**2)
        u = np.linspace(chord[0], chord[-1], 3000)
        dense = np.stack(splev(u, tck), axis=1)
        ext = 30.0
        t0 = dense[0] - dense[1]
        t1 = dense[-1] - dense[-2]
        head = dense[0] + np.outer(
            np.linspace(ext, 0, 300, endpoint=False), t0 / np.linalg.norm(t0)
        )
        tail = dense[-1] + np.outer(np.linspace(0, ext, 301)[1:], t1 / np.linalg.norm(t1))
        self.xy = np.vstack([head, dense, tail])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        front = np.flatnonzero(np.diff(np.sign(self.xy[:, 0])) != 0)
        if len(front):
            i = int(front[np.argmax(self.xy[front, 1])])
            x0, x1 = self.xy[i, 0], self.xy[i + 1, 0]
            s0 = arc[i] + (arc[i + 1] - arc[i]) * (0 - x0) / (x1 - x0)
        else:
            s0 = arc[np.argmin(np.abs(self.xy[:, 0]))]
        self.s = arc - s0

    @classmethod
    def symmetric(cls, half: Array, smoothing_mm: float = 0.3) -> "ArchCurve":
        """Symmetric curve from half-arch points (x >= 0), ordered from the midline outward."""
        half = np.asarray(half, dtype=np.float64)
        mirrored = half[::-1] * np.array([-1.0, 1.0])
        if abs(half[0, 0]) < 1e-9:
            mirrored = mirrored[:-1]  # the midline point is shared
        return cls(np.vstack([mirrored, half]), smoothing_mm)

    def point(self, s: float | Array) -> Array:
        s = np.asarray(s, dtype=np.float64)
        return np.stack(
            [np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])], axis=-1
        )

    def tangent(self, s: float) -> Array:
        """Unit 3D tangent (z = 0) in the direction of increasing s."""
        a, b = self.point(s - 0.25), self.point(s + 0.25)
        t = np.array([b[0] - a[0], b[1] - a[1], 0.0])
        return t / np.linalg.norm(t)

    def outward(self, s: float) -> Array:
        """Unit in-plane normal pointing away from the arch interior (buccally)."""
        t = self.tangent(s)
        return np.array([-t[1], t[0], 0.0])

    def project(self, xy: Array) -> Array:
        """Arc parameter s of the closest curve point to each 2D point."""
        pts = np.atleast_2d(np.asarray(xy, dtype=np.float64))[:, :2]
        out = np.empty(len(pts))
        seg = np.diff(self.xy, axis=0)
        seg_len2 = (seg**2).sum(axis=1)
        for i, p in enumerate(pts):
            t = np.clip(((p - self.xy[:-1]) * seg).sum(axis=1) / seg_len2, 0.0, 1.0)
            d2 = ((self.xy[:-1] + seg * t[:, None] - p) ** 2).sum(axis=1)
            j = int(np.argmin(d2))
            out[i] = self.s[j] + t[j] * (self.s[j + 1] - self.s[j])
        return out

    def chord_step(self, s: float, direction: int, width: float) -> float:
        """Arc parameter reached from s (moving by `direction` = +-1) at chord distance width."""
        start = self.point(s)

        def excess(t: float) -> float:
            return float(np.linalg.norm(self.point(s + direction * t) - start)) - width

        if width <= 0:
            return s
        hi = width
        while excess(hi) < 0 and hi < 4 * width:
            hi *= 1.2
        root = cast(float, brentq(excess, 0.0, hi, xtol=1e-4))
        return s + direction * root

    def chord(self, s_a: float, s_b: float) -> Array:
        """Unit 3D direction (z = 0) from curve point s_b to curve point s_a."""
        d = self.point(s_a) - self.point(s_b)
        return np.array([d[0], d[1], 0.0]) / np.linalg.norm(d)

    def as_dict(self) -> dict[str, object]:
        return {"points": self.points, "smoothing_mm": self.smoothing_mm}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "ArchCurve":
        return cls(np.asarray(d["points"], float), float(d["smoothing_mm"]))  # type: ignore[arg-type]


@dataclass
class Tooth:
    fdi: int
    center: Array  # area-weighted crown centroid
    o: Array  # occlusal long axis
    m: Array  # mesial (toward the midline)
    b: Array  # buccal/labial
    cr: Array  # center of resistance
    margin_center: Array
    mesial_contact: Array
    distal_contact: Array
    cusp: Array  # cusp tip / incisal point
    fa: Array  # facial axis point
    width: float
    crown_height: float
    root_length: float
    root_from_ct: bool
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extras"}
        d.update(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Tooth":
        known = {k for k in cls.__dataclass_fields__ if k != "extras"}
        kwargs: dict[str, Any] = {
            k: (np.asarray(v, float) if isinstance(v, list) else v)
            for k, v in d.items()
            if k in known
        }
        return cls(**kwargs, extras={k: v for k, v in d.items() if k not in known})


def submesh(mesh: trimesh.Trimesh, face_mask: NDArray[np.bool_]) -> trimesh.Trimesh:
    faces = np.asarray(mesh.faces)[face_mask]
    used, inverse = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(np.asarray(mesh.vertices)[used], inverse.reshape(-1, 3), process=False)


def _loop_matches_winding(faces: NDArray[np.int64], loop: NDArray[np.int64]) -> bool:
    """True if the loop direction equals the half-edge direction of the faces along it."""
    a, b = loop[0], loop[1]
    for f in faces[np.any(faces == a, axis=1) & np.any(faces == b, axis=1)]:
        i = int(np.flatnonzero(f == a)[0])
        return bool(f[(i + 1) % 3] == b)
    return True


def close_crown(patch: trimesh.Trimesh, axis: Array, stub: bool = True) -> trimesh.Trimesh:
    """Close a crown surface patch with fan caps; with `stub`, first extend the margin by a
    tapered stub (for fabrication, so a moved crown never exposes a gap above the gingiva)."""
    V = np.asarray(patch.vertices)
    F = np.asarray(patch.faces, dtype=np.int64)
    loops = boundary_loops(F)
    if not loops:
        return patch
    verts = [V]
    faces = [F]
    n = len(V)
    for k, loop in enumerate(loops):
        forward = _loop_matches_winding(F, loop)
        ring = V[loop]
        if k == 0 and stub:
            center = ring.mean(axis=0)
            radial = ring - center
            radial -= np.outer(radial @ axis, axis)
            stub_ring = ring - STUB_DEPTH_MM * axis - STUB_TAPER * radial
            idx_ring = loop
            idx_stub = np.arange(n, n + len(loop))
            n += len(loop)
            verts.append(stub_ring)
            nxt = np.roll(np.arange(len(loop)), -1)
            a, b = idx_ring, idx_ring[nxt]
            c, d = idx_stub, idx_stub[nxt]
            # The new faces must traverse the shared margin edge opposite to the patch.
            quads = (
                np.concatenate([np.stack([b, a, c], 1), np.stack([b, c, d], 1)])
                if forward
                else np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
            )
            faces.append(quads)
            cap_loop, cap_pts, cap_forward = idx_stub, stub_ring, forward
        else:
            cap_loop, cap_pts, cap_forward = loop, ring, forward
        cidx = n
        verts.append(cap_pts.mean(axis=0)[None])
        n += 1
        nxt = np.roll(np.arange(len(cap_loop)), -1)
        i, j = cap_loop, cap_loop[nxt]
        c = np.full_like(i, cidx)
        # The stub bottom ring runs like the margin; both caps reverse that direction.
        faces.append(np.stack([j, i, c], 1) if cap_forward else np.stack([i, j, c], 1))
    solid = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
    if solid.volume < 0:
        solid.invert()
    return solid


def _weighted_centroid(mesh: trimesh.Trimesh) -> Array:
    area = np.asarray(mesh.area_faces)
    return (np.asarray(mesh.triangles_center) * area[:, None]).sum(0) / area.sum()


def ideal_arch(
    teeth: dict[int, "Tooth"], widen: dict[int, float] | None = None, blend_to: int = 6
) -> tuple[Array, list[int]]:
    """Ideal arch control points through the contact chain, ordered along the arch.

    Point k on each side is the contact between positions k and k+1 (k = 0: the midline
    contact at x = 0). Anteriorly the points are the left/right average (symmetric); toward
    contact `blend_to - 1` they blend into each side's actual contact, so the curve meets
    teeth held in place. `widen` adds per-position transverse offsets (expansion).
    Returns the points and their contact indices."""
    widen = widen or {}
    sides: dict[bool, dict[int, Array]] = {True: {}, False: {}}
    present = {(fdi.is_patient_right(n), fdi.position(n)) for n in teeth}
    for n, t in teeth.items():
        right, k = fdi.is_patient_right(n), fdi.position(n)
        if (right, k + 1) in present:
            p = t.distal_contact[:2].copy()
            p[0] += np.sign(p[0]) * (widen.get(k, 0.0) + widen.get(k + 1, 0.0)) / 2
            sides[right][k] = p
    ks = sorted(set(sides[True]) | set(sides[False]))
    sym = {
        k: np.mean(
            [[abs(sides[r][k][0]), sides[r][k][1]] for r in (True, False) if k in sides[r]], axis=0
        )
        for k in ks
    }
    sign = {
        r: float(np.sign(np.mean([p[0] for p in sides[r].values()])))
        if sides[r]
        else (1.0 if r else -1.0)
        for r in (True, False)
    }
    if sign[True] == sign[False]:
        sign[False] = -sign[True]
    chains: dict[bool, list[Array]] = {}
    for r in (True, False):
        pts = []
        for k in ks:
            mirrored = np.array([sign[r] * sym[k][0], sym[k][1]])
            beta = float(np.clip(k / max(blend_to - 1, 1), 0.0, 1.0))
            actual = sides[r].get(k, mirrored)
            pts.append((1 - beta) * mirrored + beta * actual)
        chains[r] = pts
    neg, pos = (True, False) if sign[True] < 0 else (False, True)
    points = list(reversed(chains[neg]))
    index = list(reversed(ks))
    centrals = [t for n, t in teeth.items() if fdi.position(n) == 1]
    if len(centrals) == 2:
        points.append(np.array([0.0, float(np.mean([t.mesial_contact[1] for t in centrals]))]))
        index.append(0)
    points += chains[pos]
    index += ks
    return np.array(points), index


def fit_arch(points: dict[int, Array]) -> ArchCurve:
    """Arch curve through per-tooth points, ordered along the arch."""
    order = fdi.arch_order(list(points))
    return ArchCurve(np.array([points[n][:2] for n in order]))


def build_tooth(
    number: int,
    jaw: Jaw,
    patch: trimesh.Trimesh,
    curve: ArchCurve,
    ct_axis: Array | None,
    ct_mesh: trimesh.Trimesh | None,
    contacts: dict[str, Array],
) -> tuple[Tooth, trimesh.Trimesh]:
    """Frame and landmarks of one tooth. `contacts` holds the neighbor contacts found so far
    ("mesial"/"distal"); a missing side (arch end) falls back to the crown extreme."""
    V = np.asarray(patch.vertices)
    center = _weighted_centroid(patch)
    o = ct_axis / np.linalg.norm(ct_axis) if ct_axis is not None else np.array([0.0, 0.0, 1.0])
    x_sign = patient_right_sign(jaw) if fdi.is_patient_right(number) else -patient_right_sign(jaw)
    s_here = float(curve.project(center[:2])[0])
    m = -x_sign * curve.tangent(s_here)
    if "mesial" in contacts and "distal" in contacts:
        m = contacts["mesial"] - contacts["distal"]
    m = m - np.dot(m, o) * o
    m /= np.linalg.norm(m)
    b = np.asarray(np.cross(o, m), dtype=np.float64)
    if np.dot(b, curve.outward(s_here)) < 0:
        b = -b

    solid = close_crown(patch, o)
    loops = boundary_loops(np.asarray(patch.faces, dtype=np.int64))
    margin = V[loops[0]] if loops else V
    margin_center = margin.mean(axis=0)

    h = (V - center) @ o
    hmin, hmax = float(h.min()), float(h.max())
    upper_part = V[h >= hmin + CONTACT_HEIGHT_FRACTION * (hmax - hmin)]
    along = (upper_part - center) @ m
    mesial = contacts["mesial"] if "mesial" in contacts else upper_part[np.argmax(along)]
    distal = contacts["distal"] if "distal" in contacts else upper_part[np.argmin(along)]
    width = float(np.linalg.norm((mesial - distal) - np.dot(mesial - distal, o) * o))
    top = V[h >= np.quantile(h, 0.97)]
    cusp = top.mean(axis=0)
    mid_h = 0.5 * (float(((margin - center) @ o).mean()) + float((cusp - center) @ o))
    band = V[(np.abs(h - mid_h) < 0.75) & (np.abs((V - center) @ m) < 1.5)]
    fa = band[np.argmax((band - center) @ b)] if len(band) else center + b * 3.0

    if ct_mesh is not None:
        depth = (np.asarray(ct_mesh.vertices) - margin_center) @ o
        root_length = float(max(3.0, -depth.min()))
        from_ct = True
    else:
        root_length = fdi.ROOT_LENGTH_MM[jaw][fdi.position(number)]
        from_ct = False
    cr = margin_center - CR_ROOT_FRACTION * root_length * o
    tooth = Tooth(
        fdi=number,
        center=center,
        o=o,
        m=m,
        b=b,
        cr=cr,
        margin_center=margin_center,
        mesial_contact=mesial,
        distal_contact=distal,
        cusp=cusp,
        fa=fa,
        width=width,
        crown_height=float(np.dot(cusp - margin_center, o)),
        root_length=root_length,
        root_from_ct=from_ct,
    )
    return tooth, solid


def _toward_midline(tooth: int, neighbor: int) -> bool:
    """True if `neighbor` is on the mesial side of `tooth`."""
    if fdi.is_patient_right(tooth) != fdi.is_patient_right(neighbor):
        return True
    return fdi.position(neighbor) < fdi.position(tooth)


def neighbor_contact(patch: trimesh.Trimesh, neighbor: trimesh.Trimesh, axis: Array) -> Array:
    """Contact point of a crown with its neighbor: the occlusal-most point of the region
    where the two crowns come closest (their shared interproximal crease)."""
    V = np.asarray(patch.vertices)
    d, _ = KDTree(np.asarray(neighbor.vertices)).query(V)
    near = V[d <= d.min() + 0.25]
    h = near @ axis
    return near[h >= h.max() - 0.75].mean(axis=0)


@dataclass
class GumField:
    """Gingiva as a heightfield z = G(x, y) in the arch frame (NaN = no gingiva)."""

    height: NDArray[np.float32]
    x0: float
    y0: float
    res: float

    def save(self, path: Path) -> None:
        np.savez_compressed(path, height=self.height, x0=self.x0, y0=self.y0, res=self.res)

    @classmethod
    def load(cls, path: Path) -> "GumField":
        d = np.load(path)
        return cls(d["height"], float(d["x0"]), float(d["y0"]), float(d["res"]))

    @cached_property
    def _grids(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        h = self.height.astype(np.float64)
        valid = np.isfinite(h)
        filled = np.where(valid, h, -1e3)
        gx, gy = np.gradient(np.where(valid, h, ndimage.grey_dilation(filled, size=5)), self.res)
        return filled, valid.astype(np.float64), np.minimum(np.hypot(gx, gy), 50.0)

    def sample(self, xy: Array) -> tuple[Array, Array]:
        """Height and gradient magnitude at points (NaN outside the gingiva)."""
        filled, valid, grad = self._grids
        ij = [(xy[:, 0] - self.x0) / self.res, (xy[:, 1] - self.y0) / self.res]
        z = ndimage.map_coordinates(filled, ij, order=1, mode="constant", cval=-1e3)
        ok = ndimage.map_coordinates(valid, ij, order=1, mode="constant", cval=0.0)
        g = ndimage.map_coordinates(grad, ij, order=1, mode="nearest")
        return np.where(ok > 0.999, z, np.nan), g

    def sdf(self, pts: Array) -> Array:
        """Approximate signed distance to the gingiva solid (negative below the surface)."""
        z, g = self.sample(pts[:, :2])
        d = (pts[:, 2] - z) / np.sqrt(1 + g**2)
        return np.where(np.isfinite(d), d, 1e3)


def _harmonic_fill(h: NDArray[np.float64], unknown: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Fill `unknown` pixels harmonically from known neighbors; NaN known pixels are ignored."""
    idx = np.full(h.shape, -1)
    cells = np.argwhere(unknown)
    idx[unknown] = np.arange(len(cells))
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(cells))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = cells[:, 0] + di, cells[:, 1] + dj
        inside = (ni >= 0) & (nj >= 0) & (ni < h.shape[0]) & (nj < h.shape[1])
        ni, nj = np.where(inside, ni, 0), np.where(inside, nj, 0)
        nbr_unknown = inside & unknown[ni, nj]
        nbr_known = inside & ~unknown[ni, nj] & np.isfinite(h[ni, nj])
        k = np.arange(len(cells))
        rows += [k[nbr_unknown | nbr_known], k[nbr_unknown]]
        cols += [k[nbr_unknown | nbr_known], idx[ni, nj][nbr_unknown]]
        vals += [np.ones((nbr_unknown | nbr_known).sum()), -np.ones(nbr_unknown.sum())]
        rhs[nbr_known] += h[ni, nj][nbr_known]
    A = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(cells), len(cells)),
    )
    A = A + sparse.identity(len(cells)) * 1e-9
    out = h.copy()
    out[unknown] = np.asarray(spsolve(A.tocsc(), rhs), dtype=np.float64)
    return out


def gum_heightfield(mesh: trimesh.Trimesh, labels: NDArray[np.int64]) -> GumField:
    V = np.asarray(mesh.vertices)
    lo = V[:, :2].min(axis=0) - 2.0
    hi = V[:, :2].max(axis=0) + 2.0
    shape = np.ceil((hi - lo) / GUM_RES_MM).astype(int) + 1

    def raster(face_mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        grid = np.full(shape, -np.inf)
        part = submesh(mesh, face_mask)
        if len(part.faces) == 0:
            return grid
        count = int(part.area / (0.5 * GUM_RES_MM) ** 2) + 1
        pts, _ = sample_surface(part, count, seed=0)
        ij = np.floor((pts[:, :2] - lo) / GUM_RES_MM).astype(int)
        np.maximum.at(grid, (ij[:, 0], ij[:, 1]), pts[:, 2])
        return grid

    gum = raster(labels == 0)
    teeth = raster(labels != 0)
    gum_valid = np.isfinite(gum)
    closed = np.asarray(ndimage.binary_closing(gum_valid | np.isfinite(teeth), iterations=2), bool)
    filled = ndimage.maximum_filter(np.where(gum_valid, gum, -np.inf), size=3)
    gum = np.where(gum_valid, gum, np.where(closed & np.isfinite(filled), filled, np.nan))
    footprint = np.asarray(ndimage.binary_dilation(np.isfinite(teeth), iterations=3), bool)
    region = footprint & closed
    known = np.where(region, np.nan, gum)
    comps, _ = cast(tuple[NDArray[np.int32], int], ndimage.label(region))
    grown = np.asarray(ndimage.binary_dilation(np.isfinite(known)), bool)
    touching = np.unique(comps[grown & region])
    region &= np.isin(comps, touching[touching > 0])
    gum = np.where(footprint & closed & ~region, np.nan, gum)
    if region.any():
        socket = _harmonic_fill(known, region)
        # sink the socket gradually (no step at its rim, which would leave a trench by the crown)
        depth = np.asarray(ndimage.distance_transform_edt(region), dtype=np.float64) * GUM_RES_MM
        gum = np.where(region, socket - GUM_SOCKET_DROP_MM * np.clip(depth / 1.0, 0.0, 1.0), gum)
    # holes in the scanned gingiva that are enclosed by gingiva: fill smoothly
    valid = np.isfinite(gum)
    holes = np.asarray(ndimage.binary_fill_holes(valid), dtype=bool) & ~valid
    if holes.any():
        gum = _harmonic_fill(gum, holes)
    return GumField(gum.astype(np.float32), float(lo[0]), float(lo[1]), GUM_RES_MM)


def build_anatomy(
    out_dir: Path,
    jaw: Jaw,
    mesh: trimesh.Trimesh,
    labels: NDArray[np.int64],
    ct: dict[int, tuple[Array, trimesh.Trimesh]] | None,
) -> dict[str, Any]:
    """Build and save crowns, frames, landmarks and gingiva for one jaw (arch frame input)."""
    numbers = sorted(int(v) for v in np.unique(labels) if v)
    if len(numbers) < 2:
        raise CaseError(f"{jaw}: fewer than two teeth segmented")
    patches = {n: submesh(mesh, labels == n) for n in numbers}
    order = fdi.arch_order(numbers)
    axes = {
        n: (ct[n][0] / np.linalg.norm(ct[n][0]) if ct and n in ct else np.array([0.0, 0.0, 1.0]))
        for n in numbers
    }
    contacts: dict[int, dict[str, Array]] = {n: {} for n in numbers}
    chain = []
    for a, b in zip(order[:-1], order[1:], strict=True):
        ca = neighbor_contact(patches[a], patches[b], axes[a])
        cb = neighbor_contact(patches[b], patches[a], axes[b])
        contacts[a]["mesial" if _toward_midline(a, b) else "distal"] = ca
        contacts[b]["mesial" if _toward_midline(b, a) else "distal"] = cb
        chain.append((ca + cb) / 2)
    curve = (
        ArchCurve(np.array([p[:2] for p in chain]), smoothing_mm=0.5)
        if len(chain) >= 3
        else fit_arch({n: _weighted_centroid(p) for n, p in patches.items()})
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.ply"):
        old.unlink()
    teeth = []
    warnings = []
    for n in numbers:
        ct_axis, ct_mesh = ct[n] if ct and n in ct else (None, None)
        tooth, solid = build_tooth(n, jaw, patches[n], curve, ct_axis, ct_mesh, contacts[n])
        expected = fdi.CROWN_WIDTH_MM[jaw][fdi.position(n)]
        if not 0.6 * expected <= tooth.width <= 1.4 * expected:
            warnings.append(f"tooth {n}: width {tooth.width:.1f} mm, expected ~{expected:.1f} mm")
        write_mesh(out_dir / f"{n}_crown.ply", solid)
        write_mesh(out_dir / f"{n}_patch.ply", patches[n])
        if ct_mesh is not None:
            write_mesh(out_dir / f"{n}_root.ply", ct_mesh)
        teeth.append(tooth.as_dict())
    gum = gum_heightfield(mesh, labels)
    gum.save(out_dir / "gum.npz")
    info = {"jaw": jaw, "curve": curve.as_dict(), "teeth": teeth, "warnings": warnings}
    write_json(out_dir / "teeth.json", info)
    return info


@dataclass
class JawAnatomy:
    jaw: Jaw
    curve: ArchCurve
    teeth: dict[int, Tooth]
    crowns: dict[int, trimesh.Trimesh]  # closed with a stub below the margin (fabrication)
    roots: dict[int, trimesh.Trimesh]
    gum: GumField
    patches: dict[int, trimesh.Trimesh]  # scanned crown surface only

    @classmethod
    def load(cls, directory: Path, jaw: Jaw) -> "JawAnatomy":
        info = read_json(directory / "teeth.json")
        teeth = {t["fdi"]: Tooth.from_dict(t) for t in info["teeth"]}
        crowns = {n: read_mesh(directory / f"{n}_crown.ply") for n in teeth}
        roots = {
            n: read_mesh(directory / f"{n}_root.ply")
            for n in teeth
            if (directory / f"{n}_root.ply").exists()
        }
        return cls(
            jaw,
            ArchCurve.from_dict(info["curve"]),
            teeth,
            crowns,
            roots,
            GumField.load(directory / "gum.npz"),
            {n: read_mesh(directory / f"{n}_patch.ply") for n in teeth},
        )

    def order(self) -> list[int]:
        return fdi.arch_order(list(self.teeth))

    def enamel_samples(self, number: int, count: int = 1200, seed: int = 0) -> Array:
        """Samples of the scanned crown surface (contacts and collisions are judged on these)."""
        return sample_surface(self.patches[number], count, seed=seed)[0]

    @cached_property
    def contact_solids(self) -> dict[int, trimesh.Trimesh]:
        """Crowns closed flat at their own margin: the real enamel volume, no stub."""
        return {n: close_crown(p, self.teeth[n].o, stub=False) for n, p in self.patches.items()}
