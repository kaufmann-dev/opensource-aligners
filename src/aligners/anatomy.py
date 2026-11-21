"""Tooth solids, anatomical frames, landmarks and the gingiva heightfield (arch frame)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import igl
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import ndimage, sparse
from scipy.interpolate import splev, splprep
from scipy.optimize import brentq, least_squares
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree

from aligners import fdi
from aligners.case import CaseError, read_json, read_mesh, write_json, write_mesh
from aligners.fdi import Jaw
from aligners.geometry import (
    Array,
    MeshSDF,
    boundary_loops,
    decimate,
    grid_points,
    level_set,
    pca,
    sample_surface,
)
from aligners.scan import patient_right_sign

STUB_DEPTH_MM = 3.0
STUB_TAPER = 0.15
CR_ROOT_FRACTION = 0.4
GUM_RES_MM = 0.1
GUM_SOCKET_DROP_MM = 1.0
CONTACT_HEIGHT_FRACTION = 0.4  # contacts are searched in the top 60 % of the crown
RIDGE_BINS = 8  # mesiodistal bins of a canine's cusp-ridge crest
RIDGE_PASSES = 4
FIT_NEAR_MM = 1.5  # CT surface this close to the scanned crown, facing the same way, moves onto it
FIT_RIM_MM = 0.4  # ... unless it meets the scan at the patch's open rim
HIDDEN_CLEAR_MM = 0.3  # completed crown surface farther than this from the scan: missed enamel
SOLID_RES_MM = 0.15  # grid of the completed crown solids
SCAN_BAND_MM = 0.3  # closed scans count this close to the scanned surface (and inside the CT)
SOLID_FACES = 20000
ARCH_KNOTS_MM = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 36.0, 44.0, 54.0])
ARCH_LENGTH_MM = 80.0  # half-arch length built from the midline (beyond every tooth)
ARCH_STEP_MM = 0.25
ARCH_CURVATURE_PENALTY = 30.0  # mm of fit residual per 1/mm of curvature second difference
ARCH_SLOPE_PENALTY = 3.0  # ... per 1/mm of curvature first difference
ARCH_BEND_PENALTY = 100.0  # ... per 1/mm of per-side bend
ARCH_ANCHOR_WEIGHT = 10.0
ARCH_IRLS_PASSES = 4
ARCH_ROBUST_SCALE_MM = 1.0  # residual scale for down-weighting displaced teeth (Tukey)


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
    # loops of fewer than three vertices come from pinch vertices of the patch: nothing to cap
    loops = [loop for loop in boundary_loops(F) if len(loop) >= 3]
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


def fit_ct_tooth(ct: trimesh.Trimesh, patch: trimesh.Trimesh) -> trimesh.Trimesh:
    """The CT tooth moved along its normals onto the scanned crown where the scan saw it, the
    offset spread harmonically over the rest: the scan's crown, completed with the CT's shape
    where the scan saw nothing (enamel a crowded neighbor hid, the root)."""
    V = np.asarray(ct.vertices, dtype=np.float64)
    N = np.asarray(ct.vertex_normals, dtype=np.float64)
    PV = np.ascontiguousarray(patch.vertices, dtype=np.float64)
    PF = np.ascontiguousarray(patch.faces, dtype=np.int64)
    d2, face, closest = (
        np.asarray(a) for a in igl.point_mesh_squared_distance(np.ascontiguousarray(V), PV, PF)
    )
    facing = np.einsum("ij,ij->i", np.asarray(patch.face_normals)[face], N) > 0.5
    known = (np.sqrt(d2) < FIT_NEAR_MM) & facing
    loops = boundary_loops(PF)
    if loops:
        rim, _ = KDTree(PV[np.concatenate(loops)]).query(closest)
        known &= rim > FIT_RIM_MM
    offset = np.zeros(len(V))
    offset[known] = np.einsum("ij,ij->i", closest[known] - V[known], N[known])
    if known.any() and not known.all():
        edges = np.asarray(ct.edges_unique)
        ones = np.ones(2 * len(edges))
        adj = sparse.coo_matrix(
            (ones, (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
            shape=(len(V), len(V)),
        ).tocsr()
        lap = sparse.csr_matrix(sparse.diags(np.asarray(adj.sum(axis=1)).ravel()) - adj)
        free = ~known
        # the tiny diagonal keeps parts without any scanned vertex solvable (offset 0)
        system = sparse.csc_matrix(lap[free][:, free] + 1e-9 * sparse.identity(int(free.sum())))
        offset[free] = np.asarray(spsolve(system, -(lap[free][:, known] @ offset[known])))
    return trimesh.Trimesh(V + offset[:, None] * N, np.asarray(ct.faces), process=False)


@dataclass
class Neighbor:
    """Distance fields of a neighboring crown as scanned and as its fitted CT tooth."""

    scanned: MeshSDF
    fitted: MeshSDF | None


def _claim(own: Array, points: Array, neighbors: list[Neighbor]) -> Array:
    """Negative where a point belongs to this tooth rather than a neighbor: outside the
    neighbor's scanned crown and, where both fitted CT teeth reach, deeper in this one's (own:
    this tooth's fitted distance at the points)."""
    out = np.full(len(points), -1.0)
    for nb in neighbors:
        out = np.maximum(out, -nb.scanned(points, bound=1.0))
        if nb.fitted is not None:
            out = np.maximum(out, own - nb.fitted(points, bound=1.0))
    return out


def complete_crowns(
    patch: trimesh.Trimesh,
    crown: trimesh.Trimesh,
    solid: trimesh.Trimesh,
    fit: trimesh.Trimesh,
    floor: Callable[[Array], Array],
    enamel: Callable[[Array], Array],
    neighbors: list[Neighbor],
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """The closed scanned crown (with stub) and solid (closed at the margin) completed by the
    fitted CT tooth where it belongs to this tooth: the crown where floor(p) < 0, the solid
    where enamel(p) < 0. The closed scans count only near the scanned surface or inside the CT
    tooth, so their caps across what the scan missed give way to the CT's shape."""
    V = np.asarray(fit.vertices)
    kept = V[floor(V) < 0]
    pad = 2 * SOLID_RES_MM
    lo = np.minimum(crown.bounds[0], kept.min(axis=0) if len(kept) else crown.bounds[0]) - pad
    hi = np.maximum(crown.bounds[1], kept.max(axis=0) if len(kept) else crown.bounds[1]) + pad
    P, shape = grid_points(lo, hi, SOLID_RES_MM)
    f = MeshSDF.of(fit)(P, bound=1.0)
    part = np.maximum(f, _claim(f, P, neighbors))
    trusted = np.minimum(MeshSDF.of(patch).unsigned(P) - SCAN_BAND_MM, f)
    out = []
    for closed, region in ((crown, floor), (solid, enamel)):
        scanned = np.maximum(MeshSDF.of(closed)(P, bound=1.0), trusted)
        d = np.minimum(scanned, np.maximum(part, region(P)))
        mesh = level_set(d.reshape(shape), lo, SOLID_RES_MM)
        out.append(decimate(mesh, SOLID_FACES, max_error=0.02))
    return out[0], out[1]


def below_plane(point: Array, axis: Array) -> Callable[[Array], Array]:
    """Negative on the `axis` side of the plane through `point`."""
    return lambda p: (point - p) @ axis


def _weighted_centroid(mesh: trimesh.Trimesh) -> Array:
    area = np.asarray(mesh.area_faces)
    return (np.asarray(mesh.triangles_center) * area[:, None]).sum(0) / area.sum()


def _smoothstep(t: Array) -> Array:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class ArchForm:
    """Ideal arch: symmetric about x = 0 and built from a smooth curvature profile along its arc
    length, so it can neither wiggle between teeth nor kink. Behind the canines each side may
    bend a little more or less (`bend`) to meet its held back teeth. `delta` moves the arch
    outward along its normal, fully at the front and fading to nothing at the held teeth."""

    y0: float  # arch front at x = 0
    kappa: Array  # curvature (1/mm) at ARCH_KNOTS_MM, constant beyond the last knot
    bend: dict[int, float]  # +1 / -1 side of x: extra curvature behind the canines (1/mm)
    s_blend: float  # arc length where the bend starts (canines)
    s_anchor: float  # arc length of the held teeth (0 if none)

    def half(self, side: int) -> tuple[Array, Array]:
        """Arc lengths and points of one half-arch, mirrored to x >= 0."""
        s = np.arange(0.0, ARCH_LENGTH_MM + ARCH_STEP_MM, ARCH_STEP_MM)
        k = np.interp(s, ARCH_KNOTS_MM, self.kappa)
        if self.s_anchor > self.s_blend:
            k = k + self.bend.get(side, 0.0) * _smoothstep(
                (s - self.s_blend) / (self.s_anchor - self.s_blend)
            )
        theta = np.r_[0.0, np.cumsum(0.5 * (k[1:] + k[:-1]) * ARCH_STEP_MM)]
        dx, dy = np.cos(theta), -np.sin(theta)
        x = np.r_[0.0, np.cumsum(0.5 * (dx[1:] + dx[:-1]) * ARCH_STEP_MM)]
        y = self.y0 + np.r_[0.0, np.cumsum(0.5 * (dy[1:] + dy[:-1]) * ARCH_STEP_MM)]
        return s, np.stack([x, y], axis=1)

    def curve(self, delta: float = 0.0) -> ArchCurve:
        halves = []
        for side in (-1, 1):
            s, p = self.half(side)
            if delta:
                t = np.gradient(p, axis=0)
                normal = np.stack([-t[:, 1], t[:, 0]], axis=1)  # outward on the x >= 0 half
                normal /= np.linalg.norm(normal, axis=1)[:, None]
                fade = 1.0 - _smoothstep(s / self.s_anchor) if self.s_anchor > 0 else 1.0
                p = p + delta * np.reshape(fade, (-1, 1)) * normal
            p = p * [side, 1.0]
            halves.append(p[::-1] if side == -1 else p[1:])
        return ArchCurve(np.vstack(halves), smoothing_mm=0.005)


def _project_half(points: Array, half: Array) -> tuple[Array, Array]:
    """Arc index position and signed distance (+ outside the arch) of x >= 0 points to a
    half-arch polyline that starts at the midline."""
    seg = np.diff(half, axis=0)
    rel = points[:, None, :] - half[None, :-1, :]
    t = np.clip((rel * seg).sum(-1) / (seg**2).sum(-1), 0.0, 1.0)
    foot = half[None, :-1, :] + t[..., None] * seg
    j = np.argmin(((points[:, None, :] - foot) ** 2).sum(-1), axis=1)
    rows = np.arange(len(points))
    normal = np.stack([-seg[j, 1], seg[j, 0]], axis=1)  # left of travel = outward
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return j + t[rows, j], ((points - foot[rows, j]) * normal).sum(-1)


def fit_arch_form(
    points: Array,
    weights: Array,
    front_y: float,
    anchor_points: dict[int, Array],
    blend_points: Array,
) -> ArchForm:
    """Fit an ArchForm through `front_y` (the arch front at x = 0, i.e. the current incisors,
    so that incisor advancement is measured from where they are) to xy contact points by
    robust orthogonal least squares: Tukey weights drop displaced teeth, so they are brought
    onto the arch of the others. `anchor_points` (+1/-1 side of x) are held-tooth contacts the
    arch must meet; `blend_points` (the canines) mark where each side may start bending
    toward them."""
    side = np.where(points[:, 0] >= 0, 1, -1)
    half_pts = np.stack([np.abs(points[:, 0]), points[:, 1]], axis=1)
    anchors = {sd: np.array([abs(p[0]), p[1]]) for sd, p in anchor_points.items()}
    nk = len(ARCH_KNOTS_MM)
    robust = np.ones(len(points))

    def build(v: Array, s_blend: float = 0.0, s_anchor: float = 0.0) -> ArchForm:
        bend = {1: float(v[nk]), -1: float(v[nk + 1])} if len(v) > nk else {}
        return ArchForm(front_y, np.asarray(v[:nk]), bend, s_blend, s_anchor)

    def distances(f: ArchForm) -> Array:
        res = np.empty(len(points))
        for sd in (1, -1):
            if (side == sd).any():
                res[side == sd] = _project_half(half_pts[side == sd], f.half(sd)[1])[1]
        return res

    def fit(v: Array, *spans: float) -> Array:
        nonlocal robust

        def residuals(u: Array) -> Array:
            f = build(u, *spans)
            out = [np.sqrt(weights * robust) * distances(f)]
            for sd, p in anchors.items() if len(u) > nk else ():
                out.append(ARCH_ANCHOR_WEIGHT * _project_half(p[None], f.half(sd)[1])[1])
            k = u[:nk]
            out += [ARCH_CURVATURE_PENALTY * np.diff(k, 2), ARCH_SLOPE_PENALTY * np.diff(k)]
            out.append(ARCH_BEND_PENALTY * u[nk:])
            return np.concatenate(out)

        # curvature >= 0: an arch never turns concave (outward) at its front
        bounds = (np.r_[np.zeros(nk), np.full(len(v) - nk, -np.inf)], np.full(len(v), np.inf))
        v = np.clip(v, bounds[0], None)
        for _ in range(ARCH_IRLS_PASSES):
            v = least_squares(residuals, v, bounds=bounds).x
            res = distances(build(v, *spans))
            robust = np.clip(1 - (res / (4.685 * ARCH_ROBUST_SCALE_MM)) ** 2, 0, 1) ** 2 + 1e-3
        return v

    v = fit(np.interp(ARCH_KNOTS_MM, [0.0, 20.0, 40.0], [0.05, 0.05, 0.0]))
    if not anchors:
        return build(v)
    symmetric = build(v)

    def arc(p: Array) -> float:
        sd = 1 if p[0] >= 0 else -1
        q = np.array([[abs(p[0]), p[1]]])
        return float(_project_half(q, symmetric.half(sd)[1])[0][0] * ARCH_STEP_MM)

    s_anchor = float(np.mean([arc(p) for p in anchor_points.values()]))
    blend = [arc(p) for p in blend_points]
    s_blend = min(float(np.mean(blend)) if blend else 0.5 * s_anchor, 0.8 * s_anchor)
    v = fit(np.r_[v, 0.0, 0.0], s_blend, s_anchor)
    return build(v, s_blend, s_anchor)


def ideal_arch_form(
    teeth: dict[int, "Tooth"], widen: dict[int, float] | None = None, held: set[int] | None = None
) -> ArchForm:
    """Ideal arch through the teeth's contact points: symmetric, smooth, through the current
    central incisors, meeting the held teeth (default: first and second molars). `widen` adds
    per-position transverse offsets."""
    widen = widen or {}
    held = held if held is not None else {n for n in teeth if fdi.position(n) >= 6}
    pts, wts = [], []
    anchor: dict[int, list[Array]] = {1: [], -1: []}
    canines = []
    for n, t in teeth.items():
        for q in (t.mesial_contact, t.distal_contact):
            p = q[:2].copy()
            p[0] += np.sign(p[0]) * widen.get(fdi.position(n), 0.0)
            pts.append(p)
            wts.append(4.0 if n in held else 1.0)
        if n in held:
            side = 1 if t.mesial_contact[0] >= 0 else -1
            p = t.mesial_contact[:2].copy()
            p[0] += np.sign(p[0]) * widen.get(fdi.position(n), 0.0)
            anchor[side].append(p)
        if fdi.position(n) == 3:
            canines.append(t.distal_contact[:2])
    # the most mesial held contact per side
    front = {side: max(v, key=lambda p: p[1]) for side, v in anchor.items() if v}
    centrals = [t.mesial_contact[1] for n, t in teeth.items() if fdi.position(n) == 1]
    front_y = float(np.mean(centrals)) if centrals else float(np.max(np.array(pts)[:, 1]))
    return fit_arch_form(np.array(pts), np.array(wts), front_y, front, np.array(canines))


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
    """Frame and landmarks of one tooth, all intrinsic to the tooth.

    The mesiodistal axis m of incisors is their incisal edge and of canines their cusp-ridge
    crest, both intrinsic to the crown: where a crowded canine touches its neighbors says
    little about how it is turned. Premolars and molars, whose cusps give no single crest, run
    m between the contacts with the neighbors (`contacts`), or along the arch at the arch ends.
    Contact points and width come from the crown's own mesial and distal extremes along m, so
    overlapping (crowded) neighbors do not distort them."""
    V = np.asarray(patch.vertices)
    center = _weighted_centroid(patch)
    o = ct_axis / np.linalg.norm(ct_axis) if ct_axis is not None else np.array([0.0, 0.0, 1.0])
    x_sign = patient_right_sign(jaw) if fdi.is_patient_right(number) else -patient_right_sign(jaw)
    s_here = float(curve.project(center[:2])[0])
    along_arch = -x_sign * curve.tangent(s_here)
    solid = close_crown(patch, o)
    pos = fdi.position(number)
    between = (
        contacts["mesial"] - contacts["distal"]
        if "mesial" in contacts and "distal" in contacts
        else along_arch
    )
    if pos <= 2:
        m = incisal_axis(V, o)
    elif pos == 3:
        m = cusp_ridge_axis(V, o, between)
    else:
        m = between
    m = m - np.dot(m, o) * o
    if np.dot(m, along_arch) < 0:
        m = -m
    m /= np.linalg.norm(m)
    b = np.asarray(np.cross(o, m), dtype=np.float64)
    if np.dot(b, curve.outward(s_here)) < 0:
        b = -b

    loops = boundary_loops(np.asarray(patch.faces, dtype=np.int64))
    margin = V[loops[0]] if loops else V
    margin_center = margin.mean(axis=0)

    h = (V - center) @ o
    hmin, hmax = float(h.min()), float(h.max())
    upper_part = V[h >= hmin + CONTACT_HEIGHT_FRACTION * (hmax - hmin)]
    along = (upper_part - center) @ m
    # Robust extremes (the segmented patch can reach a fraction of a mm onto the neighbor).
    # The contact point is constructed at the center of the mesial/distal face: extreme along
    # m, the crown's buccolingual midline along b, and the face's mean height along o.
    hi_q, lo_q = np.quantile(along, 0.99), np.quantile(along, 0.01)
    across = (upper_part - center) @ b
    mid_b = 0.5 * (np.quantile(across, 0.98) + np.quantile(across, 0.02))

    def face_point(extreme: float, near: NDArray[np.bool_]) -> Array:
        height = float(((upper_part[near] - center) @ o).mean())
        return center + extreme * m + mid_b * b + height * o

    mesial = face_point(float(hi_q), along >= hi_q - 0.4)
    distal = face_point(float(lo_q), along <= lo_q + 0.4)
    width = float(hi_q - lo_q)
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


def incisal_axis(vertices: Array, o: Array) -> Array:
    """Direction of the incisal edge / cusp ridges: the long axis of the top fifth of the crown
    seen along the tooth axis (unsigned)."""
    h = vertices @ o
    top = vertices[h >= h.max() - 0.2 * (h.max() - h.min())]
    flat = top - np.outer((top - top.mean(axis=0)) @ o, o)
    return pca(flat)[2][:, 0]


def cusp_ridge_axis(vertices: Array, o: Array, m: Array) -> Array:
    """Direction of a canine's mesial and distal cusp ridges seen along the tooth axis: the
    line through the crest (highest crown points) of each mesiodistal slice, refined from the
    start direction `m` (unsigned)."""
    h = vertices @ o
    for _ in range(RIDGE_PASSES):
        m = m - np.dot(m, o) * o
        m = m / np.linalg.norm(m)
        s = vertices @ m
        bins = np.linspace(float(np.percentile(s, 8)), float(np.percentile(s, 92)), RIDGE_BINS + 1)
        crest = []
        for a, b in zip(bins[:-1], bins[1:], strict=True):
            sel = (s >= a) & (s < b)
            if sel.sum() >= 5:
                crest.append(vertices[sel][h[sel] >= np.quantile(h[sel], 0.9)].mean(axis=0))
        if len(crest) < 3:
            break
        C = np.array(crest)
        axis = pca(C - np.outer((C - C.mean(axis=0)) @ o, o))[2][:, 0]
        m = axis if np.dot(axis, m) >= 0 else -axis
    return m


def incisal_edge(vertices: Array, m: Array, o: Array) -> Array:
    """Direction of the visible incisal edge: the crest (highest crown points along o) across
    the crown's mesiodistal extent, fitted as a line in the m-o plane; returned along +m."""
    s, h = vertices @ m, vertices @ o
    lo, hi = np.percentile(s, [15, 85])
    bins = np.linspace(float(lo), float(hi), 9)
    xs, crest = [], []
    for a, b in zip(bins[:-1], bins[1:], strict=True):
        sel = (s >= a) & (s < b)
        if sel.any():
            xs.append(0.5 * (a + b))
            crest.append(h[sel].max())
    slope = float(np.polyfit(xs, crest, 1)[0]) if len(xs) >= 2 else 0.0
    e = m + slope * o
    return e / np.linalg.norm(e)


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

    def above(self, pts: Array) -> Array:
        """Negative above the gingiva surface (the socket under a crown counts at its rim's
        level, not sunk); everywhere outside the gingiva."""
        z, _ = self.sample(np.asarray(pts)[:, :2])
        return np.where(np.isfinite(z), z + GUM_SOCKET_DROP_MM - np.asarray(pts)[:, 2], -1.0)

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
    gum = gum_heightfield(mesh, labels)
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
    # the scan misses enamel where neighbors hid it: complete it from the CT tooth fitted to
    # the scan, above the gingiva for contacts, down to the stub depth for fabrication; where
    # the scan saw a neighbor, or the neighbor's CT tooth reaches deeper, this tooth is not
    fits = {n: fit_ct_tooth(ct[n][1], patches[n]) for n in numbers if ct and n in ct}
    scanned = {n: MeshSDF.of(close_crown(p, axes[n], stub=False)) for n, p in patches.items()}
    fitted = {n: MeshSDF.of(f) for n, f in fits.items()}
    for n in numbers:
        ct_axis, ct_mesh = ct[n] if ct and n in ct else (None, None)
        tooth, crown = build_tooth(n, jaw, patches[n], curve, ct_axis, ct_mesh, contacts[n])
        expected = fdi.CROWN_WIDTH_MM[jaw][fdi.position(n)]
        if not 0.6 * expected <= tooth.width <= 1.4 * expected:
            warnings.append(f"tooth {n}: width {tooth.width:.1f} mm, expected ~{expected:.1f} mm")
        solid = close_crown(patches[n], tooth.o, stub=False)
        if n in fits:
            V = np.asarray(patches[n].vertices)
            loops = boundary_loops(np.asarray(patches[n].faces, dtype=np.int64))
            margin = V[loops[0]] if loops else V
            floor = margin[np.argmin(margin @ tooth.o)] - STUB_DEPTH_MM * tooth.o
            i = order.index(n)
            neighbors = [
                Neighbor(scanned[m], fitted.get(m)) for m in order[max(i - 1, 0) : i + 2] if m != n
            ]
            crown, solid = complete_crowns(
                patches[n], crown, solid, fits[n], below_plane(floor, tooth.o), gum.above, neighbors
            )
        if ct_mesh is not None:
            write_mesh(out_dir / f"{n}_root.ply", ct_mesh)
        write_mesh(out_dir / f"{n}_crown.ply", crown)
        write_mesh(out_dir / f"{n}_solid.ply", solid)
        write_mesh(out_dir / f"{n}_patch.ply", patches[n])
        teeth.append(tooth.as_dict())
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
    contact_solids: dict[int, trimesh.Trimesh]  # crowns closed at the margin, no stub

    @classmethod
    def load(cls, directory: Path, jaw: Jaw) -> "JawAnatomy":
        info = read_json(directory / "teeth.json")
        teeth = {t["fdi"]: Tooth.from_dict(t) for t in info["teeth"]}

        def meshes(kind: str) -> dict[int, trimesh.Trimesh]:
            paths = {n: directory / f"{n}_{kind}.ply" for n in teeth}
            return {n: read_mesh(p) for n, p in paths.items() if p.exists()}

        return cls(
            jaw,
            ArchCurve.from_dict(info["curve"]),
            teeth,
            meshes("crown"),
            meshes("root"),
            GumField.load(directory / "gum.npz"),
            meshes("patch"),
            meshes("solid"),
        )

    def order(self) -> list[int]:
        return fdi.arch_order(list(self.teeth))

    def enamel_samples(self, number: int, count: int = 1200, seed: int = 0) -> Array:
        """Samples of the crown surface, scanned and hidden (contacts and collisions are judged
        on these)."""
        surface = self.patches[number]
        if number in self.hidden:
            surface = cast(
                trimesh.Trimesh, trimesh.util.concatenate([surface, self.hidden[number]])
            )
        return sample_surface(surface, count, seed=seed)[0]

    @cached_property
    def hidden(self) -> dict[int, trimesh.Trimesh]:
        """Crown surface above the gingiva the scan missed: the completed crown away from the
        scanned surface."""
        out = {}
        for n, solid in self.contact_solids.items():
            patch = self.patches[n]
            centers = np.ascontiguousarray(solid.triangles_center, dtype=np.float64)
            d2, *_ = igl.point_mesh_squared_distance(
                centers,
                np.ascontiguousarray(patch.vertices, dtype=np.float64),
                np.ascontiguousarray(patch.faces, dtype=np.int64),
            )
            keep = (np.sqrt(d2) > HIDDEN_CLEAR_MM) & (self.gum.above(centers) < -SOLID_RES_MM)
            if keep.any():
                out[n] = submesh(solid, keep)
        return out
