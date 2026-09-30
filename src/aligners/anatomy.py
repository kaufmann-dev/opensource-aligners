"""Tooth solids, anatomical frames, landmarks and the gingiva heightfield (arch frame)."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import ndimage, sparse
from scipy.interpolate import make_smoothing_spline, splev, splprep
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree

from aligners import fdi
from aligners.case import CaseError, read_json, read_mesh, write_json, write_mesh
from aligners.fdi import Jaw
from aligners.geometry import Array, boundary_loops, pca, sample_surface
from aligners.scan import patient_right_sign

STUB_DEPTH_MM = 3.0
STUB_TAPER = 0.15
CR_ROOT_FRACTION = 0.4
GUM_RES_MM = 0.1
GUM_SOCKET_DROP_MM = 1.0
CONTACT_HEIGHT_FRACTION = 0.4  # contacts are searched in the top 60 % of the crown
ARCH_SMOOTHING = 1e-2  # smoothing-spline penalty of the ideal arch r(theta)
ARCH_ROBUST_SCALE_MM = 2.5  # residual scale for down-weighting grossly displaced teeth
ARCH_INSIDE_WEIGHT = 0.1  # weight of points inside the arch (the arch hugs the outer teeth)


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


def _smoothstep(t: Array) -> Array:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class ArchForm:
    """Ideal arch as a smooth polar curve r(theta) around a point inside the arch.

    theta = 0 points anteriorly (+y), positive toward +x. The base form is symmetric and
    smooth, fitted as an outer envelope of the teeth (lingually displaced teeth are brought
    out to it); toward the held back teeth it blends into each side's actual position
    (`side_offset`). Beyond the last teeth the radius is held constant."""

    center_y: float
    base: Any  # r(theta): symmetric smoothing spline
    theta_blend: float  # |theta| where blending toward the held teeth starts
    theta_anchor: float  # |theta| of the held teeth (0 if none)
    side_offset: dict[int, float]  # +1 / -1 side of theta: radial correction at the held teeth
    theta_max: float

    def radius(self, theta: Array, delta: float = 0.0) -> Array:
        r = self.base(np.clip(theta, -self.theta_max, self.theta_max))
        if self.theta_anchor > self.theta_blend:
            beta = _smoothstep(
                (np.abs(theta) - self.theta_blend) / (self.theta_anchor - self.theta_blend)
            )
            r = r + beta * np.where(
                theta >= 0, self.side_offset.get(1, 0.0), self.side_offset.get(-1, 0.0)
            )
            fade = 1.0 - _smoothstep(np.abs(theta) / self.theta_anchor)
        else:
            fade = np.ones_like(theta)
        return r + delta * fade

    def curve(self, delta: float = 0.0) -> ArchCurve:
        theta = np.linspace(-self.theta_max - 0.2, self.theta_max + 0.2, 1500)
        r = self.radius(theta, delta)
        pts = np.stack([r * np.sin(theta), self.center_y + r * np.cos(theta)], axis=1)
        return ArchCurve(pts, smoothing_mm=0.005)


def fit_arch_form(
    points: Array, weights: Array, anchor_points: dict[int, Array], blend_points: Array
) -> ArchForm:
    """Fit an ArchForm to xy points: a symmetric penalized smoothing spline r(|theta|) with
    robust reweighting, so displaced or rotated teeth do not bend the ideal arch.
    `anchor_points` (+1/-1 side of x) are held-tooth contacts the curve must meet;
    `blend_points` (e.g. the canines) mark where blending toward them starts."""
    center_y = float(np.mean(points[:, 1]))

    def polar(p: Array) -> tuple[Array, Array]:
        return np.arctan2(p[:, 0], p[:, 1] - center_y), np.hypot(p[:, 0], p[:, 1] - center_y)

    theta, r = polar(points)
    th = np.r_[theta, -theta]
    rr = np.r_[r, r]
    ww = np.r_[weights, weights]
    order = np.argsort(th)
    th, rr, ww = th[order], rr[order], ww[order]
    keep = np.r_[True, np.diff(th) > 1e-4]
    th, rr, ww = th[keep], rr[keep], ww[keep]
    # Outer envelope: points inside the curve (lingually displaced teeth) get little weight, so
    # the arch follows the outermost teeth and crowding is resolved by bringing teeth out, not
    # by pulling well-positioned ones in. Tukey weights still drop gross outliers.
    spline = make_smoothing_spline(th, rr, w=ww, lam=ARCH_SMOOTHING)
    for _ in range(6):
        res = rr - spline(th)
        robust = np.clip(1 - (res / (4.685 * ARCH_ROBUST_SCALE_MM)) ** 2, 0, 1) ** 2 + 1e-3
        robust = np.where(res < 0, robust * ARCH_INSIDE_WEIGHT, robust)
        spline = make_smoothing_spline(th, rr, w=ww * robust, lam=ARCH_SMOOTHING)
    form = ArchForm(center_y, spline, 0.0, 0.0, {}, float(np.abs(theta).max()))
    anchors = {side: pts for side, pts in anchor_points.items() if len(pts)}
    if anchors:
        th_anchor = float(np.mean([np.abs(polar(pts)[0]).mean() for pts in anchors.values()]))
        th_blend = (
            float(np.abs(polar(blend_points)[0]).mean()) if len(blend_points) else 0.5 * th_anchor
        )
        offsets = {}
        for side, pts in anchors.items():
            t_a, r_a = polar(pts)
            offsets[side] = float(np.mean(r_a - form.radius(t_a)))
        form = ArchForm(
            center_y, spline, min(th_blend, 0.9 * th_anchor), th_anchor, offsets, form.theta_max
        )
    return form


def ideal_arch_form(
    teeth: dict[int, "Tooth"], widen: dict[int, float] | None = None, held: set[int] | None = None
) -> ArchForm:
    """Ideal arch through the teeth's contact points: symmetric, smooth, meeting the held teeth
    (default: first and second molars). `widen` adds per-position transverse offsets."""
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
    front = {side: np.array(v) for side, v in anchor.items() if v}
    # use the most mesial held contact per side
    for side, v in front.items():
        front[side] = v[np.argmax(v[:, 1])][None]
    return fit_arch_form(np.array(pts), np.array(wts), front, np.array(canines))


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

    The mesiodistal axis m of incisors is their incisal edge; of canines, perpendicular to the
    labial surface; of premolars and molars it runs between the contacts with the neighbors
    (`contacts`), or along the arch at the arch ends. Contact points and width come from the
    crown's own mesial and distal extremes along m, so overlapping (crowded) neighbors do not
    distort them."""
    V = np.asarray(patch.vertices)
    center = _weighted_centroid(patch)
    o = ct_axis / np.linalg.norm(ct_axis) if ct_axis is not None else np.array([0.0, 0.0, 1.0])
    x_sign = patient_right_sign(jaw) if fdi.is_patient_right(number) else -patient_right_sign(jaw)
    s_here = float(curve.project(center[:2])[0])
    along_arch = -x_sign * curve.tangent(s_here)
    solid = close_crown(patch, o)
    pos = fdi.position(number)
    if pos <= 2:
        m = incisal_axis(V, o)
    elif pos == 3:
        # canines are near-round in occlusal view: perpendicular to the labial surface
        m = np.cross(o, facial_normal(solid, len(patch.faces), o, curve.outward(s_here)))
    elif "mesial" in contacts and "distal" in contacts:
        m = contacts["mesial"] - contacts["distal"]
    else:
        m = along_arch
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


def facial_normal(solid: trimesh.Trimesh, n_patch_faces: int, o: Array, outward: Array) -> Array:
    """Mean outward normal of the facial (buccal/labial) crown surface, perpendicular to o.

    Starts from the arch's outward direction and re-centers on the facial faces a few times,
    so a rotated tooth converges to its own facial direction. `solid` must be the closed
    crown (consistent outward normals) whose first faces are the scanned patch."""
    normals = np.asarray(solid.face_normals)[:n_patch_faces]
    centers = np.asarray(solid.triangles_center)[:n_patch_faces]
    area = np.asarray(solid.area_faces)[:n_patch_faces]
    h = centers @ o
    lo, hi = np.quantile(h, 0.1), h.max()
    band = (h > lo + 0.25 * (hi - lo)) & (h < lo + 0.8 * (hi - lo))
    flat = normals - np.outer(normals @ o, o)
    b = outward - np.dot(outward, o) * o
    b /= np.linalg.norm(b)
    for _ in range(12):
        sel = band & (flat @ b > np.cos(np.radians(55)) * np.linalg.norm(flat, axis=1))
        if sel.sum() < 10:
            break
        new = (flat[sel] * area[sel, None]).sum(axis=0)
        new /= np.linalg.norm(new)
        if np.dot(new, b) > 0.99999:
            b = new
            break
        b = new
    return b


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
