"""Orthognathic surgery: both jaws in the head frame, dental landmarks, the surgeon's landmark
movement table, the ideal occlusion, Le Fort I / BSSO movements and the bone segments.

Head frame: +x patient right, +y anterior, +z superior (see `skull.py`). A jaw movement M is a
rigid 4x4 in the head frame; post-op, a jaw's arch frame maps to the head by Q = M @ P, where P
is its pre-op placement. Movement tables give landmark displacements in mm: forward_mm
(+ anterior), up_mm (+ superior), left_mm (+ toward the patient's left), measured at the scanned
(pre-treatment) landmark positions.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares
from scipy.spatial.transform import Rotation

from aligners import fdi
from aligners.analyze import CLASS_I_MOLAR_OFFSET_MM, arch_to_bite, posed_teeth
from aligners.anatomy import JawAnatomy
from aligners.case import LANDMARKS, Case, CaseError, read_json, read_mesh, write_json, write_mesh
from aligners.ct import Volume, mask_mesh
from aligners.fdi import JAWS, Jaw
from aligners.fields import Grid, GridSDF
from aligners.geometry import (
    Array,
    MeshSDF,
    rigid_fit,
    rotation_about,
    transform_points,
    transform_vectors,
    translation,
)
from aligners.skull import ct_to_head, load_bone, load_skull

BONE = {"upper": "maxilla", "lower": "mandible"}
JAW_OF = {"maxilla": "upper", "mandible": "lower"}
FIT_WARN_MM = 0.3
FIT_ERROR_MM = 1.5
LEFORT_ABOVE_APEX_MM = 5.0
BSSO_DISTAL_MM = 3.0
SPLIT_FACE_MM = 1.5
SEGMENT_FACES = 60000
CRANIUM_EXCLUSION_MM = 2.0
SDF_RES_MM = 0.3
SEAT_TOLERANCE_MM = 0.05
LIFT_MM = 2.0  # the upper arch starts the in-plane fit opened by this much
SEAT_REACH_MM = 6.0  # occlusal signed distances are resolved (with a gradient) up to this far
SOFTMIN_MM = 0.1


@dataclass
class HeadFrames:
    ct_to_head: Array
    P: dict[Jaw, Array]  # arch frame -> head, pre-op (lower in the case's bite)
    bone: dict[Jaw, Array]  # CT world -> head of each jaw's bone, pre-op
    ct_to_arch: dict[Jaw, Array]
    head_pitch_deg: float


def _incisor_midline(anatomy: JawAnatomy, poses: dict[int, Array] | None = None) -> Array | None:
    teeth = posed_teeth(anatomy, poses)
    centrals = [fdi.fdi_number(anatomy.jaw, side, 1) for side in (True, False)]
    if all(c in teeth for c in centrals):
        return (teeth[centrals[0]].cusp + teeth[centrals[1]].cusp) / 2
    return None


def head_frames(case: Case, plan: dict[str, Any], anatomies: dict[Jaw, JawAnatomy]) -> HeadFrames:
    to_bite = arch_to_bite(case)
    if to_bite is None:
        raise CaseError("surgical modes need the bite (bite-registered scans or a CT)")
    frames = read_json(case.work / "frames.json")["jaws"]
    ct_to_arch: dict[Jaw, Array] = {
        jaw: np.asarray(frames[jaw]["scan_to_arch"], float)
        @ np.asarray(frames[jaw]["ct_to_scan"], float)
        for jaw in JAWS
    }
    upper = anatomies["upper"]
    origin_arch = _incisor_midline(upper)
    if origin_arch is None:
        origin_arch = np.mean([t.cusp for t in upper.teeth.values()], axis=0)
    origin_ct = transform_points(np.linalg.inv(ct_to_arch["upper"]), origin_arch[None])[0]
    pitch = float(plan["surgery"]["head_pitch_deg"])
    H = ct_to_head(load_skull(case), pitch, origin_ct)
    P_upper = H @ np.linalg.inv(ct_to_arch["upper"])
    P_lower = P_upper @ np.linalg.inv(to_bite["upper"]) @ to_bite["lower"]
    return HeadFrames(
        ct_to_head=H,
        P={"upper": P_upper, "lower": P_lower},
        bone={"upper": H, "lower": P_lower @ ct_to_arch["lower"]},
        ct_to_arch=ct_to_arch,
        head_pitch_deg=pitch,
    )


# -- landmarks ----------------------------------------------------------------------------------
def mesiobuccal_cusp(anatomy: JawAnatomy, n: int) -> Array:
    """Highest crown point (along the long axis) in the mesiobuccal quarter of the crown."""
    t = anatomy.teeth[n]
    V = np.asarray(anatomy.patches[n].vertices)
    rel = V - t.center
    sel = (rel @ t.m > 0) & (rel @ t.b > 0)
    pts = V[sel] if sel.sum() >= 10 else V
    return pts[np.argmax(pts @ t.o)]


def dental_landmarks(
    anatomy: JawAnatomy, poses: dict[int, Array] | None = None
) -> dict[str, Array]:
    """Named landmarks of one arch in its arch frame (posed if `poses` is given)."""
    poses = poses or {}
    jaw = anatomy.jaw
    teeth = posed_teeth(anatomy, poses)
    out: dict[str, Array] = {}
    mid = _incisor_midline(anatomy, poses)
    if mid is not None:
        out["incisor_midline"] = mid
    for right, name in ((True, "right"), (False, "left")):
        canine = fdi.fdi_number(jaw, right, 3)
        if canine in teeth:
            out[f"canine_{name}"] = teeth[canine].cusp
        molar = fdi.fdi_number(jaw, right, 6)
        if molar in teeth:
            p = mesiobuccal_cusp(anatomy, molar)
            out[f"molar_{name}"] = transform_points(poses.get(molar, np.eye(4)), p[None])[0]
    return out


def to_head(T: Array, landmarks: dict[str, Array]) -> dict[str, Array]:
    return {k: transform_points(T, v[None])[0] for k, v in landmarks.items()}


def mandible_landmarks(case: Case, hf: HeadFrames, lower: JawAnatomy) -> dict[str, Array]:
    """Approximate skeletal landmarks of the mandible in the pre-op head frame."""
    mesh = read_mesh(case.require(case.ct_dir / "jawbone_lower.ply", "ct"))
    V = transform_points(hf.bone["lower"], np.asarray(mesh.vertices))
    P = hf.P["lower"]
    teeth = lower.teeth
    lm = to_head(P, dental_landmarks(lower))
    cusps = {n: transform_points(P, t.cusp[None])[0] for n, t in teeth.items()}
    x_mid = (
        lm["incisor_midline"][0]
        if "incisor_midline" in lm
        else float(np.mean([c[0] for c in cusps.values()]))
    )
    y_front = (
        lm["incisor_midline"][1]
        if "incisor_midline" in lm
        else float(max(c[1] for c in cusps.values()))
    )
    out: dict[str, Array] = {}
    for right, name in ((True, "right"), (False, "left")):
        side = [n for n in teeth if fdi.is_patient_right(n) == right]
        if not side:
            continue
        y_back = min(transform_points(P, teeth[n].distal_contact[None])[0][1] for n in side)
        sign = 1.0 if right else -1.0
        pts = V[(sign * (V[:, 0] - x_mid) > 0) & (V[:, 1] < y_back - 3.0)]
        if len(pts):
            out[f"condylion_{name}"] = pts[np.argmax(pts[:, 2])]
            out[f"gonion_{name}"] = pts[np.argmax(-pts[:, 1] - pts[:, 2])]
    apex = [
        transform_points(P, np.asarray(lower.roots[n].vertices)).min(axis=0)[2]
        for n in teeth
        if fdi.position(n) == 1 and n in lower.roots
    ]
    top = max(c[2] for c in cusps.values())
    apex_z = min(apex) if apex else top - 20.0
    # the symphysis below the incisor roots; the chin midline is the middle of its most
    # anterior 4 mm (robust where the lower border is flat, and free to deviate from the teeth)
    sym = V[(np.abs(V[:, 0] - x_mid) < 25.0) & (V[:, 2] < apex_z) & (V[:, 1] > y_front - 25.0)]
    if len(sym):
        anterior = sym[sym[:, 1] > sym[:, 1].max() - 4.0]
        chin_x = 0.5 * (anterior[:, 0].min() + anterior[:, 0].max())
        band = sym[np.abs(sym[:, 0] - chin_x) < 2.0]
        out["pogonion"] = band[np.argmax(band[:, 1])]
        out["menton"] = band[np.argmin(band[:, 2])]
    return out


def mandibular_plane_normal(mand: dict[str, Array]) -> Array | None:
    """Upward normal of the plane through menton and both gonia (head frame)."""
    if not {"menton", "gonion_right", "gonion_left"} <= set(mand):
        return None
    n = np.cross(mand["gonion_right"] - mand["menton"], mand["gonion_left"] - mand["menton"])
    n /= np.linalg.norm(n)
    return n if n[2] > 0 else -n


# -- movement tables and angles -----------------------------------------------------------------
def table_vector(move: dict[str, float]) -> Array:
    return np.array([-move["left_mm"], move["forward_mm"], move["up_mm"]], dtype=np.float64)


def vector_table(d: Array) -> dict[str, float]:
    return {
        "forward_mm": round(float(d[1]), 1) + 0.0,
        "up_mm": round(float(d[2]), 1) + 0.0,
        "left_mm": round(float(-d[0]), 1) + 0.0,
    }


def table_to_transform(
    table: dict[str, dict[str, float]], pre: dict[str, Array], bone: str
) -> tuple[Array, dict[str, float]]:
    """Rigid head-frame movement from landmark displacements, and the per-landmark residuals."""
    if not table:
        return np.eye(4), {}
    missing = [name for name in table if name not in pre]
    if missing:
        raise CaseError(f"surgery.{bone}: landmark(s) {', '.join(missing)} missing (no tooth)")
    if len(table) == 2:
        raise CaseError(
            f"surgery.{bone}: give one landmark (pure translation) or at least three, not two"
        )
    names = list(table)
    src = np.array([pre[n] for n in names])
    if len(names) >= 3 and np.linalg.svd(src - src.mean(axis=0), compute_uv=False)[1] < 2.0:
        raise CaseError(f"surgery.{bone}: the given landmarks are (nearly) on one line")
    dst = src + np.array([table_vector(table[n]) for n in names])
    T, residuals = rigid_fit(src, dst)
    res = {n: float(r) for n, r in zip(names, residuals, strict=True)}
    bad = {n: r for n, r in res.items() if r > FIT_ERROR_MM}
    if bad:
        detail = ", ".join(f"{n} {r:.1f} mm" for n, r in bad.items())
        raise CaseError(
            f"surgery.{bone}: the displacements are not one rigid movement ({detail} off); "
            "check signs and left/right"
        )
    return T, res


def transform_to_table(M: Array, pre: dict[str, Array]) -> dict[str, dict[str, float]]:
    return {
        name: vector_table(transform_points(M, pre[name][None])[0] - pre[name])
        for name in LANDMARKS
        if name in pre
    }


def jaw_angles(lm: dict[str, Array]) -> dict[str, float]:
    """Occlusal pitch (+ anterior up), cant (+ patient-left side up), yaw (+ incisors toward
    patient left) and midline (+ patient left) of one arch from head-frame landmarks."""
    out: dict[str, float] = {}
    inc = lm.get("incisor_midline")
    if inc is not None:
        out["midline_mm"] = float(-inc[0])
    pairs = [k for k in ("molar", "canine") if f"{k}_right" in lm and f"{k}_left" in lm]
    for k in pairs:
        out[f"cant_{k}s_mm"] = float(lm[f"{k}_left"][2] - lm[f"{k}_right"][2])
    if pairs:
        r, left = lm[f"{pairs[0]}_right"], lm[f"{pairs[0]}_left"]
        d = left - r
        out["cant_deg"] = float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))
        if inc is not None:
            a = inc - (r + left) / 2
            out["occlusal_pitch_deg"] = float(np.degrees(np.arctan2(a[2], a[1])))
            out["yaw_deg"] = float(np.degrees(np.arctan2(-a[0], a[1])))
    return out


def sagittal_angle_deg(a: Array, b: Array) -> float:
    """Angle between two directions projected onto the head's sagittal (y-z) plane."""
    u, v = np.array([a[1], a[2]]), np.array([b[1], b[2]])
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def incisor_angles(
    anatomies: dict[Jaw, JawAnatomy],
    T: dict[Jaw, Array],
    mand: dict[str, Array],
    poses: dict[Jaw, dict[int, Array]] | None = None,
) -> dict[str, float]:
    """Upper incisor axis to the head horizontal, and lower incisor axis to the mandibular plane
    (IMPA); both measured between the crownward axis and the posterior reference direction."""
    poses = poses or {}
    out: dict[str, float] = {}
    for jaw in JAWS:
        teeth = posed_teeth(anatomies[jaw], poses.get(jaw))
        axes = [
            transform_vectors(T[jaw], teeth[n].o[None])[0] for n in teeth if fdi.position(n) == 1
        ]
        if not axes:
            continue
        o = np.mean(axes, axis=0)
        if jaw == "upper":
            out["upper_incisor_deg"] = sagittal_angle_deg(o, np.array([0.0, -1.0, 0.0]))
        elif mandibular_plane_normal(mand) is not None:
            back = (mand["gonion_right"] + mand["gonion_left"]) / 2 - mand["menton"]
            out["impa_deg"] = sagittal_angle_deg(o, back)
    return out


def incisor_target_axes(
    anatomies: dict[Jaw, JawAnatomy],
    T: dict[Jaw, Array],
    mp_normal: Array | None,
    cfg: dict[str, Any],
) -> dict[Jaw, dict[int, Array]]:
    """Decompensated crownward incisor axes in each arch frame: o = cos(t-90) n + sin(t-90) b,
    n = upright reference (upper: head down; lower: mandibular-plane normal), b = labial."""
    out: dict[Jaw, dict[int, Array]] = {}
    for jaw in JAWS:
        if jaw == "upper":
            n_head, theta = np.array([0.0, 0.0, -1.0]), cfg["upper_incisor_deg"]
        elif mp_normal is not None:
            n_head, theta = mp_normal, cfg["lower_incisor_deg"]
        else:
            continue
        n = transform_vectors(np.linalg.inv(T[jaw]), n_head[None])[0]
        n /= np.linalg.norm(n)
        a = np.radians(theta - 90.0)
        axes = {}
        for num, t in anatomies[jaw].teeth.items():
            if fdi.position(num) > 2:
                continue
            b = t.b - (t.b @ n) * n
            b /= np.linalg.norm(b)
            axes[num] = np.cos(a) * n + np.sin(a) * b
        out[jaw] = axes
    return out


# -- ideal occlusion ----------------------------------------------------------------------------
def _band_sdf(mesh: trimesh.Trimesh, z_lo: float, z_hi: float) -> GridSDF:
    exact = MeshSDF.of(mesh)
    bound = SEAT_REACH_MM
    lo = np.r_[exact.lo[:2] - bound, z_lo]
    hi = np.r_[exact.hi[:2] + bound, z_hi]
    grid = Grid.around(lo, hi, SDF_RES_MM)
    values = np.empty(grid.shape, dtype=np.float32)
    for i0 in range(0, grid.shape[0], 16):
        i1 = min(grid.shape[0], i0 + 16)
        pts = grid.points(slice(i0, i1))
        values[i0:i1] = exact(pts, bound=bound).reshape(i1 - i0, grid.shape[1], grid.shape[2])
    return GridSDF(grid.origin, SDF_RES_MM, values, bound)


def _labial_profile(points: Array) -> tuple[Array, Array]:
    """Most labial y of the lower incisors per height z (smooth overjet reference)."""
    z = points[:, 2]
    edges = np.arange(z.min(), z.max() + 0.25, 0.25)
    centers, ys = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        sel = (z >= a) & (z < b)
        if sel.any():
            centers.append(0.5 * (a + b))
            ys.append(points[sel, 1].max())
    return np.array(centers), np.array(ys)


def _correction(x: Array) -> Array:
    C = np.eye(4)
    C[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    C[:3, 3] = x[3:]
    return C


class Occlusion:
    """Finds the upper->lower arch transform of the ideal occlusion of two posed dentitions:
    Class I molars and canines, the target overjet, coinciding midlines, a centered
    transverse relation, and seated (touching, not penetrating) posterior teeth."""

    def __init__(
        self,
        upper: JawAnatomy,
        up_poses: dict[int, Array],
        lower: JawAnatomy,
        low_poses: dict[int, Array],
        overjet_mm: float,
    ):
        self.overjet = overjet_mm
        self.U = posed_teeth(upper, up_poses)
        self.L = posed_teeth(lower, low_poses)
        solids = [
            trimesh.Trimesh(
                transform_points(up_poses.get(n, np.eye(4)), np.asarray(s.vertices)),
                s.faces,
                process=False,
            )
            for n, s in upper.contact_solids.items()
        ]
        merged = cast(trimesh.Trimesh, trimesh.util.concatenate(solids))
        cusp_z = [t.cusp[2] for t in self.U.values()]
        self.sdf = _band_sdf(
            merged, min(cusp_z) - 4.0, float(merged.vertices[:, 2].max()) + SEAT_REACH_MM
        )
        self.samples: dict[int, Array] = {}
        for n, t in self.L.items():
            pts = transform_points(low_poses.get(n, np.eye(4)), lower.enamel_samples(n, 600, 8))
            h = (pts - t.margin_center) @ t.o
            top = pts[h >= 0.6 * h.max()]
            self.samples[n] = top[:: max(1, len(top) // 150)]
        self.all_samples = np.vstack(list(self.samples.values()))
        self.posterior = [
            n
            for n in self.L
            if fdi.position(n) >= 4
            and fdi.fdi_number("upper", fdi.is_patient_right(n), fdi.position(n)) in self.U
        ]
        lc = [n for n in (31, 41) if n in self.L]
        self.profile = (
            _labial_profile(
                np.vstack(
                    [
                        transform_points(
                            low_poses.get(n, np.eye(4)), np.asarray(lower.patches[n].vertices)
                        )
                        for n in lc
                    ]
                )
            )
            if lc
            else None
        )

    def _up(self, X: Array, n: int, field: str) -> Array:
        return transform_points(X, getattr(self.U[n], field)[None])[0]

    def relation(self, X: Array) -> dict[str, float]:
        """Overjet, midline, Class I offsets (molars, canines) and transverse asymmetry."""
        out: dict[str, float] = {}
        uc = [n for n in (11, 21) if n in self.U]
        if uc and self.profile is not None:
            edge = np.mean([self._up(X, n, "cusp") for n in uc], axis=0)
            z, y = self.profile
            out["overjet_mm"] = float(edge[1] - np.interp(edge[2], z, y))
        if {11, 21} <= set(self.U) and {31, 41} <= set(self.L):
            u_mid = (self._up(X, 11, "mesial_contact") + self._up(X, 21, "mesial_contact")) / 2
            l_mid = (self.L[31].mesial_contact + self.L[41].mesial_contact) / 2
            out["midline_mm"] = float(u_mid[0] - l_mid[0])
        for side, (u6, l6, u3, l3) in {
            "right": (16, 46, 13, 43),
            "left": (26, 36, 23, 33),
        }.items():
            if u6 in self.U and l6 in self.L:
                out[f"molar_{side}_mm"] = float(
                    self._up(X, u6, "mesial_contact")[1]
                    - self.L[l6].mesial_contact[1]
                    + CLASS_I_MOLAR_OFFSET_MM
                )
            if u3 in self.U and l3 in self.L:
                out[f"canine_{side}_mm"] = float(
                    self._up(X, u3, "cusp")[1] - self.L[l3].distal_contact[1]
                )
        lateral = {1: [], -1: []}
        for n in self.posterior:
            u = fdi.fdi_number("upper", fdi.is_patient_right(n), fdi.position(n))
            sign = 1.0 if fdi.is_patient_right(n) else -1.0  # lower frame +x = patient right
            lateral[int(sign)].append(sign * (self._up(X, u, "cusp")[0] - self.L[n].cusp[0]))
        if lateral[1] and lateral[-1]:
            out["transverse_mm"] = float(np.mean(lateral[1]) - np.mean(lateral[-1]))
        return out

    def _relation_residuals(self, X: Array) -> list[float]:
        r = self.relation(X)
        res = []
        if "overjet_mm" in r:
            res.append(r["overjet_mm"] - self.overjet)
        for key, w in (("midline_mm", 1.0), ("transverse_mm", 0.5)):
            if key in r:
                res.append(w * r[key])
        for side in ("right", "left"):
            if f"molar_{side}_mm" in r:
                res.append(r[f"molar_{side}_mm"])
            if f"canine_{side}_mm" in r:
                res.append(0.5 * r[f"canine_{side}_mm"])
        return res

    def _seat_residuals(self, X: Array) -> Array:
        inv = np.linalg.inv(X)
        d_all = self.sdf(transform_points(inv, self.all_samples))
        penetration = 10.0 * np.maximum(0.0, -d_all - SEAT_TOLERANCE_MM)
        contacts = []
        for n in self.posterior:
            d = self.sdf(transform_points(inv, self.samples[n]))
            d = np.clip(d, -1.0, SEAT_REACH_MM)
            soft = -SOFTMIN_MM * np.log(np.mean(np.exp(-d / SOFTMIN_MM)))
            contacts.append(soft)
        return np.r_[penetration, contacts]

    def solve(self, start: Array) -> tuple[Array, dict[str, Any]]:
        """In-plane fit of the landmark relation with the arches held apart, then seating
        (vertical, roll, pitch about the posterior teeth), then a joint 6-DOF refinement."""
        lift = translation(np.array([0.0, 0.0, LIFT_MM]))
        posterior = [self.samples[n] for n in self.posterior] or [self.all_samples]
        c = np.vstack(posterior).mean(axis=0)
        to_c, from_c = translation(-c), translation(c)

        def about_c(x: Array) -> Array:
            return from_c @ _correction(x) @ to_c

        def planar(x: Array) -> Array:
            C = about_c(np.array([0.0, 0.0, x[0], x[1], x[2], 0.0]))
            return np.asarray(self._relation_residuals(C @ lift @ start))

        p = least_squares(planar, np.zeros(3), x_scale=np.array([0.02, 0.5, 0.5]))
        X1 = about_c(np.array([0.0, 0.0, p.x[0], p.x[1], p.x[2], 0.0])) @ lift @ start

        def seat(x: Array) -> Array:
            return self._seat_residuals(about_c(np.array([x[0], x[1], 0.0, 0.0, 0.0, x[2]])) @ X1)

        q = least_squares(seat, np.zeros(3), x_scale=np.array([0.02, 0.02, 0.5]), diff_step=1e-3)
        X2 = about_c(np.array([q.x[0], q.x[1], 0.0, 0.0, 0.0, q.x[2]])) @ X1

        def full(x: Array) -> Array:
            X = about_c(x) @ X2
            return np.r_[np.asarray(self._relation_residuals(X)), self._seat_residuals(X)]

        f = least_squares(
            full, np.zeros(6), x_scale=np.array([0.02] * 3 + [0.5] * 3), diff_step=1e-3
        )
        X = about_c(f.x) @ X2
        metrics: dict[str, Any] = self.relation(X)
        d = self.sdf(transform_points(np.linalg.inv(X), self.all_samples))
        metrics["max_penetration_mm"] = float(max(0.0, -d.min()))
        warnings = []
        worst = max(
            [abs(v) for k, v in metrics.items() if k.startswith(("molar_", "canine_"))] or [0.0]
        )
        if worst > 1.0 or abs(metrics.get("overjet_mm", self.overjet) - self.overjet) > 1.0:
            warnings.append(
                f"Class I and the {self.overjet:.1f} mm overjet cannot both be met "
                f"(off by up to {worst:.1f} mm): tooth sizes or incisor inclinations do not match"
            )
        metrics["warnings"] = warnings
        return X, metrics


# -- distributing the occlusion to the jaws -----------------------------------------------------
def lefort_matrix(pivot: Array, t: Array, pitch: float, roll: float, yaw: float) -> Array:
    """T(t) T(p) Rz(yaw) Ry(roll) Rx(pitch) T(-p) in the head frame (angles in radians)."""
    R = Rotation.from_euler("ZYX", [yaw, roll, pitch]).as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t + pivot - R @ pivot
    return M


def distribute(
    X: Array,
    hf: HeadFrames,
    cfg: dict[str, Any],
    upper_lm: dict[str, Array],
    mand: dict[str, Array],
) -> tuple[Array, Array, list[str]]:
    """Jaw movements (maxilla, mandible) realizing the upper->lower relation X.

    `upper_lm` are the upper landmarks of the setup (arch frame), whose post-op cant, yaw and
    midline the Le Fort I corrects."""
    P_u, P_l = hf.P["upper"], hf.P["lower"]
    cuts = set(cfg["osteotomies"])
    warnings: list[str] = []

    def mandible_for(M_max: Array) -> Array:
        return M_max @ P_u @ np.linalg.inv(X) @ np.linalg.inv(P_l)

    if cuts == {"bsso"}:
        return np.eye(4), mandible_for(np.eye(4)), warnings
    pre = to_head(P_u, upper_lm)
    pivot = pre.get("incisor_midline", np.mean(list(pre.values()), axis=0))
    if "bsso" in cuts:
        forward, up = cfg["maxilla_forward_mm"], cfg["maxilla_up_mm"]
        pitch = np.radians(cfg["maxilla_pitch_deg"])

        def residual(x: Array) -> Array:
            M = lefort_matrix(pivot, np.array([x[0], forward, up]), pitch, x[1], x[2])
            a = jaw_angles(to_head(M, pre))
            return np.array(
                [a.get("midline_mm", 0.0), a.get("cant_deg", 0.0), a.get("yaw_deg", 0.0)]
            )

        x = least_squares(residual, np.zeros(3), x_scale=np.array([1.0, 0.02, 0.02])).x
        M_max = lefort_matrix(pivot, np.array([x[0], forward, up]), pitch, x[1], x[2])
        return M_max, mandible_for(M_max), warnings
    # Le Fort I only: the mandible autorotates about the condyles; the maxilla meets it
    if not {"condylion_right", "condylion_left"} <= set(mand):
        raise CaseError("Le Fort I alone needs both condyles in the CT (mandible autorotation)")
    c_r, c_l = mand["condylion_right"], mand["condylion_left"]

    def maxilla_for(theta: float) -> tuple[Array, Array]:
        hinge = rotation_about(c_l - c_r, theta, c_r)
        return hinge @ P_l @ X @ np.linalg.inv(P_u), hinge

    def rise(theta: float) -> float:
        M, _ = maxilla_for(theta)
        return float(transform_points(M, pivot[None])[0][2] - pivot[2] - cfg["maxilla_up_mm"])

    thetas = np.radians(np.linspace(-15, 15, 61))
    values = np.array([rise(t) for t in thetas])
    crossing = np.flatnonzero(np.diff(np.sign(values)) != 0)
    if len(crossing):
        i = int(crossing[np.argmin(np.abs(thetas[crossing]))])
        theta = float(cast(float, brentq(rise, thetas[i], thetas[i + 1], xtol=1e-6)))
    else:
        theta = float(thetas[np.argmin(np.abs(values))])
        warnings.append(
            f"maxilla_up_mm = {cfg['maxilla_up_mm']:.1f} cannot be reached by autorotation"
        )
    M_max, M_mand = maxilla_for(theta)
    return M_max, M_mand, warnings


# -- bone segments ------------------------------------------------------------------------------
def _segment_key(case: Case) -> str:
    parts = [(case.ct_dir / "ct.json").read_text()]
    parts += [(case.anatomy_dir(jaw) / "teeth.json").read_text() for jaw in JAWS]
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _world(vol: Volume, mask: NDArray[np.bool_]) -> tuple[NDArray[np.intp], Array]:
    ijk = np.argwhere(mask)
    return ijk, vol.to_world(ijk.astype(float))


def _only(vol: Volume, ijk: NDArray[np.intp], keep: NDArray[np.bool_]) -> Volume:
    data = np.zeros(vol.data.shape, dtype=bool)
    data[tuple(ijk[keep].T)] = True
    return Volume(data, vol.origin, vol.spacing, vol.direction)


def lefort_plane_z(upper: JawAnatomy) -> float:
    """Le Fort I level in the upper arch frame (z toward the lower arch): above every apex."""
    apexes = [float(np.asarray(r.vertices)[:, 2].min()) for r in upper.roots.values()]
    if not apexes:
        apexes = [t.cusp[2] - t.crown_height - t.root_length for t in upper.teeth.values()]
    return min(apexes) - LEFORT_ABOVE_APEX_MM


def _bsso_planes(lower: JawAnatomy) -> dict[str, tuple[Array, Array]]:
    """Per side: a point and the distal direction (lower arch frame) of the split plane."""
    planes = {}
    for right, name in ((True, "right"), (False, "left")):
        side = [n for n in lower.teeth if fdi.is_patient_right(n) == right]
        if not side:
            continue
        last = lower.teeth[max(side, key=fdi.position)]
        d = last.distal_contact - last.mesial_contact
        d[2] = 0.0
        d /= np.linalg.norm(d)
        planes[name] = (last.distal_contact + BSSO_DISTAL_MM * d, d)
    return planes


def bone_segments(
    case: Case, hf: HeadFrames, anatomies: dict[Jaw, JawAnatomy]
) -> dict[str, trimesh.Trimesh]:
    """Maxilla (Le Fort I), mandible distal and proximal segments and the cranium, in CT
    coordinates (DICOM LPS). Cut in voxel space and meshed, cached in work/surgery/."""
    out_dir = case.work / "surgery"
    key = _segment_key(case)
    names = ("maxilla", "mandible_distal", "proximal_right", "proximal_left", "cranium")
    info_path = out_dir / "segments.json"
    if info_path.exists() and read_json(info_path).get("key") == key:
        return {
            n: read_mesh(out_dir / f"{n}.ply") for n in names if (out_dir / f"{n}.ply").exists()
        }
    jb = np.load(case.ct_dir / "jawbone.npz")
    grid = (jb["origin"], jb["spacing"], jb["direction"])
    meshes: dict[str, trimesh.Trimesh] = {}
    z_cut = lefort_plane_z(anatomies["upper"])
    upper = Volume(jb["upper"], *grid)
    ijk, w = _world(upper, upper.data)
    below = transform_points(hf.ct_to_arch["upper"], w)[:, 2] >= z_cut
    if below.any():
        meshes["maxilla"] = mask_mesh(upper, _only(upper, ijk, below).data, SEGMENT_FACES)
    lower = Volume(jb["lower"], *grid)
    ijk, w = _world(lower, lower.data)
    a = transform_points(hf.ct_to_arch["lower"], w)
    distal = np.ones(len(a), dtype=bool)
    for name, (point, d) in _bsso_planes(anatomies["lower"]).items():
        sign = 1.0 if name == "right" else -1.0  # lower arch frame +x = patient right
        proximal = ((a - point) @ d > 0) & (sign * a[:, 0] > 0)
        distal &= ~proximal
        if proximal.any():
            meshes[f"proximal_{name}"] = mask_mesh(
                lower, _only(lower, ijk, proximal).data, SEGMENT_FACES
            )
    if distal.any():
        meshes["mandible_distal"] = mask_mesh(lower, _only(lower, ijk, distal).data, SEGMENT_FACES)
    bone, labels = load_bone(case)
    lower_parts = (labels == fdi.TS_LOWER_JAWBONE) | np.isin(
        labels, [k for k, n in {**fdi.TS_TOOTH_LABELS, **fdi.TS_PULP_LABELS}.items() if n > 30]
    )
    from scipy import ndimage

    far = np.asarray(ndimage.distance_transform_edt(~lower_parts, sampling=bone.spacing))
    ijk, w = _world(bone, bone.data & (far > CRANIUM_EXCLUSION_MM))
    above = transform_points(hf.ct_to_arch["upper"], w)[:, 2] < z_cut
    if above.any():
        meshes["cranium"] = mask_mesh(
            bone, _only(bone, ijk, above).data, SEGMENT_FACES, voxel_mm=float(bone.spacing[0])
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.ply"):
        old.unlink()
    for name, mesh in meshes.items():
        write_mesh(out_dir / f"{name}.ply", mesh)
    write_json(info_path, {"key": key, "coordinates": "CT world, DICOM LPS mm"})
    return meshes


def split_gaps(
    hf: HeadFrames, lower: JawAnatomy, M_mand: Array, meshes: dict[str, trimesh.Trimesh]
) -> dict[str, float]:
    """Per side, how far the distal segment moves away from (+ gap) or into (- overlap) the
    proximal segment at the split, along the arch at the osteotomy."""
    out = {}
    if "mandible_distal" not in meshes:
        return out
    arch_to_head = hf.P["lower"]
    V = transform_points(hf.ct_to_arch["lower"], np.asarray(meshes["mandible_distal"].vertices))
    for name, (point, d) in _bsso_planes(lower).items():
        face = V[np.abs((V - point) @ d) < SPLIT_FACE_MM]
        if not len(face):
            continue
        c = transform_points(arch_to_head, face.mean(axis=0)[None])[0]
        d_head = transform_vectors(arch_to_head, d[None])[0]
        moved = transform_points(M_mand, c[None])[0] - c
        out[name] = float(-(moved @ d_head))
    return out


# -- report -------------------------------------------------------------------------------------
def surgery_report(
    case: Case,
    plan: dict[str, Any],
    anatomies: dict[Jaw, JawAnatomy],
    hf: HeadFrames,
    M: dict[str, Array],
    final: dict[Jaw, dict[int, Array]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Everything a surgeon needs: movement tables (input format), angles, transforms, skeletal
    landmarks, split gaps. `M` maps "maxilla"/"mandible" to head-frame movements."""
    H = hf.ct_to_head
    Q = {jaw: M[BONE[jaw]] @ hf.P[jaw] for jaw in JAWS}
    scanned = {jaw: to_head(hf.P[jaw], dental_landmarks(anatomies[jaw])) for jaw in JAWS}
    setup = {jaw: dental_landmarks(anatomies[jaw], final[jaw]) for jaw in JAWS}
    mand = mandible_landmarks(case, hf, anatomies["lower"])
    mand_post = to_head(M["mandible"], mand)
    angles = {
        BONE[jaw]: {
            "scanned": jaw_angles(scanned[jaw]),
            "before_surgery": jaw_angles(to_head(hf.P[jaw], setup[jaw])),
            "after_surgery": jaw_angles(to_head(Q[jaw], setup[jaw])),
        }
        for jaw in JAWS
    }
    meshes = bone_segments(case, hf, anatomies)
    report: dict[str, Any] = {
        "mode": plan["surgery"]["mode"],
        "movements": {BONE[jaw]: transform_to_table(M[BONE[jaw]], scanned[jaw]) for jaw in JAWS},
        "angles": angles,
        "incisors": {
            "scanned": incisor_angles(anatomies, hf.P, mand),
            "after_surgery": incisor_angles(anatomies, Q, mand_post, final),
        },
        "chin_mm": {k: float(-mand_post[k][0]) for k in ("pogonion", "menton") if k in mand_post},
        "split_gap_mm": split_gaps(hf, anatomies["lower"], M["mandible"], meshes)
        if extra["bsso"]
        else {},
        "transforms_head": {k: v for k, v in M.items()},
        "transforms_ct_lps": {k: np.linalg.inv(H) @ v @ H for k, v in M.items()},
        "ct_to_head": H,
        "head_pitch_deg": hf.head_pitch_deg,
        "arch_to_head": {"before_surgery": hf.P, "after_surgery": Q},
        "mandible_bone_to_head": hf.bone["lower"],
        "landmarks_scanned_head": scanned,
        "mandible_landmarks_head": {"before_surgery": mand, "after_surgery": mand_post},
        "notes": [
            "head frame: +x patient right, +y anterior, +z superior (mm); movement tables: "
            "forward_mm + anterior, up_mm + superior, left_mm + toward the patient's left, "
            "measured at the scanned landmark positions",
            "transforms_ct_lps map pre-op positions (mandible in occlusion) to post-op "
            "positions, both in CT DICOM LPS coordinates",
            "skeletal landmarks and split gaps are approximate",
        ],
        **extra,
    }
    return report


def landmarks_document(case: Case, report: dict[str, Any]) -> dict[str, Any]:
    """Named pre-op landmarks in CT LPS and head coordinates: the contract other planning
    software uses to express a plan as a movement table (evaluate its transforms here)."""
    H = np.asarray(report["ct_to_head"], float)
    inv = np.linalg.inv(H)
    out: dict[str, Any] = {
        "format": "aligners-surgery-landmarks",
        "version": 1,
        "head_frame_in_ct_lps": {
            "origin": inv[:3, 3],
            "x_patient_right": inv[:3, 0],
            "y_anterior": inv[:3, 1],
            "z_superior": inv[:3, 2],
        },
        "jaws": {},
        "note": "dental landmarks at their scanned positions; the mandible in occlusion",
    }
    for jaw, bone in BONE.items():
        lm = {k: np.asarray(v, float) for k, v in report["landmarks_scanned_head"][jaw].items()}
        out["jaws"][bone] = {
            k: {"ct_lps": transform_points(inv, v[None])[0], "head": v} for k, v in lm.items()
        }
    return out


def movements_toml(report: dict[str, Any]) -> str:
    """The planned movements in plan.toml's input format (replace the two tables there)."""
    lines = [
        "# Planned jaw movements (aligners surgery output). Replace [surgery.maxilla] and",
        '# [surgery.mandible] in plan.toml with these and set [surgery] mode = "manual" to align',
        "# the teeth to exactly this plan (head frame, mm: + forward, + up, + patient-left).",
        f"# head_pitch_deg = {report['head_pitch_deg']}",
    ]
    for bone, label in (("maxilla", "Le Fort I"), ("mandible", "BSSO distal segment")):
        lines += ["", f"[surgery.{bone}]  # {label}"]
        for name, m in report["movements"][bone].items():
            lines.append(
                f"{name} = {{ forward_mm = {m['forward_mm']:.1f}, up_mm = {m['up_mm']:.1f}, "
                f"left_mm = {m['left_mm']:.1f} }}"
            )
    return "\n".join(lines) + "\n"


def posed_tooth_meshes(anatomy: JawAnatomy, poses: dict[int, Array], T: Array) -> trimesh.Trimesh:
    """Whole teeth (CT tooth meshes where available, else crowns) at their poses, mapped by T."""
    parts = []
    for n in anatomy.order():
        mesh = anatomy.roots.get(n, anatomy.contact_solids[n])
        parts.append(
            trimesh.Trimesh(
                transform_points(T @ poses.get(n, np.eye(4)), np.asarray(mesh.vertices)),
                mesh.faces,
                process=False,
            )
        )
    return cast(trimesh.Trimesh, trimesh.util.concatenate(parts))


def segment_poses(report: dict[str, Any]) -> dict[str, tuple[Array, Array]]:
    """Head-frame transforms (before, after surgery) of each bone segment from CT coordinates.
    Without a split (Le Fort I alone) the whole mandible autorotates."""
    H = np.asarray(report["ct_to_head"], float)
    B = np.asarray(report["mandible_bone_to_head"], float)
    M_max = np.asarray(report["transforms_head"]["maxilla"], float)
    M_mand = np.asarray(report["transforms_head"]["mandible"], float)
    proximal = B if report["bsso"] else M_mand @ B
    return {
        "maxilla": (H, M_max @ H),
        "mandible_distal": (B, M_mand @ B),
        "proximal_right": (B, proximal),
        "proximal_left": (B, proximal),
        "cranium": (H, H),
    }
