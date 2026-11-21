"""Orthognathic surgery planned on the teeth alone: each arch is aligned on its own jaw, and the
rigid movement of a jaw that brings the two aligned arches into the ideal occlusion is
reported (auto), or a surgeon's planned movements are applied (manual).

Movement frame (right-handed, mm): origin at the upper incisal midline, +x patient right,
+y anterior, +z superior; its horizontal is the upper occlusal plane tilted by `pitch_deg`
(+ = anterior up). A jaw movement M is a rigid 4x4 in this frame; a jaw's arch frame maps into
it by P before and by M @ P after surgery. Movement tables give landmark displacements in mm:
forward_mm (+ anterior), up_mm (+ superior), left_mm (+ toward the patient's left), measured at
the scanned (pre-treatment) landmark positions.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aligners import fdi
from aligners.analyze import (
    CLASS_I_MOLAR_OFFSET_MM,
    arch_to_bite,
    incisor_inclination,
    posed_teeth,
)
from aligners.anatomy import JawAnatomy
from aligners.case import LANDMARKS, Case, CaseError, read_json
from aligners.fdi import JAWS, Jaw
from aligners.fields import Grid, GridSDF
from aligners.geometry import (
    Array,
    MeshSDF,
    rigid_fit,
    rotation_about,
    transform_points,
    translation,
)

BONE = {"upper": "maxilla", "lower": "mandible"}
JAW_OF = {"maxilla": "upper", "mandible": "lower"}
FIT_WARN_MM = 0.3
FIT_ERROR_MM = 1.5
SDF_RES_MM = 0.3
SEAT_TOLERANCE_MM = 0.05
LIFT_MM = 2.0  # the upper arch starts the in-plane fit opened by this much
SEAT_REACH_MM = 6.0  # occlusal signed distances are resolved (with a gradient) up to this far
SOFTMIN_MM = 0.1
# relation priorities: the incisors and canines decide the occlusion, the molars follow
OVERJET_WEIGHT = 2.0
MIDLINE_WEIGHT = 1.0
CANINE_WEIGHT = 1.0
MOLAR_WEIGHT = 0.5
TRANSVERSE_WEIGHT = 0.5
UPPER_ARCH_TO_FRAME = np.diag([-1.0, 1.0, -1.0, 1.0])  # upper arch: x patient left, z down


@dataclass
class DentalFrames:
    P: dict[Jaw, Array]  # arch frame -> movement frame, before surgery (scanned bite)
    ct_to_frame: dict[Jaw, Array]  # CT LPS -> movement frame of each jaw's CT teeth (scanned bite)
    to_bite: dict[Jaw, Array]  # arch frame -> the case's bite frame (frames.json)
    pitch_deg: float

    @property
    def bite_to_frame(self) -> Array:
        return self.P["upper"] @ np.linalg.inv(self.to_bite["upper"])


def dental_frames(case: Case, plan: dict[str, Any]) -> DentalFrames:
    to_bite = arch_to_bite(case)
    if to_bite is None:
        raise CaseError("surgical modes need the bite (bite-registered scans or a CT)")
    frames = read_json(case.work / "frames.json")["jaws"]
    ct_to_arch = {
        jaw: np.asarray(frames[jaw]["scan_to_arch"], float)
        @ np.asarray(frames[jaw]["ct_to_scan"], float)
        for jaw in JAWS
    }
    pitch = float(plan["surgery"]["pitch_deg"])
    # + pitch tilts the frame's horizontal anterior-up against the occlusal plane, so the
    # occlusal plane's front lies below it: turn about +x (patient right) by -pitch
    P_u = rotation_about(np.array([1.0, 0.0, 0.0]), -np.radians(pitch)) @ UPPER_ARCH_TO_FRAME
    P_l = P_u @ np.linalg.inv(to_bite["upper"]) @ to_bite["lower"]
    P = {"upper": P_u, "lower": P_l}
    return DentalFrames(
        P=P,
        ct_to_frame={jaw: P[jaw] @ ct_to_arch[jaw] for jaw in JAWS},
        to_bite=to_bite,
        pitch_deg=pitch,
    )


# -- landmarks ----------------------------------------------------------------------------------
def _incisor_midline(anatomy: JawAnatomy, poses: dict[int, Array] | None = None) -> Array | None:
    teeth = posed_teeth(anatomy, poses)
    centrals = [fdi.fdi_number(anatomy.jaw, side, 1) for side in (True, False)]
    if all(c in teeth for c in centrals):
        return (teeth[centrals[0]].cusp + teeth[centrals[1]].cusp) / 2
    return None


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


def to_frame(T: Array, landmarks: dict[str, Array]) -> dict[str, Array]:
    return {k: transform_points(T, v[None])[0] for k, v in landmarks.items()}


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
    """Rigid movement from landmark displacements, and the per-landmark residuals."""
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


def movement_angles(M: Array) -> dict[str, float]:
    """Rotation of a jaw movement in degrees: pitch (+ = front up), roll (+ = patient-left side
    up), yaw (+ = front toward the patient's left), as M = Rz(yaw) Ry(roll) Rx(pitch)."""
    yaw, roll, pitch = Rotation.from_matrix(M[:3, :3]).as_euler("ZYX", degrees=True)
    return {"pitch_deg": float(pitch), "roll_deg": float(roll), "yaw_deg": float(yaw)}


def incisor_target_axes(
    anatomies: dict[Jaw, JawAnatomy], cfg: dict[str, Any]
) -> dict[Jaw, dict[int, Array]]:
    """Decompensated crownward incisor axes in each arch frame: tilted labially by the target
    inclination from the perpendicular to the arch's own occlusal plane."""
    out: dict[Jaw, dict[int, Array]] = {}
    for jaw, key in (("upper", "upper_incisor_deg"), ("lower", "lower_incisor_deg")):
        a = np.radians(cfg[key])
        axes = {}
        for n, t in anatomies[jaw].teeth.items():
            if fdi.position(n) > 2:
                continue
            b = np.array([t.b[0], t.b[1], 0.0])
            b /= np.linalg.norm(b)
            axes[n] = np.cos(a) * np.array([0.0, 0.0, 1.0]) + np.sin(a) * b
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
    the target overjet, coinciding midlines, Class I canines and molars, a centered transverse
    relation, and seated (touching, not penetrating) posterior teeth. The incisors and canines
    take priority over the molars when tooth sizes do not allow all of them."""

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
            pts = transform_points(low_poses.get(n, np.eye(4)), lower.enamel_samples(n, 1500, 8))
            h = (pts - t.margin_center) @ t.o
            self.samples[n] = pts[h >= 0.6 * h.max()]  # dense: cusp tips must not slip through
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
        lateral: dict[int, list[float]] = {1: [], -1: []}
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
            res.append(OVERJET_WEIGHT * (r["overjet_mm"] - self.overjet))
        for key, w in (("midline_mm", MIDLINE_WEIGHT), ("transverse_mm", TRANSVERSE_WEIGHT)):
            if key in r:
                res.append(w * r[key])
        for side in ("right", "left"):
            if f"molar_{side}_mm" in r:
                res.append(MOLAR_WEIGHT * r[f"molar_{side}_mm"])
            if f"canine_{side}_mm" in r:
                res.append(CANINE_WEIGHT * r[f"canine_{side}_mm"])
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
        canines = [abs(v) for k, v in metrics.items() if k.startswith("canine_")]
        molars = [abs(v) for k, v in metrics.items() if k.startswith("molar_")]
        if (
            abs(metrics.get("overjet_mm", self.overjet) - self.overjet) > 1.0
            or max(canines or [0.0]) > 1.5
        ):
            warnings.append(
                f"the {self.overjet:.1f} mm overjet and Class I canines cannot both be met (canines "
                f"off by up to {max(canines or [0.0]):.1f} mm): tooth sizes do not match "
                "(see bolton_anterior_pct)"
            )
        if max(molars or [0.0]) > 2.0:
            warnings.append(
                f"molars end up to {max(molars):.1f} mm from Class I with the incisors and canines "
                "in place"
            )
        metrics["warnings"] = warnings
        return X, metrics


# -- jaw movements ------------------------------------------------------------------------------
def movements_for_relation(X: Array, frames: DentalFrames, move: str) -> dict[str, Array]:
    """Jaw movements (movement frame) that give the upper->lower arch relation X, moving only
    `move` ("maxilla" or "mandible")."""
    P_u, P_l = frames.P["upper"], frames.P["lower"]
    if move == "mandible":
        return {"maxilla": np.eye(4), "mandible": P_u @ np.linalg.inv(X) @ np.linalg.inv(P_l)}
    return {"maxilla": P_l @ X @ np.linalg.inv(P_u), "mandible": np.eye(4)}


def arch_to_bite_after(frames: DentalFrames, M: dict[str, Array]) -> dict[Jaw, Array]:
    """Arch -> bite-frame transforms of both jaws after surgery."""
    to_bite = np.linalg.inv(frames.bite_to_frame)
    return {jaw: to_bite @ M[BONE[jaw]] @ frames.P[jaw] for jaw in JAWS}


def surgery_report(
    plan: dict[str, Any],
    anatomies: dict[Jaw, JawAnatomy],
    frames: DentalFrames,
    M: dict[str, Array],
    final: dict[Jaw, dict[int, Array]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Movement tables (the input format), rotations, transforms and landmarks of the jaw
    movements `M` ("maxilla"/"mandible", movement frame)."""
    C = frames.ct_to_frame["upper"]
    scanned = {jaw: to_frame(frames.P[jaw], dental_landmarks(anatomies[jaw])) for jaw in JAWS}
    after = {jaw: M[BONE[jaw]] @ frames.P[jaw] for jaw in JAWS}
    relative = np.linalg.inv(M["maxilla"]) @ M["mandible"]
    return {
        "mode": plan["surgery"]["mode"],
        "movements": {BONE[jaw]: transform_to_table(M[BONE[jaw]], scanned[jaw]) for jaw in JAWS},
        "rotations": {bone: movement_angles(T) for bone, T in M.items()},
        "mandible_relative_to_maxilla": {
            "movement": transform_to_table(relative, scanned["lower"]),
            "rotation": movement_angles(relative),
        },
        "incisor_inclination_deg": {
            jaw: {
                "scanned": incisor_inclination(anatomies[jaw]),
                "planned": incisor_inclination(anatomies[jaw], final[jaw]),
            }
            for jaw in JAWS
        },
        "transforms_frame": M,
        "transforms_ct_lps": {k: np.linalg.inv(C) @ v @ C for k, v in M.items()},
        "ct_to_frame": frames.ct_to_frame,
        "pitch_deg": frames.pitch_deg,
        "arch_to_frame": {"before_surgery": frames.P, "after_surgery": after},
        "landmarks_scanned_frame": scanned,
        "notes": [
            "movement frame: origin at the upper incisal midline, +x patient right, "
            "+y anterior, +z superior (mm); horizontal = the upper occlusal plane tilted by "
            "pitch_deg (+ anterior up)",
            "movement tables: forward_mm + anterior, up_mm + superior, left_mm + toward the "
            "patient's left, measured at the scanned landmark positions",
            "transforms_ct_lps map pre-op positions (scanned bite) to post-op positions, both "
            "in CT DICOM LPS coordinates",
        ],
        **extra,
    }


LANDMARKS_RULE = (
    "Move each landmark's ct_lps point with the jaw segment that contains it in the CT image, "
    "then d = moved point - reference_lps (mm, LPS). forward_mm = d . y_anterior, "
    "up_mm = d . z_superior, left_mm = -(d . x_patient_right). Write the values as "
    "[surgery.maxilla] / [surgery.mandible] tables of plan.toml, rounded to 0.1 mm."
)


def landmarks_document(report: dict[str, Any]) -> dict[str, Any]:
    """The contract other planning software uses to express its plan as a movement table.

    `reference_lps` is where a displacement is measured from (scanned teeth, in the occlusion of
    the scans); `ct_lps` is where that point lies in the CT image. They differ for the mandible
    when the CT was taken in another bite than the scans. (`head_frame` holds the movement
    frame's axes.)"""
    C = {jaw: np.asarray(report["ct_to_frame"][jaw], float) for jaw in JAWS}
    frame_to_ct = np.linalg.inv(C["upper"])
    out: dict[str, Any] = {
        "format": "aligners-surgery-landmarks",
        "version": 1,
        "coordinates": "DICOM patient LPS mm of the CT",
        "head_frame": {
            "origin": frame_to_ct[:3, 3],
            "x_patient_right": frame_to_ct[:3, 0],
            "y_anterior": frame_to_ct[:3, 1],
            "z_superior": frame_to_ct[:3, 2],
        },
        "head_pitch_deg": report["pitch_deg"],
        "rule": LANDMARKS_RULE,
        "jaws": {},
    }
    for jaw, bone in BONE.items():
        to_ct = np.linalg.inv(C[jaw])
        out["jaws"][bone] = {
            name: {
                "reference_lps": transform_points(frame_to_ct, np.asarray(p, float)[None])[0],
                "ct_lps": transform_points(to_ct, np.asarray(p, float)[None])[0],
            }
            for name, p in report["landmarks_scanned_frame"][jaw].items()
        }
    return out


def movements_toml(report: dict[str, Any]) -> str:
    """The planned movements in plan.toml's input format (replace the two tables there)."""
    lines = [
        "# Planned jaw movements (aligners surgery output). Replace [surgery.maxilla] and",
        '# [surgery.mandible] in plan.toml with these and set [surgery] mode = "manual" to align',
        "# the teeth to exactly this plan (movement frame, mm: + forward, + up, + patient-left).",
        f"# pitch_deg = {report['pitch_deg']}",
    ]
    for bone in ("maxilla", "mandible"):
        lines += ["", f"[surgery.{bone}]"]
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
