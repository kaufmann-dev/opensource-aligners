"""CT tooth registration onto intraoral scans."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from aligners import fdi
from aligners.case import Case, CaseError, read_json, read_mesh
from aligners.fdi import Jaw
from aligners.geometry import (
    Array,
    ICPTarget,
    frame_matrix,
    orthonormal_frame,
    rotation_about,
    rotation_angle_deg,
    subsample,
    transform_points,
    translation,
)

REGISTRATION_BONE_MM = 2.5  # CT surface farther than this from bone is likely visible enamel
MIN_CROWN_POINTS = 50
TOOTH_MAX_SHIFT_MM = 1.0
TOOTH_MAX_ANGLE_DEG = 5.0
MAX_VISIBLE_RMSE_MM = 1.0  # worse fits are teeth absent from the scan (e.g. unerupted)


@dataclass
class CTTooth:
    fdi: int
    mesh: trimesh.Trimesh  # CT world coordinates
    bone_dist: Array  # per vertex
    axis: Array  # crown-ward long axis, CT world

    def crown_points(self, min_bone_mm: float = REGISTRATION_BONE_MM) -> Array:
        return np.asarray(self.mesh.vertices)[self.bone_dist > min_bone_mm]


def load_ct_teeth(case: Case, jaw: Jaw) -> list[CTTooth]:
    info = read_json(case.require(case.ct_dir / "ct.json", "ct"))
    teeth = []
    for t in info["teeth"]:
        if t["jaw"] != jaw:
            continue
        number = t["fdi"]
        teeth.append(
            CTTooth(
                fdi=number,
                mesh=read_mesh(case.ct_dir / "teeth" / f"{number}.ply"),
                bone_dist=np.load(case.ct_dir / "teeth" / f"{number}_bone_dist.npy"),
                axis=np.asarray(t["axis"], float),
            )
        )
    return teeth


def ct_arch_frame(teeth: list[CTTooth], jaw: Jaw) -> Array:
    """CT world -> arch frame (same convention as scan arch frames)."""
    visible = [t for t in teeth if (t.bone_dist > REGISTRATION_BONE_MM).sum() >= MIN_CROWN_POINTS]
    z = np.mean([t.axis for t in visible], axis=0)
    crowns = {t.fdi: t.crown_points() for t in visible}
    centers = {n: p.mean(axis=0) for n, p in crowns.items()}

    def mean_of(positions: tuple[int, ...]) -> Array:
        pts = [c for n, c in centers.items() if fdi.position(n) in positions]
        return np.mean(pts, axis=0)

    incisors = mean_of((1,))
    molars = mean_of((6, 7))
    x, y, z = orthonormal_frame(z, incisors - molars)
    right = np.mean([c for n, c in centers.items() if fdi.is_patient_right(n)], axis=0)
    left = np.mean([c for n, c in centers.items() if not fdi.is_patient_right(n)], axis=0)
    # +x must be patient-right in the lower jaw, patient-left in the upper jaw.
    want = 1.0 if jaw == "lower" else -1.0
    if np.sign(np.dot(right - left, x)) != want:
        raise CaseError("CT tooth labels are mirrored relative to the arch geometry")
    incisal = np.vstack([crowns[n] for n in crowns if fdi.position(n) == 1])
    tip = incisal[np.argmax(incisal @ z)]
    origin = incisors + np.dot(tip - incisors, z) * z
    return np.linalg.inv(frame_matrix(origin, x, y, z))


def register_jaw(
    teeth: list[CTTooth], jaw: Jaw, scan: trimesh.Trimesh, scan_to_arch: Array
) -> dict[str, Any]:
    """Rigidly register CT teeth of one jaw onto a scan: global, then per tooth."""
    target = ICPTarget(np.asarray(scan.vertices, float), np.asarray(scan.vertex_normals, float))
    visible = [t for t in teeth if (t.bone_dist > REGISTRATION_BONE_MM).sum() >= MIN_CROWN_POINTS]
    src_all = np.vstack([t.crown_points() for t in visible])
    coarse_src = src_all[subsample(src_all, 3000)]
    fine_src = src_all[subsample(src_all, 20000, seed=1)]

    ct_to_arch = ct_arch_frame(teeth, jaw)
    arch_to_scan = np.linalg.inv(scan_to_arch)
    best: tuple[float, Array] | None = None
    for yaw in np.arange(-30, 31, 5):
        for dz in (-2.0, 0.0, 2.0):
            init = (
                arch_to_scan
                @ translation(np.array([0.0, 0.0, dz]))
                @ rotation_about(np.array([0.0, 0.0, 1.0]), np.radians(yaw))
                @ ct_to_arch
            )
            T, rmse = target.icp(coarse_src, init, iterations=20, trim=0.7)
            if best is None or rmse < best[0]:
                best = (rmse, T)
    assert best is not None
    T, _ = target.icp(fine_src, best[1], iterations=30, trim=0.8, max_dist=2.0)
    T, rmse = target.icp(fine_src, T, iterations=30, trim=0.8, max_dist=1.0)

    per_tooth = []
    for t in teeth:
        pts = t.crown_points()
        entry: dict[str, Any] = {
            "fdi": t.fdi,
            "transform": T,
            "visible": len(pts) >= MIN_CROWN_POINTS,
        }
        if not entry["visible"]:
            entry.update(rmse_mm=None, correction_mm=0.0, correction_deg=0.0, refined=False)
            per_tooth.append(entry)
            continue
        Tk, rk = target.icp(pts, T, iterations=20, trim=0.7, max_dist=1.0)
        delta = Tk @ np.linalg.inv(T)
        center = transform_points(T, pts.mean(axis=0)[None])[0]
        shift = float(np.linalg.norm(transform_points(delta, center[None])[0] - center))
        angle = rotation_angle_deg(delta)
        ok = shift <= TOOTH_MAX_SHIFT_MM and angle <= TOOTH_MAX_ANGLE_DEG and np.isfinite(rk)
        tooth_rmse = _rmse(target, pts, Tk if ok else T)
        entry.update(
            visible=tooth_rmse <= MAX_VISIBLE_RMSE_MM,
            transform=Tk if ok else T,
            rmse_mm=tooth_rmse,
            correction_mm=shift,
            correction_deg=angle,
            refined=ok,
        )
        per_tooth.append(entry)
    return {"global": T, "global_rmse_mm": rmse, "teeth": per_tooth}


def _rmse(target: ICPTarget, pts: Array, T: Array) -> float:
    d, _ = target.tree.query(transform_points(T, pts))
    d = np.sort(d)[: max(1, int(0.7 * len(d)))]
    return float(np.sqrt(np.mean(d**2)))


def bite_disagreement(upper_ct_to_scan: Array, lower_ct_to_scan: Array, points: Array) -> float:
    """Largest displacement (mm) of CT points (upper crowns) between the CT bite and a shared
    scan frame: 0 when the CT bite equals the scanner bite."""
    D = np.linalg.inv(lower_ct_to_scan) @ upper_ct_to_scan
    return float(np.linalg.norm(transform_points(D, points) - points, axis=1).max())
