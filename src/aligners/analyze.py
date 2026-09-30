"""Space, arch and bite measurements of (optionally posed) dentitions."""

from dataclasses import replace
from typing import Any

import numpy as np
import trimesh

from aligners import fdi
from aligners.anatomy import ArchCurve, JawAnatomy, Tooth, ideal_arch
from aligners.case import Case, read_json, write_json
from aligners.fdi import JAWS, Jaw
from aligners.geometry import Array, MeshSDF, sample_surface, transform_points, transform_vectors
from aligners.scan import patient_right_sign

Poses = dict[int, Array]
CLASS_I_MOLAR_OFFSET_MM = 2.5
POINT_FIELDS = ("center", "cr", "margin_center", "mesial_contact", "distal_contact", "cusp", "fa")
VECTOR_FIELDS = ("o", "m", "b")


def posed_tooth(tooth: Tooth, T: Array) -> Tooth:
    changes: dict[str, Any] = {
        k: transform_points(T, getattr(tooth, k)[None])[0] for k in POINT_FIELDS
    }
    changes.update({k: transform_vectors(T, getattr(tooth, k)[None])[0] for k in VECTOR_FIELDS})
    return replace(tooth, **changes)


def posed_teeth(anatomy: JawAnatomy, poses: Poses | None = None) -> dict[int, Tooth]:
    poses = poses or {}
    return {n: posed_tooth(t, poses.get(n, np.eye(4))) for n, t in anatomy.teeth.items()}


def posed_mesh(mesh: trimesh.Trimesh, T: Array) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        transform_points(T, np.asarray(mesh.vertices)), mesh.faces, process=False
    )


def posed_crown(anatomy: JawAnatomy, number: int, poses: Poses | None = None) -> trimesh.Trimesh:
    """Fabrication crown (with the stub below the margin) at its pose."""
    return posed_mesh(anatomy.crowns[number], (poses or {}).get(number, np.eye(4)))


def posed_patch(anatomy: JawAnatomy, number: int, poses: Poses | None = None) -> trimesh.Trimesh:
    """Scanned crown surface at its pose."""
    return posed_mesh(anatomy.patches[number], (poses or {}).get(number, np.eye(4)))


def contact_pairs(order: list[int]) -> list[tuple[int, int]]:
    """Adjacent teeth in arch order (patient-right distal to patient-left distal)."""
    return list(zip(order[:-1], order[1:], strict=True))


def contact_gaps(
    anatomy: JawAnatomy, poses: Poses | None = None, samples: int = 3000
) -> dict[str, float]:
    """Signed gap per adjacent pair: positive = space, negative = overlap depth (mm)."""
    out = {}
    poses = poses or {}
    for a, b in contact_pairs(anatomy.order()):
        Ta, Tb = poses.get(a, np.eye(4)), poses.get(b, np.eye(4))
        pa = transform_points(np.linalg.inv(Tb) @ Ta, anatomy.enamel_samples(a, samples, seed=1))
        pb = transform_points(np.linalg.inv(Ta) @ Tb, anatomy.enamel_samples(b, samples, seed=1))
        d_ab = MeshSDF.of(anatomy.contact_solids[b])(pa, bound=5.0).min()
        d_ba = MeshSDF.of(anatomy.contact_solids[a])(pb, bound=5.0).min()
        out[f"{a}-{b}"] = float(min(d_ab, d_ba))
    return out


def rotation_deg(tooth: Tooth, ideal: ArchCurve, jaw: Jaw) -> float:
    """Rotation about the long axis relative to the ideal arch; + = mesial edge buccal.

    The ideal mesiodistal direction is the chord of the ideal arch spanning the tooth's width
    around the projection of its contact midpoint.
    """
    side = patient_right_sign(jaw) if fdi.is_patient_right(tooth.fdi) else -patient_right_sign(jaw)
    s = float(ideal.project(((tooth.mesial_contact + tooth.distal_contact) / 2)[:2])[0])
    ideal_m = ideal.chord(s - side * tooth.width / 2, s + side * tooth.width / 2)
    ideal_b = ideal.outward(s)
    m = tooth.m - np.dot(tooth.m, [0.0, 0.0, 1.0]) * np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arctan2(np.dot(m, ideal_b), np.dot(m, ideal_m))))


def fit_occlusal_curve(points: Array) -> tuple[float, float]:
    """z = c0 + c2 x^2 through cusp tips (symmetric curve of Spee proxy)."""
    x, z = points[:, 0], points[:, 2]
    A = np.stack([np.ones_like(x), x**2], axis=1)
    (c0, c2), *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(c0), float(c2)


def _pair_sum(teeth: dict[int, Tooth], jaw: Jaw, positions: range) -> float | None:
    numbers = [fdi.fdi_number(jaw, side, p) for side in (True, False) for p in positions]
    if not all(n in teeth for n in numbers):
        return None
    return float(sum(teeth[n].width for n in numbers))


def analyze_jaw(anatomy: JawAnatomy, poses: Poses | None = None) -> dict[str, Any]:
    jaw = anatomy.jaw
    teeth = posed_teeth(anatomy, poses)
    order = anatomy.order()
    curve = ArchCurve(
        np.array([((teeth[n].mesial_contact + teeth[n].distal_contact) / 2)[:2] for n in order])
    )
    right = patient_right_sign(jaw)
    ideal = ArchCurve(ideal_arch(teeth)[0])

    # Space: on the ideal arch (symmetric anteriorly, meeting the first molars), walk each
    # side's teeth contact to contact from the arch midline and compare with where the first
    # molars' mesial contacts are. Positive = crowding (the same measure the setup balances).
    s_mid = float(ideal.project(np.array([0.0, ideal.point(0.0)[1]]))[0])
    curve_used = curve
    curve = ideal
    available = required = 0.0
    for patient_right in (True, False):
        side_teeth = [n for n in order if fdi.is_patient_right(n) == patient_right]
        molar = fdi.fdi_number(jaw, patient_right, 6)
        if molar not in teeth:
            continue
        direction = int(np.sign(float(curve.project(teeth[molar].mesial_contact[:2])[0]) - s_mid))
        s = s_mid
        for n in sorted((n for n in side_teeth if fdi.position(n) <= 5), key=fdi.position):
            s = curve.chord_step(s, direction, teeth[n].width)
        required += abs(s - s_mid)
        available += abs(float(curve.project(teeth[molar].mesial_contact[:2])[0]) - s_mid)
    curve = curve_used
    cusps = np.array([teeth[n].cusp for n in order])
    c0, c2 = fit_occlusal_curve(cusps)

    def width_between(pos: int) -> float | None:
        a, b = fdi.fdi_number(jaw, True, pos), fdi.fdi_number(jaw, False, pos)
        return (
            float(np.linalg.norm(teeth[a].cusp - teeth[b].cusp))
            if a in teeth and b in teeth
            else None
        )

    centrals = [fdi.fdi_number(jaw, s, 1) for s in (True, False)]
    midline_left = None
    if all(c in teeth for c in centrals):
        point = (teeth[centrals[0]].mesial_contact + teeth[centrals[1]].mesial_contact) / 2
        midline_left = float(-right * point[0])
    return {
        "teeth": order,
        "widths_mm": {n: teeth[n].width for n in order},
        "arch_curve": curve.as_dict(),
        "available_mm": available,
        "required_mm": required,
        "discrepancy_mm": required - available,
        "discrepancy_meaning": "positive = crowding, negative = spacing",
        "contacts_mm": contact_gaps(anatomy, poses),
        "rotation_deg": {n: rotation_deg(teeth[n], ideal, jaw) for n in order},
        "level_deviation_mm": {
            n: float(teeth[n].cusp[2] - (c0 + c2 * teeth[n].cusp[0] ** 2)) for n in order
        },
        "spee_depth_mm": float(-c2 * max(abs(cusps[:, 0])) ** 2) if len(cusps) else 0.0,
        "intercanine_mm": width_between(3),
        "intermolar_mm": width_between(6),
        "midline_offset_left_mm": midline_left,
        "anterior_sum_mm": _pair_sum(teeth, jaw, range(1, 4)),
    }


def arch_to_bite(case: Case) -> dict[Jaw, Array] | None:
    frames = read_json(case.require(case.work / "frames.json", "segment"))
    jaws = frames["jaws"]
    if frames["bite_source"] is None or not {"upper", "lower"} <= set(jaws):
        return None
    return {jaw: np.asarray(jaws[jaw]["arch_to_bite"], float) for jaw in JAWS}


def analyze_bite(
    upper: JawAnatomy,
    lower: JawAnatomy,
    to_bite: dict[Jaw, Array],
    poses: dict[Jaw, Poses] | None = None,
) -> dict[str, Any]:
    """Bite metrics expressed in the lower arch frame (+z up toward the upper arch)."""
    poses = poses or {}
    up_to_low = np.linalg.inv(to_bite["lower"]) @ to_bite["upper"]
    U = {n: posed_tooth(t, up_to_low) for n, t in posed_teeth(upper, poses.get("upper")).items()}
    L = posed_teeth(lower, poses.get("lower"))
    u_pose = {n: up_to_low @ poses.get("upper", {}).get(n, np.eye(4)) for n in U}
    l_pose = {n: poses.get("lower", {}).get(n, np.eye(4)) for n in L}
    u_crowns = {n: posed_mesh(upper.patches[n], u_pose[n]) for n in U}
    l_crowns = {n: posed_mesh(lower.patches[n], l_pose[n]) for n in L}
    out: dict[str, Any] = {}
    uc = [n for n in (11, 21) if n in U]
    lc = [n for n in (31, 41) if n in L]
    if uc and lc:
        u_edge = np.mean([U[n].cusp for n in uc], axis=0)
        l_edge = np.mean([L[n].cusp for n in lc], axis=0)
        # lower incisor labial surface at the height of the upper incisal edge
        l_pts = np.vstack([np.asarray(l_crowns[n].vertices) for n in lc])
        level = l_pts[np.abs(l_pts[:, 2] - u_edge[2]) < 0.5]
        if len(level) == 0:
            level = l_pts[l_pts[:, 2] >= l_pts[:, 2].max() - 0.5]
        l_labial = float(level[:, 1].max())
        out["overjet_mm"] = float(u_edge[1] - l_labial)
        out["overbite_mm"] = float(l_edge[2] - u_edge[2])
    if len(uc) == 2 and len(lc) == 2:
        u_mid = (U[11].mesial_contact + U[21].mesial_contact) / 2
        l_mid = (L[31].mesial_contact + L[41].mesial_contact) / 2
        # lower frame: +x is patient-right
        out["midline_upper_vs_lower_left_mm"] = float(-(u_mid[0] - l_mid[0]))
    # Molar relationship: in Class I the upper first molar sits about half a cusp distal to
    # the lower one; measure the mesial contacts along the arch (+ = upper too mesial).
    molars = {}
    for side, (u6, l6) in {"right": (16, 46), "left": (26, 36)}.items():
        if u6 in U and l6 in L:
            offset = float(
                U[u6].mesial_contact[1] - L[l6].mesial_contact[1] + CLASS_I_MOLAR_OFFSET_MM
            )
            cls = "I" if abs(offset) <= 1.5 else ("II" if offset > 0 else "III")
            molars[side] = {"offset_mm": offset, "class": cls}
    out["molar_relationship"] = molars
    crossbite = []
    for n in U:
        pos = fdi.position(n)
        if pos < 4:
            continue
        low = fdi.fdi_number("lower", fdi.is_patient_right(n), pos)
        if low not in L:
            continue
        ux = np.abs(np.asarray(u_crowns[n].vertices)[:, 0]).max()
        lx = np.abs(np.asarray(l_crowns[low].vertices)[:, 0]).max()
        if ux < lx - 0.5:
            crossbite.append(n)
    out["posterior_crossbite"] = crossbite
    interferences = []
    for n, crown in l_crowns.items():
        pts, _ = sample_surface(crown, 1500, seed=2)
        for m in U:
            sdf = MeshSDF.of(posed_mesh(upper.contact_solids[m], u_pose[m]))
            near = np.all((pts > sdf.lo - 0.5) & (pts < sdf.hi + 0.5), axis=1)
            if near.any():
                depth = -float(sdf(pts[near], bound=2.0).min())
                if depth > 0.1:
                    interferences.append({"lower": n, "upper": m, "depth_mm": depth})
    out["interferences"] = interferences
    return out


def run_analyze(case: Case) -> dict[str, Any]:
    anatomies = {jaw: JawAnatomy.load(case.anatomy_dir(jaw), jaw) for jaw in case.jaws()}
    result: dict[str, Any] = {
        "jaws": {j: analyze_jaw(a) for j, a in anatomies.items()},
        "warnings": [],
    }
    to_bite = arch_to_bite(case)
    if to_bite and {"upper", "lower"} <= set(anatomies):
        result["bite"] = analyze_bite(anatomies["upper"], anatomies["lower"], to_bite)
        up = result["jaws"]["upper"]["anterior_sum_mm"]
        low = result["jaws"]["lower"]["anterior_sum_mm"]
        if up and low:
            result["bolton_anterior_pct"] = 100.0 * low / up
    else:
        result["bite"] = None
        result["warnings"].append("bite unavailable: no CT and scans are not bite-registered")
    write_json(case.work / "analysis.json", result)
    return result
