"""Plan validation: stage limits, collisions, roots, bite and fabrication."""

from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree

from aligners import fdi
from aligners.analyze import analyze_bite, arch_to_bite, contact_pairs, posed_tooth
from aligners.anatomy import JawAnatomy
from aligners.case import Case, read_json, write_json
from aligners.fdi import JAWS, Jaw
from aligners.geometry import Array, sample_surface, transform_points
from aligners.motion import decompose

LIMIT_SLACK = 1.02
ROOT_GAP_MM = 0.5
ROOT_OUTSIDE_INCREASE = 0.15
MAX_RMSE_MM = 0.5


def _finding(severity: str, code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"severity": severity, "code": code, "message": message, **extra}


def stage_limits(
    anatomy: JawAnatomy, poses: list[dict[int, Array]], limits: dict[str, float]
) -> list[dict[str, Any]]:
    out = []
    for s in range(1, len(poses)):
        for n, t in anatomy.teeth.items():
            prev, cur = poses[s - 1][n], poses[s][n]
            mv = decompose(posed_tooth(t, prev), cur @ np.linalg.inv(prev))
            rot_limit = (
                limits["rotation_canine_premolar_deg"]
                if fdi.is_canine_or_premolar(n)
                else limits["rotation_deg"]
            )
            checks = {
                "translation": (np.hypot(mv.mesial, mv.buccal), limits["translation_mm"]),
                "extrusion": (max(mv.extrusion, 0.0), limits["extrusion_mm"]),
                "intrusion": (max(-mv.extrusion, 0.0), limits["intrusion_mm"]),
                "rotation": (abs(mv.rotation), rot_limit),
                "tip": (abs(mv.tip), limits["tip_deg"]),
                "torque": (abs(mv.torque), limits["torque_deg"]),
            }
            for name, (value, limit) in checks.items():
                if value > limit * LIMIT_SLACK + 1e-6:
                    out.append(
                        _finding(
                            "error",
                            "stage_limit",
                            f"stage {s} tooth {n}: {name} {value:.2f} exceeds {limit}",
                            stage=s,
                            tooth=n,
                        )
                    )
    return out


def _root_samples(anatomy: JawAnatomy) -> dict[int, Array]:
    out = {}
    for n, root in anatomy.roots.items():
        t = anatomy.teeth[n]
        pts = sample_surface(root, 3000, seed=6)[0]
        out[n] = pts[(pts - t.margin_center) @ t.o < -1.0]  # root part only
    return out


def roots(
    case: Case, jaw: Jaw, anatomy: JawAnatomy, final: dict[int, Array]
) -> list[dict[str, Any]]:
    """Root-root proximity at the final setup, and roots moved out of the jawbone (CT).

    Root surface samples lie on the tooth/bone interface and the CT bone label is thin
    labially, so the fraction outside bone is compared with the scanned (initial) position."""
    out = []
    frames = read_json(case.work / "frames.json")["jaws"][jaw]
    samples = _root_samples(anatomy)
    posed = {n: transform_points(final[n], pts) for n, pts in samples.items()}
    for a, b in contact_pairs(anatomy.order()):
        if a in posed and b in posed and len(posed[a]) and len(posed[b]):
            d, _ = KDTree(posed[b]).query(posed[a])
            d0, _ = KDTree(samples[b]).query(samples[a])
            if d.min() < ROOT_GAP_MM and d.min() < d0.min() - 0.1:
                out.append(
                    _finding(
                        "warning",
                        "root_proximity",
                        f"roots of {a} and {b} come within {d.min():.2f} mm",
                    )
                )
    bone = np.load(case.ct_dir / "jawbone.npz")
    mask = np.asarray(ndimage.binary_dilation(bone[jaw], iterations=2), dtype=float)
    ct_to_arch = np.asarray(frames["scan_to_arch"], float) @ np.asarray(frames["ct_to_scan"], float)
    arch_to_ct = np.linalg.inv(ct_to_arch)
    to_index = np.linalg.inv(np.asarray(bone["direction"], float)).T

    def outside(pts: Array) -> float:
        ijk = ((transform_points(arch_to_ct, pts) - bone["origin"]) @ to_index) / bone["spacing"]
        return float(
            1.0 - ndimage.map_coordinates(mask, ijk.T, order=0, mode="constant", cval=0.0).mean()
        )

    for n, pts in posed.items():
        if not len(pts):
            continue
        before, after = outside(samples[n]), outside(pts)
        if after - before > ROOT_OUTSIDE_INCREASE:
            out.append(
                _finding(
                    "warning",
                    "root_outside_bone",
                    f"tooth {n}: root surface outside the jawbone grows from {100 * before:.0f}% "
                    f"to {100 * after:.0f}%",
                    tooth=n,
                )
            )
    return out


def run_check(case: Case) -> dict[str, Any]:
    staged = read_json(case.require(case.work / "stages.json", "stage"))
    limits = case.config()["staging"]
    registration = read_json(case.work / "registration.json")
    findings: list[dict[str, Any]] = []
    anatomies = {jaw: JawAnatomy.load(case.anatomy_dir(jaw), jaw) for jaw in JAWS}
    final: dict[str, dict[int, Array]] = {}
    for jaw in JAWS:
        info = staged["jaws"][jaw]
        poses = [
            {int(n): np.asarray(T, float) for n, T in stage.items()} for stage in info["poses"]
        ]
        final[jaw] = poses[-1]
        findings += stage_limits(anatomies[jaw], poses, limits)
        for c in info["unresolved_collisions"]:
            findings.append(
                _finding(
                    "error",
                    "collision",
                    f"{jaw} stage {c['stage']}: teeth {c['pair']} overlap {c['overlap_mm']:.2f} mm",
                )
            )
        moved = {n for n, t in info["teeth"].items() if t["stages"] > 0}
        for t in registration[jaw]["teeth"]:
            if str(t["fdi"]) in moved and t["rmse_mm"] and t["rmse_mm"] > MAX_RMSE_MM:
                findings.append(
                    _finding(
                        "error",
                        "registration",
                        f"tooth {t['fdi']} moves but its CT fit is poor ({t['rmse_mm']:.2f} mm)",
                    )
                )
        findings += roots(case, jaw, anatomies[jaw], final[jaw])
    to_bite = arch_to_bite(case)
    if to_bite:
        start = analyze_bite(anatomies["upper"], anatomies["lower"], to_bite)
        end = analyze_bite(anatomies["upper"], anatomies["lower"], to_bite, final)
        before = {(i["lower"], i["upper"]): i["depth_mm"] for i in start["interferences"]}
        for i in end["interferences"]:
            if i["depth_mm"] > before.get((i["lower"], i["upper"]), 0.0) + 0.1:
                findings.append(
                    _finding(
                        "warning",
                        "occlusion",
                        f"final setup: lower {i['lower']} hits upper {i['upper']} ({i['depth_mm']:.2f} mm)",
                    )
                )
        if end.get("overjet_mm", 1.0) < 0:
            findings.append(
                _finding(
                    "warning",
                    "overjet",
                    f"final overjet {end['overjet_mm']:.1f} mm (anterior crossbite)",
                )
            )
        for n in end["posterior_crossbite"]:
            findings.append(
                _finding("warning", "crossbite", f"tooth {n} stays in posterior crossbite")
            )
        for side, m in end["molar_relationship"].items():
            if m["class"] != "I":
                findings.append(
                    _finding(
                        "warning",
                        "molar_class",
                        f"{side} molars Class {m['class']} ({m['offset_mm']:+.1f} mm): correcting this needs "
                        "elastics, which this tool does not plan",
                    )
                )
    manifest = case.out / "manifest.json"
    if manifest.exists():
        for r in read_json(manifest)["results"]:
            if r["kind"] != "model" and (not r["watertight"] or r.get("components", 1) != 1):
                findings.append(
                    _finding("error", "fabrication", f"{r['file']} is not one watertight part")
                )
    errors = [f for f in findings if f["severity"] == "error"]
    result = {"ok": not errors, "errors": len(errors), "findings": findings, "warnings": []}
    write_json(case.work / "checks.json", result)
    return result
