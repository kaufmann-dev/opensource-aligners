"""Printable outputs: aligner shells, thermoforming models, bonding templates, manifest."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

from aligners import fdi
from aligners.analyze import posed_teeth
from aligners.anatomy import JawAnatomy, Tooth
from aligners.case import Case, CaseError, read_json, write_json
from aligners.fdi import JAWS, Jaw
from aligners.fields import (
    Dentition,
    Grid,
    GridSDF,
    ToothSolid,
    banded,
    blocked,
    edt,
    extract,
    label_field,
)
from aligners.geometry import Array, sample_surface, transform_points
from aligners.motion import decompose

ATTACH_PROTRUSION_MM = 0.8
ATTACH_LONG_MM = 3.0
ATTACH_SHORT_MM = 2.0
LABEL_HEIGHT_MM = 2.5
BASE_LABEL_HEIGHT_MM = 3.5
HORSESHOE_MM = 7.0
GROOVE_WIDTH_MM = 0.4
GROOVE_DEPTH_MM = 0.3
BASE_LABEL_DEPTH_MM = 0.6
DISCLAIMER = "Experimental, not for clinical use."


# -- attachments ------------------------------------------------------------------------------
def plan_attachments(
    anatomy: JawAnatomy, targets: dict[int, Array], plan: dict[str, Any]
) -> dict[int, str]:
    """Attachment type per tooth: 'vertical' (rotation/root control) or 'horizontal' (extrusion)."""
    cfg = plan["attachments"]
    out: dict[int, str] = {}
    if cfg["auto"]:
        for n, t in anatomy.teeth.items():
            mv = decompose(t, targets[n])
            if mv.extrusion > 0.5:
                out[n] = "horizontal"
            elif (fdi.is_canine_or_premolar(n) and abs(mv.rotation) > 5.0) or max(
                abs(mv.tip), abs(mv.torque)
            ) > 8.0:
                out[n] = "vertical"
    for n in cfg["add"]:
        if int(n) in anatomy.teeth:
            out.setdefault(int(n), "vertical")
    for n in cfg["remove"]:
        out.pop(int(n), None)
    return out


def attachment_box(t: Tooth, kind: str) -> tuple[Array, Array, Array]:
    """(center, axes as columns (m, o, b), half extents) in the tooth's initial frame.

    The box reaches 1 mm into the crown so it merges with the curved surface."""
    along_m, along_o = (
        (ATTACH_SHORT_MM, ATTACH_LONG_MM)
        if kind == "vertical"
        else (ATTACH_LONG_MM, ATTACH_SHORT_MM)
    )
    half = np.array([along_m / 2, along_o / 2, (ATTACH_PROTRUSION_MM + 1.0) / 2])
    center = t.fa + t.b * (ATTACH_PROTRUSION_MM - 1.0) / 2
    return center, np.stack([t.m, t.o, t.b], axis=1), half


def crown_sdfs(case: Case, jaw: Jaw, anatomy: JawAnatomy) -> dict[int, GridSDF]:
    """Per-crown distance grids, cached in work/sdf/<jaw>/ (rebuilt when the crown changes)."""
    out = {}
    directory = case.work / "sdf" / jaw
    directory.mkdir(parents=True, exist_ok=True)
    for n in anatomy.order():
        path = directory / f"{n}.npz"
        crown = case.anatomy_dir(jaw) / f"{n}_crown.ply"
        if path.exists() and path.stat().st_mtime >= crown.stat().st_mtime:
            out[n] = GridSDF.load(path)
        else:
            out[n] = GridSDF.build(anatomy.crowns[n])
            out[n].save(path)
    return out


def dentition(
    anatomy: JawAnatomy,
    sdfs: dict[int, GridSDF],
    poses: dict[int, Array],
    attachments: dict[int, str],
) -> Dentition:
    teeth = []
    for n in anatomy.order():
        boxes = [attachment_box(anatomy.teeth[n], attachments[n])] if n in attachments else []
        teeth.append(
            ToothSolid(
                sdfs[n],
                boxes,
                poses.get(n, np.eye(4)),
                sample_surface(anatomy.crowns[n], 300000, seed=7)[0],
            )
        )
    return Dentition(teeth, anatomy.gum)


# -- solids -----------------------------------------------------------------------------------
def _d_block(dent: Dentition, block: Any, pts: Array) -> tuple[Array, Array]:
    dt = dent.d_teeth(pts, bound=4.0)
    db = np.minimum(np.minimum(dt, dent.d_gum(pts)), block(pts))
    return dt, db


def shell_mesh(
    dent: Dentition, fab: dict[str, Any], label: str, label_frame: tuple[Array, Array, Array, Array]
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    gap, thick, trim = fab["gap_mm"], fab["thickness_mm"], fab["trim_offset_mm"]
    lo, hi = dent.bounds(gap + thick + trim + 1.0)
    block = blocked(dent, fab["retention_depth_mm"])
    center, right, up, normal = label_frame
    glyph = label_field(label, LABEL_HEIGHT_MM, center, right, up)
    outer = gap + thick

    def field(pts: Array) -> Array:
        dt, db = _d_block(dent, block, pts)
        s = np.maximum(np.maximum(gap - db, db - outer), dt - db - trim)
        w = (pts - center) @ normal
        engraving = np.maximum(
            np.maximum(glyph(pts), (outer - fab["engrave_depth_mm"]) - db), np.abs(w) - 2.0
        )
        return np.maximum(s, -engraving)

    grid = Grid.around(lo, hi, fab["voxel_mm"])
    values = banded(grid, field, (-0.4, 0.4), coarse_factor=3)
    mesh = extract(grid, values, fab["target_faces"])
    _, db = _d_block(dent, block, np.asarray(mesh.vertices))
    inner = db[db < gap + thick / 2]
    outer_v = db[db >= gap + thick / 2]
    report = {
        "watertight": bool(mesh.is_watertight),
        "components": len(mesh.split(only_watertight=False)),
        "inner_gap_mm": float(np.median(inner)) if len(inner) else None,
        "wall_mm": float(np.median(outer_v) - np.median(inner))
        if len(inner) and len(outer_v)
        else None,
        "faces": len(mesh.faces),
    }
    return mesh, report


def model_mesh(dent: Dentition, fab: dict[str, Any], z_base: float, label: str) -> trimesh.Trimesh:
    res = fab["model_voxel_mm"]
    lo, hi = dent.bounds(1.0)
    lo[:2] -= HORSESHOE_MM
    hi[:2] += HORSESHOE_MM
    lo[2] = z_base - 0.5
    block = blocked(dent, fab["retention_depth_mm"])
    # horseshoe: keep material within HORSESHOE_MM (in the occlusal plane) of any crown
    g2 = 0.2
    nx, ny = int(np.ceil((hi[0] - lo[0]) / g2)) + 1, int(np.ceil((hi[1] - lo[1]) / g2)) + 1
    foot = np.zeros((nx, ny), dtype=bool)
    for t in dent.teeth:
        v = transform_points(t.pose, t.vertices)
        ij = np.clip(((v[:, :2] - lo[:2]) / g2).astype(int), 0, [nx - 1, ny - 1])
        foot[ij[:, 0], ij[:, 1]] = True
    foot = np.asarray(ndimage.binary_closing(foot, iterations=5), dtype=bool)
    horse = edt(~foot) * g2 - HORSESHOE_MM
    label_center = np.array([0.0, float(lo[1] + (hi[1] - lo[1]) * 0.75), z_base])
    glyph = label_field(
        label,
        BASE_LABEL_HEIGHT_MM,
        label_center,
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    trim = fab["trim_offset_mm"]

    def field(pts: Array) -> Array:
        dt, db = _d_block(dent, block, pts)
        h = ndimage.map_coordinates(horse, ((pts[:, :2] - lo[:2]) / g2).T, order=1, mode="nearest")
        m = np.maximum(np.maximum(db, z_base - pts[:, 2]), h)
        t = dt - db - trim
        groove = np.maximum(np.abs(t) - GROOVE_WIDTH_MM / 2, -GROOVE_DEPTH_MM - db)
        engraving = np.maximum(glyph(pts), pts[:, 2] - (z_base + BASE_LABEL_DEPTH_MM))
        return np.maximum(np.maximum(m, -groove), -engraving)

    grid = Grid.around(lo, hi, res)
    values = banded(grid, field, (-0.4, 0.4), coarse_factor=4)
    return extract(grid, values, fab["target_faces"])


# -- export -----------------------------------------------------------------------------------
def _label_frame(
    anatomy: JawAnatomy, poses: dict[int, Array], fab: dict[str, Any]
) -> tuple[Array, Array, Array, Array]:
    """Engraving frame on the outer shell surface over the buccal face of the last molar."""
    teeth = posed_teeth(anatomy, poses)
    order = anatomy.order()
    n = order[0] if fdi.position(order[0]) >= fdi.position(order[-1]) else order[-1]
    t = teeth[n]
    b = t.b / np.linalg.norm(t.b)
    up = t.o / np.linalg.norm(t.o)
    right = np.cross(up, b)
    right /= np.linalg.norm(right)
    center = t.fa + b * (fab["gap_mm"] + fab["thickness_mm"])
    return center, right, up, b


def _task(args: tuple[str, str, int, str]) -> dict[str, Any]:
    root, jaw, stage, kind = args
    case = Case(Path(root))
    fab = case.config()["fabrication"]
    anatomy = JawAnatomy.load(case.anatomy_dir(jaw), jaw)
    stages = read_json(case.work / "stages.json")["jaws"][jaw]
    export = read_json(case.work / "export_plan.json")[jaw]
    poses = {int(n): np.asarray(T, float) for n, T in stages["poses"][stage].items()}
    attachments = {int(n): k for n, k in export["attachments"].items()}
    dent = dentition(anatomy, crown_sdfs(case, jaw, anatomy), poses, attachments)
    shift = np.array([0.0, 0.0, -export["z_base"]])
    prefix = "U" if jaw == "upper" else "L"
    out_dir = case.out / jaw
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "model":
        mesh = model_mesh(dent, fab, export["z_base"], f"{prefix}{stage:02d}")
        report: dict[str, Any] = {"watertight": bool(mesh.is_watertight)}
        name = f"model_{stage:02d}.stl"
    else:
        text = f"{prefix}T" if kind == "template" else f"{prefix}{stage:02d}"
        mesh, report = shell_mesh(dent, fab, text, _label_frame(anatomy, poses, fab))
        name = "template.stl" if kind == "template" else f"aligner_{stage:02d}.stl"
    mesh.apply_translation(shift)
    mesh.export(out_dir / name)
    return {"jaw": jaw, "stage": stage, "kind": kind, "file": f"{jaw}/{name}", **report}


def run_export(case: Case, stages: str = "", jobs: int = 4) -> dict[str, Any]:
    staged = read_json(case.require(case.work / "stages.json", "stage"))
    setup = read_json(case.require(case.work / "setup.json", "setup"))
    frames = read_json(case.work / "frames.json")
    plan = case.plan()
    wanted: set[int] | None = None
    if stages:
        try:
            a, _, b = stages.partition("-")
            wanted = set(range(int(a), int(b or a) + 1))
        except ValueError as e:
            raise CaseError(f"bad --stages '{stages}' (use e.g. 3 or 1-5)") from e
    export_plan: dict[str, Any] = {}
    tasks: list[tuple[str, str, int, str]] = []
    for jaw in JAWS:
        anatomy = JawAnatomy.load(case.anatomy_dir(jaw), jaw)
        targets = {
            int(n): np.asarray(t["pose"], float) for n, t in setup["jaws"][jaw]["teeth"].items()
        }
        attachments = plan_attachments(anatomy, targets, plan)
        z_base = (
            min(float(t.margin_center[2]) for t in anatomy.teeth.values())
            - case.config()["fabrication"]["base_depth_mm"]
        )
        export_plan[jaw] = {"attachments": attachments, "z_base": z_base}
        crown_sdfs(case, jaw, anatomy)  # build the cache once, before the workers start
        count = staged["jaws"][jaw]["stages"]
        for s in range(0, count + 1):
            if wanted is not None and s not in wanted:
                continue
            tasks.append((str(case.root), jaw, s, "model"))
            tasks.append((str(case.root), jaw, s, "template" if s == 0 else "aligner"))
    write_json(case.work / "export_plan.json", export_plan)
    if wanted is None:
        # a full export replaces every stage file: stale stages of an older plan must not remain
        for jaw in JAWS:
            for old in (case.out / jaw).glob("*.stl"):
                old.unlink()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=_mp_context()) as pool:
            results = list(pool.map(_task, tasks))
    else:
        results = [_task(t) for t in tasks]

    manifest: dict[str, Any] = {"disclaimer": DISCLAIMER, "jaws": {}, "parameters": case.config()}
    warnings = []
    for jaw in JAWS:
        scan_to_arch = np.asarray(frames["jaws"][jaw]["scan_to_arch"], float)
        shift = np.eye(4)
        shift[2, 3] = -export_plan[jaw]["z_base"]
        manifest["jaws"][jaw] = {
            "stages": staged["jaws"][jaw]["stages"],
            "files": sorted(r["file"] for r in results if r["jaw"] == jaw),
            "ipr_before_stage_1_mm": staged["jaws"][jaw]["ipr_mm"],
            "attachments": {
                str(n): {"type": k, "bond": "before stage 1 with template.stl"}
                for n, k in export_plan[jaw]["attachments"].items()
            },
            "output_to_scan": np.linalg.inv(shift @ scan_to_arch),
            "wear": "each aligner 10-14 days, in order; model_00 and template are the starting state",
        }
    for r in results:
        if r["kind"] != "model":
            if not r["watertight"] or r["components"] != 1:
                warnings.append(f"{r['file']}: not a single watertight part")
            if (
                r["inner_gap_mm"] is not None
                and abs(r["inner_gap_mm"] - manifest["parameters"]["fabrication"]["gap_mm"]) > 0.05
            ):
                warnings.append(f"{r['file']}: measured inner gap {r['inner_gap_mm']:.2f} mm")
    manifest["results"] = results
    write_json(case.out / "manifest.json", manifest)
    return {"files": [r["file"] for r in results], "results": results, "warnings": warnings}


def _mp_context() -> Any:
    import multiprocessing

    return multiprocessing.get_context("spawn" if os.name == "nt" else "forkserver")
