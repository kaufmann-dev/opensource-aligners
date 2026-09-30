"""Named render sets for humans and agents: segmentation, setup, stage:N, bite, bite:final."""

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from aligners.analyze import arch_to_bite, posed_patch
from aligners.anatomy import JawAnatomy
from aligners.case import Case, CaseError, read_json
from aligners.fdi import Jaw
from aligners.geometry import Array, transform_points
from aligners.render import Item, render, tooth_color
from aligners.scan import patient_right_sign


def _poses(case: Case, source: str, jaw: Jaw, stage: int | None = None) -> dict[int, Array]:
    if source == "initial":
        return {}
    if source == "setup":
        info = read_json(case.require(case.work / "setup.json", "setup"))["jaws"][jaw]
        return {int(n): np.asarray(t["pose"], float) for n, t in info["teeth"].items()}
    info = read_json(case.require(case.work / "stages.json", "stage"))["jaws"][jaw]
    assert stage is not None
    s = min(stage, info["stages"])
    return {int(n): np.asarray(T, float) for n, T in info["poses"][s].items()}


def _crowns(
    anatomy: JawAnatomy, poses: dict[int, Array], T: Array | None = None, opacity: float = 1.0
) -> list[Item]:
    items = []
    for n in anatomy.order():
        mesh = posed_patch(anatomy, n, poses)  # scanned surface only, no fabrication stub
        if T is not None:
            mesh = trimesh.Trimesh(
                transform_points(T, np.asarray(mesh.vertices)), mesh.faces, process=False
            )
        items.append(Item(mesh, color=tooth_color(n), opacity=opacity))
    return items


def _tags(
    anatomy: JawAnatomy, poses: dict[int, Array], T: Array | None = None
) -> list[tuple[Array, str]]:
    out = []
    for n, t in anatomy.teeth.items():
        p = transform_points(poses.get(n, np.eye(4)), t.cusp[None])[0] + t.o * 2.5
        if T is not None:
            p = transform_points(T, p[None])[0]
        out.append((p, str(n)))
    return out


def render_what(case: Case, what: str) -> dict[str, Any]:
    from aligners.segment import render_segmentation

    files: list[Path] = []
    jaws = case.jaws()
    if what == "segmentation":
        files = [render_segmentation(case, jaw) for jaw in jaws]
    elif what == "setup" or what.startswith("stage:"):
        stage = None
        if what.startswith("stage:"):
            try:
                stage = int(what.split(":", 1)[1])
            except ValueError as e:
                raise CaseError(f"bad stage in '{what}'") from e
        for jaw in jaws:
            anatomy = JawAnatomy.load(case.anatomy_dir(jaw), jaw)
            poses = _poses(case, "setup" if stage is None else "stage", jaw, stage)
            ghost = [
                Item(i.mesh, color=(0.6, 0.6, 0.6), opacity=0.25) for i in _crowns(anatomy, {})
            ]
            name = f"setup_{jaw}.png" if stage is None else f"stage{stage:02d}_{jaw}.png"
            title = (
                f"{jaw} {'target setup' if stage is None else f'stage {stage}'} (grey = initial)"
            )
            right = np.array([patient_right_sign(jaw), 0.0, 0.0])
            files.append(
                render(
                    case.renders / name,
                    ghost + _crowns(anatomy, poses),
                    _tags(anatomy, poses),
                    right,
                    title,
                )
            )
    elif what in ("bite", "bite:final"):
        to_bite = arch_to_bite(case)
        if not to_bite or len(jaws) < 2:
            raise CaseError("bite unavailable: needs both jaws and a CT or bite-registered scans")
        up = JawAnatomy.load(case.anatomy_dir("upper"), "upper")
        low = JawAnatomy.load(case.anatomy_dir("lower"), "lower")
        src = "setup" if what == "bite:final" else "initial"
        up_to_low = np.linalg.inv(to_bite["lower"]) @ to_bite["upper"]
        items = _crowns(low, _poses(case, src, "lower")) + _crowns(
            up, _poses(case, src, "upper"), up_to_low
        )
        name = "bite_final.png" if src == "setup" else "bite.png"
        files.append(
            render(case.renders / name, items, (), np.array([1.0, 0.0, 0.0]), f"bite ({src})")
        )
    else:
        raise CaseError(f"unknown render '{what}'")
    return {"files": [str(f) for f in files]}
