"""Case directory layout, editable TOML inputs, and artifact IO."""

import copy
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from aligners.fdi import JAWS, Jaw


class CaseError(Exception):
    """User-facing error: bad input, missing prerequisite, or failed step."""


CASE_DEFAULTS: dict[str, Any] = {
    "inputs": {"upper": "", "lower": "", "ct": "", "ct_seg": "", "series": ""},
    "ct": {"device": "cpu"},
    "staging": {
        "translation_mm": 0.25,
        "extrusion_mm": 0.2,
        "intrusion_mm": 0.2,
        "rotation_deg": 2.0,
        "rotation_canine_premolar_deg": 1.5,
        "tip_deg": 2.0,
        "torque_deg": 1.0,
        "crown_point_mm": 0.3,
        "collision_tolerance_mm": 0.1,
    },
    "fabrication": {
        "voxel_mm": 0.1,
        "model_voxel_mm": 0.15,
        "gap_mm": 0.05,
        "thickness_mm": 0.6,
        "trim_offset_mm": 1.5,
        "retention_depth_mm": 0.25,
        "base_depth_mm": 6.0,
        "engrave_depth_mm": 0.2,
        "target_faces": 250000,
    },
}

CASE_TEMPLATE = """\
# aligners case configuration. Experimental, not for clinical use.

[inputs]              # required: both scans and a CT (or its TotalSegmentator labelmap)
upper = {upper}     # upper jaw intraoral scan (STL/PLY/OBJ)
lower = {lower}     # lower jaw intraoral scan; export both scans in occlusion for the best bite
ct = {ct}        # DICOM directory or CT NIfTI ("" when ct_seg is given)
ct_seg = {ct_seg}    # existing TotalSegmentator `teeth` multilabel NIfTI, "" to compute it
series = {series}    # DICOM SeriesInstanceUID or SeriesNumber override, "" = automatic

[ct]
device = "cpu"      # TotalSegmentator device: "cpu" or "gpu"

[staging]           # maximum movement per aligner stage
translation_mm = 0.25                # mesiodistal / buccolingual, at the center of resistance
extrusion_mm = 0.2
intrusion_mm = 0.2
rotation_deg = 2.0                   # about the long axis
rotation_canine_premolar_deg = 1.5
tip_deg = 2.0
torque_deg = 1.0
crown_point_mm = 0.3                 # any crown surface point
collision_tolerance_mm = 0.1         # allowed intermediate overlap beyond the scanned/final contact

[fabrication]
voxel_mm = 0.1                       # aligner shell field resolution
model_voxel_mm = 0.15                # thermoforming model field resolution
gap_mm = 0.05                        # inner clearance between tooth and shell
thickness_mm = 0.6                   # shell wall thickness
trim_offset_mm = 1.5                 # shell coverage onto the gingiva beyond the crowns
retention_depth_mm = 0.25            # undercut depth kept for retention; deeper undercuts are blocked out
base_depth_mm = 6.0                  # model base height below the lowest gingival margin
engrave_depth_mm = 0.2               # stage label depth on shells
target_faces = 250000                # decimation target per exported mesh
"""

PLAN_DEFAULTS: dict[str, Any] = {
    "goals": {
        "align": True,
        "derotate": True,
        "tipping": True,
        "tipping_pivot": 0.7,
        "harmonize_inclination": True,
        "level_incisal_edges": True,
        "level": True,
        "spee_flatten": 0.0,
        "close_spaces": True,
        "anchors": ["16", "17", "18", "26", "27", "28", "36", "37", "38", "46", "47", "48"],
        "expand_canine_mm": 0.0,
        "expand_premolar_mm": 0.0,
        "expand_molar_mm": 0.0,
        "midline_shift_mm": {"upper": 0.0, "lower": 0.0},
        "max_incisor_advance_mm": 1.5,
        "max_incisor_retract_mm": 2.0,
        "ipr_max_anterior_mm": 0.3,
        "ipr_max_posterior_mm": 0.5,
        "ipr_max_total_mm": 3.0,
        "bite": {
            "min_overjet_mm": 0.5,
            "min_overbite_mm": 1.0,
            "enforce": False,
            "overjet_mm": 2.5,
            "overbite_mm": 2.0,
            "overbite_split": 0.5,
            "align_midlines": True,
        },
    },
    "teeth": {},
    "attachments": {"auto": True, "add": [], "remove": []},
}

TOOTH_OVERRIDE_KEYS = {
    "mesial_mm": 0.0,
    "buccal_mm": 0.0,
    "extrusion_mm": 0.0,
    "rotation_deg": 0.0,
    "tip_deg": 0.0,
    "torque_deg": 0.0,
    "fixed": False,
}

PLAN_TEMPLATE = """\
# aligners treatment plan. Edited by you or an AI agent; `aligners setup` solves it.
# Experimental, not for clinical use.

[goals]
align = true              # place teeth on a smooth ideal arch with touching contacts
derotate = true           # turn each tooth so its mesiodistal axis follows the arch
tipping = true            # move crowns outward/inward by tipping (changes inclination), not bodily
tipping_pivot = 0.7       # tipping center: fraction of the root length below the gingival margin
harmonize_inclination = true  # give left/right homologous teeth the same outward/inward inclination
level_incisal_edges = true    # tip incisors so their incisal edges are horizontal (straight smile line)
level = true              # bring cusp tips / incisal edges onto a smooth occlusal curve
spee_flatten = 0.0        # 0 = keep curve of Spee depth, 1 = flatten it completely
close_spaces = true       # close spacing by retracting incisors (within max_incisor_retract_mm)
anchors = ["16", "17", "18", "26", "27", "28", "36", "37", "38", "46", "47", "48"]  # keep place and occlusion; follow the arch transversely (expansion); remove to align them fully
expand_canine_mm = 0.0    # total intercanine width change (both sides together)
expand_premolar_mm = 0.0  # total interpremolar width change
expand_molar_mm = 0.0     # total intermolar width change (moves anchors transversely)
midline_shift_mm = { upper = 0.0, lower = 0.0 }  # + = toward patient-left
max_incisor_advance_mm = 1.5   # incisor proclination allowed to resolve crowding
max_incisor_retract_mm = 2.0   # incisor retraction allowed to close spaces
ipr_max_anterior_mm = 0.3      # interproximal reduction per contact, incisors/canines
ipr_max_posterior_mm = 0.5     # per contact, premolars/molars
ipr_max_total_mm = 3.0         # per arch

[goals.bite]              # the bite comes from bite-registered scans, else from the CT
min_overjet_mm = 0.5      # always kept: upper incisors advance if the setup would go below this
min_overbite_mm = 1.0     # always kept: incisors extrude (split below) if the bite would open
enforce = false           # true = also correct overjet/overbite to the targets below
overjet_mm = 2.5
overbite_mm = 2.0
overbite_split = 0.5      # share of overbite correction done by upper incisors
align_midlines = true     # move the upper dental midline onto the lower one

# Per-tooth overrides in the tooth's own frame, pivoting about its center of resistance.
# Applied on top of the solved target. Signs: mesial + toward midline, buccal + labial,
# extrusion + occlusal, rotation + mesial edge buccally, tip + crown mesially,
# torque + crown buccally (root lingually). `fixed = true` keeps a tooth exactly as scanned.
# [teeth."13"]
# rotation_deg = 5.0
# fixed = false

[attachments]
auto = true               # add attachments where movements need grip
add = []                  # e.g. ["13", "23"]
remove = []
"""


def _toml_str(value: str) -> str:
    return json.dumps(value)


def _merge(defaults: dict[str, Any], given: dict[str, Any], where: str) -> dict[str, Any]:
    out = copy.deepcopy(defaults)
    for key, value in given.items():
        if key not in defaults:
            raise CaseError(f"unknown key '{where}{key}'")
        if isinstance(defaults[key], dict) and defaults[key] and isinstance(value, dict):
            out[key] = _merge(defaults[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    teeth: dict[str, Any] = {}
    for fdi, override in plan["teeth"].items():
        if not isinstance(override, dict):
            raise CaseError(f'[teeth."{fdi}"] must be a table')
        teeth[str(fdi)] = _merge(TOOTH_OVERRIDE_KEYS, override, f"teeth.{fdi}.")
    plan["teeth"] = teeth
    return plan


@dataclass(frozen=True)
class Case:
    root: Path

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def ct_dir(self) -> Path:
        return self.work / "ct"

    @property
    def scan_dir(self) -> Path:
        return self.work / "scan"

    @property
    def renders(self) -> Path:
        return self.work / "renders"

    def anatomy_dir(self, jaw: Jaw) -> Path:
        return self.work / "anatomy" / jaw

    @property
    def config_path(self) -> Path:
        return self.root / "case.toml"

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.toml"

    def config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise CaseError(f"{self.config_path} not found; run `aligners init` first")
        with self.config_path.open("rb") as f:
            return _merge(CASE_DEFAULTS, tomllib.load(f), "")

    def plan(self) -> dict[str, Any]:
        if not self.plan_path.exists():
            raise CaseError(f"{self.plan_path} not found; run `aligners init` first")
        with self.plan_path.open("rb") as f:
            return _validate_plan(_merge(PLAN_DEFAULTS, tomllib.load(f), ""))

    def jaws(self) -> list[Jaw]:
        return list(JAWS)

    def require(self, path: Path, step: str) -> Path:
        if not path.exists():
            raise CaseError(f"{path.relative_to(self.root)} missing; run `aligners {step}` first")
        return path


def init_case(root: Path, upper: str, lower: str, ct: str, ct_seg: str, series: str) -> Case:
    if not (upper and lower and (ct or ct_seg)):
        raise CaseError(
            "required inputs: --upper and --lower intraoral scans plus --ct (DICOM or NIfTI) or "
            "--ct-seg (TotalSegmentator `teeth` labelmap); bite-aligned scans give the best bite"
        )
    for label, value in (
        ("--upper", upper),
        ("--lower", lower),
        ("--ct", ct),
        ("--ct-seg", ct_seg),
    ):
        if value and not Path(value).exists():
            raise CaseError(f"{label} path does not exist: {value}")
    case = Case(root)
    if case.config_path.exists():
        raise CaseError(f"{case.config_path} already exists")
    root.mkdir(parents=True, exist_ok=True)

    def resolved(value: str) -> str:
        return str(Path(value).resolve()) if value else ""

    case.config_path.write_text(
        CASE_TEMPLATE.format(
            upper=_toml_str(resolved(upper)),
            lower=_toml_str(resolved(lower)),
            ct=_toml_str(resolved(ct)),
            ct_seg=_toml_str(resolved(ct_seg)),
            series=_toml_str(series),
        )
    )
    case.plan_path.write_text(PLAN_TEMPLATE)
    case.config()
    case.plan()
    return case


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_default))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def to_jsonable(data: Any) -> Any:
    return json.loads(json.dumps(data, default=_default))


def write_mesh(path: Path, mesh: trimesh.Trimesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def read_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise CaseError(f"{path} is not a triangle mesh")
    return mesh
