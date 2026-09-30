import numpy as np
import trimesh

from aligners.anatomy import GumField
from aligners.fabricate import model_mesh, shell_mesh
from aligners.fields import Dentition, GridSDF, ToothSolid, blocked
from aligners.geometry import sample_surface

FAB = {
    "voxel_mm": 0.1,
    "model_voxel_mm": 0.15,
    "gap_mm": 0.05,
    "thickness_mm": 0.6,
    "trim_offset_mm": 1.5,
    "retention_depth_mm": 0.25,
    "base_depth_mm": 6.0,
    "engrave_depth_mm": 0.2,
    "target_faces": 100000,
}


def _dentition() -> Dentition:
    crown = trimesh.creation.icosphere(subdivisions=5, radius=3.5)  # bulge above the gingiva
    crown.apply_translation([0, 0, 3.0])
    x = np.arange(-15, 15.01, 0.1)
    height = np.zeros((len(x), len(x)), dtype=np.float32)  # flat gingiva at z = 0
    gum = GumField(height, -15.0, -15.0, 0.1)
    samples = sample_surface(crown, 200000, seed=1)[0]
    tooth = ToothSolid(GridSDF.build(crown), [], np.eye(4), samples)
    return Dentition([tooth], gum)


def test_shell_is_one_watertight_part_with_requested_gap_and_wall():
    dent = _dentition()
    frame = (
        np.array([0, 3.5 + 0.65, 2.0]),
        np.array([-1.0, 0, 0]),
        np.array([0, 0, 1.0]),
        np.array([0, 1.0, 0]),
    )
    mesh, report = shell_mesh(dent, FAB, "T1", frame)
    assert report["watertight"] and report["components"] == 1
    assert abs(report["inner_gap_mm"] - FAB["gap_mm"]) < 0.02
    assert abs(report["wall_mm"] - FAB["thickness_mm"]) < 0.03
    # trimmed: nothing farther out on the gingiva than trim_offset + wall from the crown
    r = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
    assert r.max() < 3.5 + FAB["trim_offset_mm"] + FAB["thickness_mm"] + 0.3


def test_blockout_fills_undercut_below_the_crown_bulge():
    dent = _dentition()
    block = blocked(dent, FAB["retention_depth_mm"])
    below_bulge = np.array([[2.9, 0.0, 0.5]])  # 0.45 mm outside the sphere, under its equator
    assert block(below_bulge)[0] < 0


def test_model_is_watertight():
    mesh = model_mesh(_dentition(), FAB, -6.0, "T1")
    assert mesh.is_watertight
    assert mesh.bounds[0][2] > -6.2
