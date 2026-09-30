import numpy as np
import trimesh

from aligners.anatomy import ArchCurve, GumField, JawAnatomy, Tooth, close_crown, submesh
from aligners.motion import Movement, compose
from aligners.staging import Stager

LIMITS = {
    "translation_mm": 0.25,
    "extrusion_mm": 0.2,
    "intrusion_mm": 0.2,
    "rotation_deg": 2.0,
    "rotation_canine_premolar_deg": 1.5,
    "tip_deg": 2.0,
    "torque_deg": 1.0,
    "crown_point_mm": 0.3,
    "collision_tolerance_mm": 0.1,
}


def _tooth(number: int, x0: float) -> tuple[Tooth, trimesh.Trimesh]:
    box = trimesh.creation.box(extents=(4.0, 1.5, 6.0)).subdivide().subdivide()
    box.apply_translation([x0 + 2.0, 0.75, 3.0])
    patch = submesh(box, box.triangles_center[:, 2] > 0.01)  # open at the bottom (the margin)
    c = np.array([x0 + 2.0, 0.75, 3.0])
    t = Tooth(
        fdi=number,
        center=c,
        o=np.array([0.0, 0, 1]),
        m=np.array([-1.0, 0, 0]),
        b=np.array([0.0, 1, 0]),
        cr=c - np.array([0, 0, 6.0]),
        margin_center=c - np.array([0, 0, 3.0]),
        mesial_contact=c - np.array([2.0, 0, 0]),
        distal_contact=c + np.array([2.0, 0, 0]),
        cusp=c + np.array([0, 0, 3.0]),
        fa=c + np.array([0, 0.75, 0]),
        width=4.0,
        crown_height=6.0,
        root_length=12.0,
        root_from_ct=False,
    )
    return t, patch


def test_collision_is_resolved_by_delaying_the_tooth_moving_into_its_neighbor():
    ta, pa = _tooth(44, -4.0)
    tb, pb = _tooth(45, 0.1)
    gum = GumField(np.full((10, 10), -50.0, dtype=np.float32), -100.0, -100.0, 20.0)
    anatomy = JawAnatomy(
        "lower",
        ArchCurve(np.array([[-20.0, 0.0], [0.0, 0.0], [20.0, 0.0]])),
        {44: ta, 45: tb},
        {44: close_crown(pa, ta.o), 45: close_crown(pb, tb.o)},
        {},
        gum,
        {44: pa, 45: pb},
    )
    targets = {
        44: compose(ta, Movement(mesial=-1.0)),  # moves +x into 45's current place (4 stages)
        45: compose(tb, Movement(buccal=2.0)),  # moves +y out of the way (8 stages)
    }
    stager = Stager(anatomy, targets, LIMITS)
    assert stager.plans[44].stages == 4 and stager.plans[45].stages == 8
    assert stager.first_collision(0.5) is not None
    assert stager.schedule() == []
    assert stager.plans[44].start > 0 and stager.plans[45].start == 0
