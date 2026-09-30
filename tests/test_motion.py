import numpy as np

from aligners.anatomy import Tooth
from aligners.motion import Movement, compose, decompose, interpolate


def _tooth() -> Tooth:
    z = np.zeros(3)
    return Tooth(
        fdi=11,
        center=np.array([0.0, 0, 5]),
        o=np.array([0.0, 0, 1]),
        m=np.array([-1.0, 0, 0]),
        b=np.array([0.0, 1, 0]),
        cr=np.array([0.0, 0, -4]),
        margin_center=z,
        mesial_contact=z,
        distal_contact=z,
        cusp=np.array([0.0, 0, 9]),
        fa=z,
        width=8.0,
        crown_height=9.0,
        root_length=13.0,
        root_from_ct=False,
    )


def test_compose_decompose_roundtrip():
    t = _tooth()
    mv = Movement(0.5, -0.3, 0.2, 4.0, -2.0, 1.5)
    back = decompose(t, compose(t, mv))
    for a, b in zip(mv.as_dict().values(), back.as_dict().values(), strict=True):
        assert abs(a - b) < 1e-6


def test_sign_conventions():
    t = _tooth()
    T = compose(t, Movement(tip=10.0))
    crown_tip = T[:3, :3] @ t.cusp + T[:3, 3]
    assert crown_tip[0] < 0  # mesial is -x for this tooth: crown tips mesially
    T = compose(t, Movement(torque=10.0))
    assert (T[:3, :3] @ t.cusp + T[:3, 3])[1] > 0  # crown moves buccally


def test_interpolate_endpoints():
    t = _tooth()
    T = compose(t, Movement(1.0, 0, 0, 20.0, 0, 0))
    assert np.allclose(interpolate(t, T, 0.0), np.eye(4))
    assert np.allclose(interpolate(t, T, 1.0), T)
    half = decompose(t, interpolate(t, T, 0.5))
    assert abs(half.rotation - 10.0) < 1e-6 and abs(half.mesial - 0.5) < 1e-6
