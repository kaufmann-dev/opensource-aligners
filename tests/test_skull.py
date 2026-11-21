import numpy as np
import SimpleITK as sitk

from aligners.case import Case, read_json
from aligners.ct import Volume
from aligners.geometry import transform_points
from aligners.skull import build_head_frame, ct_to_head

YAW = np.radians(6.0)


def _phantom(tmp_path) -> tuple[Case, Volume, np.ndarray]:
    """A skull-like CT in LPS: a bone shell with a nose (anterior = -y) whose midsagittal plane
    is turned 6 degrees about the superior axis, and an upper tooth label below it."""
    n, res = 90, 2.0
    origin = -np.array([n, n, n]) * res / 2
    idx = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1).astype(float)
    p = origin + idx * res
    c, s = np.cos(YAW), np.sin(YAW)
    x, y = c * p[..., 0] + s * p[..., 1], -s * p[..., 0] + c * p[..., 1]
    z = p[..., 2]
    r = np.sqrt((x / 60) ** 2 + (y / 70) ** 2 + ((z - 10) / 55) ** 2)
    ct = np.where(r < 1.0, 40.0, -1000.0)
    ct[(r > 0.85) & (r < 1.0)] = 1200.0  # shell
    ct[(np.abs(x) < 6) & (y < -55) & (y > -75) & (z > 5) & (z < 35)] = 1200.0  # nose
    ct[(x > 15) & (x < 30) & (y > 20) & (y < 40) & (z > 30) & (z < 45)] = 1200.0  # a ridge,
    ct[(x < -15) & (x > -30) & (y > 20) & (y < 40) & (z > 30) & (z < 45)] = 1200.0  # mirrored
    labels = np.zeros(ct.shape, dtype=np.int32)
    labels[(np.abs(x) < 10) & (y < -30) & (y > -50) & (z > -40) & (z < -30)] = 11  # tooth 11
    image = sitk.GetImageFromArray(ct.transpose(2, 1, 0).astype(np.int16))
    image.SetOrigin(tuple(origin))
    image.SetSpacing((res, res, res))
    case = Case(tmp_path)
    case.ct_dir.mkdir(parents=True)
    sitk.WriteImage(image, str(case.ct_dir / "ct.nii.gz"))
    seg = Volume(labels, origin, np.full(3, res), np.eye(3))
    return case, seg, labels


def test_head_frame_finds_the_midsagittal_plane_and_patient_right(tmp_path):
    case, seg, labels = _phantom(tmp_path)
    info = build_head_frame(case, seg, labels)
    normal = np.asarray(read_json(case.ct_dir / "skull.json")["plane_normal"])
    truth = np.array([-np.cos(YAW), -np.sin(YAW), 0.0])  # LPS -x = patient right, turned
    assert normal @ truth > 0.999
    assert info["symmetry_rmse_mm"] < 1.5
    assert (case.ct_dir / "bone.npz").exists()


def test_ct_to_head_axes_and_pitch():
    skull = {
        "plane_normal": np.array([-1.0, 0.0, 0.0]),
        "plane_point": np.array([5.0, 0.0, 0.0]),
        "ct_superior": np.array([0.0, 0.0, 1.0]),
    }
    H = ct_to_head(skull, 0.0, np.array([5.0, 0.0, 0.0]))
    # LPS: -x = patient right, -y = anterior, +z = superior
    pts = np.array([[5.0 - 10, 0, 0], [5.0, -10, 0], [5.0, 0, 10]])
    assert np.allclose(transform_points(H, pts), [[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    tilted = ct_to_head(skull, 5.0, np.array([5.0, 0.0, 0.0]))
    # + pitch tilts the head horizontal anterior-up: a CT-horizontal anterior point drops
    assert transform_points(tilted, pts[1:2])[0, 2] < -0.5
