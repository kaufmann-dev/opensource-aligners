import numpy as np
import trimesh

from aligners.geometry import rotation_about, transform_points
from aligners.scan import cusp_plane_leveling


def test_cusp_plane_leveling_makes_the_tooth_tips_horizontal():
    # eight box "teeth" along a U, then the whole arch pitched 15 deg about x
    parts, labels = [], []
    for k, angle in enumerate(np.radians(np.linspace(-150, -30, 8))):
        box = trimesh.creation.box(extents=(4.0, 4.0, 6.0))
        box.apply_translation([20 * np.cos(angle), 20 * np.sin(angle) + 20, 3.0])
        parts.append(box)
        labels += [11 + k] * len(box.faces)
    tilt = rotation_about(np.array([1.0, 0, 0]), np.radians(15.0))
    offsets = np.cumsum([0] + [len(b.vertices) for b in parts[:-1]])
    arch = trimesh.Trimesh(
        transform_points(tilt, np.vstack([b.vertices for b in parts])),
        np.vstack([b.faces + o for b, o in zip(parts, offsets, strict=True)]),
        process=False,
    )
    R, degrees = cusp_plane_leveling(arch, np.array(labels))
    assert abs(degrees - 15.0) < 0.5
    tops = transform_points(R, np.asarray(arch.vertices))
    tips = [tops[np.unique(arch.faces[np.array(labels) == 11 + k])][:, 2].max() for k in range(8)]
    assert np.ptp(tips) < 0.05
    assert R[1, 1] > 0.9  # +y stays anterior
