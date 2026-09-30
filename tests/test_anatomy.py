import igl
import numpy as np
import trimesh

from aligners.anatomy import ArchCurve, close_crown, gum_heightfield, submesh


def _cap(invert: bool, hole: bool = False) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=4.0)
    keep = sphere.triangles_center[:, 2] > -1.0
    if hole:
        c = sphere.triangles_center
        keep &= ~((c[:, 2] > 3.0) & (np.abs(c[:, 0]) < 0.8))
    patch = submesh(sphere, keep)
    if invert:
        patch.invert()
    return patch


def test_close_crown_is_closed_and_outward_for_both_windings():
    for invert in (False, True):
        for hole in (False, True):
            solid = close_crown(_cap(invert, hole), np.array([0.0, 0.0, 1.0]))
            assert solid.is_watertight
            assert solid.volume > 0
            wn = np.asarray(
                igl.fast_winding_number(
                    np.asarray(solid.vertices, float),
                    np.asarray(solid.faces, np.int64),
                    np.array([[0.0, 0.0, 0.0], [0.0, 0, -2.5]]),
                )
            )
            assert np.all(wn > 0.9)


def test_arch_curve_arc_length_and_symmetry():
    half = np.array([[4.0, 0.0], [11.0, -4.0], [17.0, -10.0], [21.0, -17.0], [24.0, -26.0]])
    curve = ArchCurve.symmetric(half)
    assert abs(curve.point(0.0)[0]) < 1e-6
    right = curve.point(20.0)
    left = curve.point(-20.0)
    assert abs(right[0] + left[0]) < 0.1 and abs(right[1] - left[1]) < 0.1
    s = curve.project(right)[0]
    assert abs(s - 20.0) < 0.02


def test_gum_heightfield_fills_sockets_below_margin():
    # flat gum plane at z=0 with a tooth bump in the middle
    plane = trimesh.creation.box(extents=(20, 20, 0.01))
    plane = submesh(plane, plane.face_normals[:, 2] > 0.5)
    plane = plane.subdivide().subdivide().subdivide().subdivide()
    tooth = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
    tooth = submesh(tooth, tooth.face_normals[:, 2] > 0)
    mesh = trimesh.Trimesh(
        np.vstack([plane.vertices, tooth.vertices]),
        np.vstack([plane.faces, tooth.faces + len(plane.vertices)]),
    )
    labels = np.r_[np.zeros(len(plane.faces), int), np.full(len(tooth.faces), 11)]
    gum = gum_heightfield(mesh, labels)
    z, _ = gum.sample(np.array([[0.0, 0.0], [8.0, 8.0]]))
    assert abs(z[1]) < 0.05
    assert -1.2 < z[0] < -0.8


def test_arch_curve_projection_on_steep_legs():
    half = np.array(
        [[4.0, 0.0], [11.0, -4.0], [17.0, -10.0], [21.0, -17.0], [24.0, -26.0], [25.0, -36.0]]
    )
    curve = ArchCurve.symmetric(half)
    s0 = 35.0
    offset = curve.point(s0) + 1.0 * curve.outward(s0)[:2]
    assert abs(curve.project(offset)[0] - s0) < 0.05


def _rotated(mesh: trimesh.Trimesh, degrees: float) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), [0, 0, 1]))
    return m


def test_incisal_axis_follows_a_rotated_incisal_edge():
    from aligners.anatomy import incisal_axis

    # incisor-like crown: wide mesiodistally (x), thin labiolingually (y), tall (z)
    crown = trimesh.creation.box(extents=(8.0, 2.0, 10.0)).subdivide().subdivide()
    crown = _rotated(crown, 25.0)
    axis = incisal_axis(np.asarray(crown.vertices), np.array([0.0, 0.0, 1.0]))
    expected = np.array([np.cos(np.radians(25)), np.sin(np.radians(25)), 0.0])
    assert abs(np.dot(axis, expected)) > 0.99


def test_facial_normal_converges_on_a_rotated_canine_like_crown():
    from aligners.anatomy import facial_normal

    # canine-like crown: near-round outline, but a broad flat labial face (+y)
    cap = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    cap.apply_scale([3.5, 3.75, 5.0])
    v = cap.vertices.copy()
    v[v[:, 1] > 0, 1] *= 0.55
    cap.vertices = v
    cap = submesh(cap, cap.triangles_center[:, 2] > -2.0)
    for degrees in (15.0, 30.0, 45.0):
        rotated = _rotated(cap, degrees)
        solid = close_crown(rotated, np.array([0.0, 0.0, 1.0]))
        b = facial_normal(
            solid, len(rotated.faces), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        )
        expected = np.array([-np.sin(np.radians(degrees)), np.cos(np.radians(degrees)), 0.0])
        assert np.degrees(np.arccos(np.clip(np.dot(b, expected), -1, 1))) < 2.0
