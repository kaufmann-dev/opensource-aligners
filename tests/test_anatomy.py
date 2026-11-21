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


def test_incisal_edge_follows_a_sloped_edge_not_the_long_axis():
    from aligners.anatomy import incisal_edge

    # incisor-like crown whose incisal edge rises 1 mm over its 8 mm width toward +x
    crown = trimesh.creation.box(extents=(8.0, 2.0, 10.0)).subdivide().subdivide().subdivide()
    v = crown.vertices.copy()
    top = v[:, 2] > 4.9
    v[top, 2] += v[top, 0] / 8.0
    crown.vertices = v
    e = incisal_edge(v, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert abs(np.degrees(np.arctan2(e[2], e[0])) - np.degrees(np.arctan(1 / 8))) < 1.0


def test_arch_form_is_smooth_through_the_incisors_and_ignores_a_displaced_tooth():
    from aligners.anatomy import fit_arch_form

    # contact points on a symmetric U-shaped arch, one tooth blocked out 3 mm lingually
    t = np.linspace(0.0, 1.0, 9)
    half = np.stack([26.0 * np.sin(1.4 * t), -40.0 * t**1.6 - 1.0], axis=1)
    pts = np.vstack([half, half[1:] * [-1.0, 1.0]])
    normal = np.array([np.sin(1.4 * 0.375), 0.5])
    pts_bad = pts.copy()
    pts_bad[3] -= 3.0 * normal / np.linalg.norm(normal)
    anchors = {1: half[-2], -1: half[-2] * [-1.0, 1.0]}
    form = fit_arch_form(pts_bad, np.ones(len(pts)), -1.0, anchors, np.array([half[3]]))
    curve = form.curve()
    assert abs(curve.point(0.0)[1] + 1.0) < 0.05 and abs(curve.point(0.0)[0]) < 0.05
    assert np.all(form.kappa >= 0.0)
    dev = [
        abs(
            float(
                (p - curve.point(curve.project(p)[0]))
                @ curve.outward(float(curve.project(p)[0]))[:2]
            )
        )
        for p in np.delete(pts, 3, axis=0)
    ]
    assert max(dev) < 0.6
    s3 = float(curve.project(pts[3])[0])
    assert np.linalg.norm(curve.point(s3) - pts[3]) < 0.8  # the displaced tooth comes back
    moved = form.curve(1.0)
    assert abs(moved.point(0.0)[1] - curve.point(0.0)[1] - 1.0) < 0.05
