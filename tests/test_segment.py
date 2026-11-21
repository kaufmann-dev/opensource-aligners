import itertools

import numpy as np
import trimesh

from aligners.anatomy import submesh
from aligners.segment import alpha_expansion, cleanup, fill_enclosed_gum, snap_gumline


def _energy(lab, unary, pairs, weights):
    return (
        unary[np.arange(len(lab)), lab].sum()
        + (weights * (lab[pairs[:, 0]] != lab[pairs[:, 1]])).sum()
    )


def test_single_expansion_move_is_optimal():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n, n_labels = 7, 3
        unary = rng.uniform(0, 3, (n, n_labels))
        pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5])
        weights = rng.uniform(0, 2, len(pairs))
        init = rng.integers(0, n_labels, n)
        active = np.ones(n, dtype=bool)
        active[0] = False  # one fixed node exercises the border terms
        # brute-force best expansion move for alpha = 0 starting from init
        best = np.inf
        free = np.flatnonzero(active)
        for switch in itertools.product([0, 1], repeat=len(free)):
            lab = init.copy()
            lab[free[np.array(switch, bool)]] = 0
            best = min(best, _energy(lab, unary, pairs, weights))
        # one cycle but only alpha = 0 matters for the comparison: restrict unary to force it
        result = alpha_expansion(unary, pairs, weights, init, active, cycles=1)
        assert _energy(result, unary, pairs, weights) <= best + 1e-9
        assert result[0] == init[0]


def test_cleanup_keeps_one_region_per_label():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    z = mesh.triangles_center[:, 2]
    labels = np.where(z > 0.3, 11, 0)
    stray = np.flatnonzero(z < -0.9)[:3]
    labels[stray] = 11
    out, _ = cleanup(mesh, labels)
    assert np.all(out[stray] == 0)
    assert np.all(out[z > 0.35] == 11)


def test_mesh_sdf_bounded_keeps_sign_beyond_bound():
    from aligners.geometry import MeshSDF

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=3.0)
    sdf = MeshSDF.of(sphere)
    d = sdf(np.array([[0.0, 0, 0], [0, 0, 2.9], [0, 0, 8.0]]), bound=1.0)
    assert d[0] == -1.0 and d[2] == 1.0 and abs(d[1] + 0.1) < 0.05


def test_cleanup_keeps_separate_large_gum_regions():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=5.0)
    z = mesh.triangles_center[:, 2]
    labels = np.where(np.abs(z) < 1.0, 11, 0)  # a tooth belt splits the gum into two caps
    out, _ = cleanup(mesh, labels)
    assert np.all(out[z > 1.2] == 0) and np.all(out[z < -1.2] == 0)


def test_snap_gumline_moves_the_boundary_onto_the_crease():
    # "gingiva" plane with a "crown" bump; the crown meets the plane in a concave crease at
    # r = 3 mm, and the initial labels spill 0.6 mm onto the gingiva
    box = trimesh.creation.box(extents=(16.0, 16.0, 1.0))
    fine = trimesh.remesh.subdivide_to_size(box.vertices, box.faces, 0.25)
    v, faces = fine[0], fine[1]
    top = v[:, 2] > 0.4
    r = np.hypot(v[top, 0], v[top, 1])
    v[top, 2] += np.where(r < 3.0, 4.0 * np.sqrt(np.clip(1 - (r / 3.0) ** 2, 0, 1)) + 0.3, 0.0)
    mesh = trimesh.Trimesh(v, faces, process=False)
    c = mesh.triangles_center
    upper = c[:, 2] > 0.45
    rc = np.hypot(c[:, 0], c[:, 1])
    labels = np.where(upper & (rc < 3.6), 11, 0)
    out = snap_gumline(mesh, labels)
    ring = upper & (rc > 3.25) & (rc < 3.6)
    assert np.mean(out[ring] == 0) > 0.9  # the spilled band returned to gingiva
    assert np.all(out[upper & (rc < 2.5)] == 11)


def test_fill_enclosed_gum_fills_holes_in_one_crown_only():
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=5.0)
    mesh = submesh(sphere, sphere.triangles_center[:, 2] > -4.0)  # open at the bottom
    c = mesh.triangles_center
    labels = np.where(c[:, 2] > 0.0, 11, 0)
    labels[(c[:, 2] > 0.0) & (c[:, 0] > 0)] = 21
    hole = np.hypot(c[:, 0] - 2.5, c[:, 1]) < 1.0
    hole &= c[:, 2] > 3.0
    labels[hole] = 0  # gingiva island inside 21
    between = (np.abs(c[:, 0]) < 0.6) & (np.abs(c[:, 2] - 4.0) < 0.6)
    labels[between] = 0  # gingiva touching 11 and 21 stays
    out = fill_enclosed_gum(mesh, labels)
    assert np.all(out[hole] == 21)
    assert np.all(out[between] == 0)
    assert np.all(out[c[:, 2] < -1.0] == 0)


def _grid(nx: int, ny: int, step: float) -> trimesh.Trimesh:
    xs, ys = np.meshgrid(np.arange(nx) * step, np.arange(ny) * step, indexing="ij")
    v = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = (
        idx[:-1, :-1].ravel(),
        idx[1:, :-1].ravel(),
        idx[1:, 1:].ravel(),
        idx[:-1, 1:].ravel(),
    )
    return trimesh.Trimesh(v, np.r_[np.stack([a, b, c], 1), np.stack([a, c, d], 1)], process=False)


def test_snap_interproximal_cuts_a_strip_off_at_its_root():
    from aligners.segment import snap_interproximal

    mesh = _grid(81, 41, 0.25)  # 20 x 10 mm, two crowns split at x = 10
    c = mesh.triangles_center
    labels = np.where(c[:, 0] < 10.0, 11, 21)
    strip = (c[:, 0] >= 10.0) & (c[:, 0] < 14.0) & (np.abs(c[:, 1] - 5.0) < 0.3)
    labels[strip] = 11  # crown 11 runs 4 mm along crown 21 as a thin strip
    out = snap_interproximal(mesh, labels)
    assert (out[strip] == 21).mean() > 0.95
    assert np.all(out[c[:, 0] < 9.0] == 11) and np.all(out[c[:, 0] > 11.0] == 21)


def test_cleanup_fills_a_thin_island_from_all_sides():
    mesh = _grid(81, 41, 0.25)
    c = mesh.triangles_center
    labels = np.where(c[:, 0] < 10.0, 11, 21)
    island = (c[:, 0] >= 10.5) & (c[:, 0] < 14.0) & (np.abs(c[:, 1] - 5.0) < 0.3)
    labels[island] = 11  # a separate small region of 11 inside 21
    labels[(c[:, 0] >= 10.0) & (c[:, 0] < 10.5) & (np.abs(c[:, 1] - 5.0) < 0.3)] = 21
    out, _ = cleanup(mesh, labels)
    assert np.all(out[island] == 21)
