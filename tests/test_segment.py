import itertools

import numpy as np
import trimesh

from aligners.segment import alpha_expansion, cleanup


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
