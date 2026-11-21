import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from aligners.anatomy import ArchCurve, GumField, JawAnatomy, Tooth, close_crown, submesh
from aligners.case import CaseError
from aligners.geometry import mirror_plane, rigid_fit, rotation_about, transform_points
from aligners.surgery import (
    HeadFrames,
    Occlusion,
    distribute,
    jaw_angles,
    lefort_matrix,
    table_to_transform,
    table_vector,
    to_head,
    transform_to_table,
)

PRE = {  # scanned upper landmarks, head frame (x right, y anterior, z up)
    "incisor_midline": np.array([0.0, 0.0, 0.0]),
    "canine_right": np.array([16.0, -8.0, 1.0]),
    "canine_left": np.array([-16.0, -8.0, 1.0]),
    "molar_right": np.array([24.0, -30.0, 2.0]),
    "molar_left": np.array([-24.0, -30.0, 2.0]),
}


def test_rigid_fit_recovers_a_transform_and_a_translation():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.05, -0.02, 0.1]).as_matrix()
    T[:3, 3] = [1.0, -2.0, 0.5]
    src = np.array(list(PRE.values()))
    est, res = rigid_fit(src, transform_points(T, src))
    assert np.allclose(est, T, atol=1e-9) and res.max() < 1e-9
    one, _ = rigid_fit(src[:1], src[:1] + [3.0, 0.0, 0.0])
    assert np.allclose(one[:3, :3], np.eye(3)) and np.allclose(one[:3, 3], [3.0, 0, 0])


def test_table_signs_follow_the_head_frame():
    assert np.allclose(table_vector({"forward_mm": 1, "up_mm": 0, "left_mm": 0}), [0, 1, 0])
    assert np.allclose(table_vector({"forward_mm": 0, "up_mm": 1, "left_mm": 0}), [0, 0, 1])
    assert np.allclose(table_vector({"forward_mm": 0, "up_mm": 0, "left_mm": 1}), [-1, 0, 0])


def test_movement_table_round_trip():
    M = lefort_matrix(PRE["incisor_midline"], np.array([1.0, 4.0, -2.0]), 0.03, -0.02, 0.04)
    table = transform_to_table(M, PRE)
    back, residuals = table_to_transform(table, PRE, "maxilla")
    assert max(residuals.values()) < 0.1
    moved = np.array(list(to_head(back, PRE).values()))
    assert np.abs(moved - np.array(list(to_head(M, PRE).values()))).max() < 0.1
    shift = {"incisor_midline": {"forward_mm": 4.0, "up_mm": 0.0, "left_mm": 0.0}}
    T, _ = table_to_transform(shift, PRE, "maxilla")
    assert np.allclose(T[:3, 3], [0.0, 4.0, 0.0]) and np.allclose(T[:3, :3], np.eye(3))


def test_movement_table_errors():
    move = {"forward_mm": 4.0, "up_mm": 0.0, "left_mm": 0.0}
    with pytest.raises(CaseError, match="not two"):
        table_to_transform({"molar_right": move, "molar_left": move}, PRE, "maxilla")
    wrong = {n: dict(move) for n in PRE}
    wrong["molar_left"]["left_mm"] = 3.0  # a sign error: the others say no transverse movement
    with pytest.raises(CaseError, match="not one rigid movement"):
        table_to_transform(wrong, PRE, "maxilla")
    with pytest.raises(CaseError, match="missing"):
        table_to_transform({"incisor_midline": move}, {}, "maxilla")


def test_angle_signs():
    lm = {k: v.copy() for k, v in PRE.items()}
    assert abs(jaw_angles(lm)["cant_deg"]) < 1e-9
    lm["molar_left"][2] += 2.0
    assert jaw_angles(lm)["cant_deg"] > 0  # + = patient-left side up
    lm = {k: v.copy() for k, v in PRE.items()}
    lm["incisor_midline"] += [-1.0, 0.0, 3.0]  # incisors toward patient left and up
    a = jaw_angles(lm)
    assert a["yaw_deg"] > 0 and a["midline_mm"] > 0 and a["occlusal_pitch_deg"] > 0


def test_lefort_matrix_rotation_signs():
    p = np.zeros(3)
    front, right = np.array([[0.0, 10.0, 0.0]]), np.array([[10.0, 0.0, 0.0]])
    assert transform_points(lefort_matrix(p, p, 0.1, 0.0, 0.0), front)[0, 2] > 0  # pitch: up
    assert transform_points(lefort_matrix(p, p, 0.0, 0.1, 0.0), right)[0, 2] < 0  # roll: R down
    assert transform_points(lefort_matrix(p, p, 0.0, 0.0, 0.1), front)[0, 0] < 0  # yaw: left


def test_mirror_plane_finds_a_tilted_symmetry_plane():
    rng = np.random.default_rng(3)
    half = rng.uniform([5, -40, -20], [60, 40, 30], size=(2000, 3))
    half[:, 2] += 0.02 * half[:, 1] ** 2  # no other symmetry
    pts = np.vstack([half, half * [-1, 1, 1]])
    R = rotation_about(np.array([0.0, 0.0, 1.0]), np.radians(4.0)) @ rotation_about(
        np.array([0.0, 1.0, 0.0]), np.radians(3.0)
    )
    pts = transform_points(R, pts)
    normals = np.tile(R[:3, 0], (len(pts), 1))
    normal, _, rmse = mirror_plane(pts, normals, np.array([0.0, 0.0, 1.0]), contain_axis=False)
    assert abs(abs(normal @ R[:3, 0]) - 1.0) < 1e-3 and rmse < 0.5


def _frames() -> HeadFrames:
    P_u = np.eye(4)
    P_u[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    P_u[:3, 3] = [0.5, 0.0, 0.0]
    P_l = np.eye(4)
    P_l[:3, 3] = [0.0, -3.0, -12.0]
    return HeadFrames(np.eye(4), {"upper": P_u, "lower": P_l}, {}, {}, 0.0)


def test_distribute_realizes_the_occlusion_and_levels_the_maxilla():
    hf = _frames()
    X = np.eye(4)
    X[:3, :3] = Rotation.from_euler("yz", [180, 4], degrees=True).as_matrix()
    X[:3, 3] = [1.0, -2.0, 11.0]
    upper_lm = to_head(np.linalg.inv(hf.P["upper"]), PRE)  # arch frame
    upper_lm["molar_left"] = upper_lm["molar_left"] + [0.0, 0.0, 1.5]  # a cant in the arch
    cfg = {
        "osteotomies": ["lefort1", "bsso"],
        "maxilla_forward_mm": 2.0,
        "maxilla_up_mm": 1.0,
        "maxilla_pitch_deg": 0.0,
    }
    M_max, M_mand, _ = distribute(X, hf, cfg, upper_lm, {})
    Q_u, Q_l = M_max @ hf.P["upper"], M_mand @ hf.P["lower"]
    assert np.allclose(np.linalg.inv(Q_l) @ Q_u, X, atol=1e-9)
    a = jaw_angles(to_head(Q_u, upper_lm))
    assert abs(a["cant_deg"]) < 0.01 and abs(a["yaw_deg"]) < 0.01 and abs(a["midline_mm"]) < 0.01
    moved = transform_points(M_max, PRE["incisor_midline"][None])[0] - PRE["incisor_midline"]
    assert abs(moved[1] - 2.0) < 1e-6 and abs(moved[2] - 1.0) < 1e-6

    M_max, M_mand, _ = distribute(X, hf, {**cfg, "osteotomies": ["bsso"]}, upper_lm, {})
    assert np.allclose(M_max, np.eye(4))
    Q_l = M_mand @ hf.P["lower"]
    assert np.allclose(np.linalg.inv(Q_l) @ hf.P["upper"], X, atol=1e-9)

    condyles = {
        "condylion_right": np.array([50.0, -80.0, 20.0]),
        "condylion_left": np.array([-50.0, -80.0, 20.0]),
    }
    M_max, M_mand, _ = distribute(X, hf, {**cfg, "osteotomies": ["lefort1"]}, upper_lm, condyles)
    for c in condyles.values():
        assert np.allclose(transform_points(M_mand, c[None])[0], c, atol=1e-9)
    Q_u, Q_l = M_max @ hf.P["upper"], M_mand @ hf.P["lower"]
    assert np.allclose(np.linalg.inv(Q_l) @ Q_u, X, atol=1e-9)
    moved = transform_points(M_max, PRE["incisor_midline"][None])[0] - PRE["incisor_midline"]
    assert abs(moved[2] - 1.0) < 1e-4


# -- ideal occlusion on box teeth -------------------------------------------------------------
# Lower frame: +x patient right, +y anterior, +z toward the upper arch. Lower crowns occupy
# z 0..6; upper crowns are designed in the lower frame in the ideal relation (Class I molars,
# canine on the lower canine's distal contact, 2.5 mm overjet, posterior teeth touching).
LOWER = {  # position: (x, y, width, depth)
    1: (2.6, 0.0, 5.0, 2.0),
    3: (8.0, -4.0, 6.0, 5.0),
    4: (12.0, -10.0, 7.0, 7.0),
    5: (14.0, -17.0, 7.0, 7.0),
    6: (17.0, -27.0, 10.0, 10.0),
}
X_TRUE = np.eye(4)
X_TRUE[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
X_TRUE[2, 3] = 12.0


def _box_tooth(number: int, c: np.ndarray, w: float, d: float, z0: float, z1: float, up: bool):
    """Box crown with its margin side open; `up` = crown points to +z (lower teeth)."""
    box = trimesh.creation.box(extents=(w, d, z1 - z0)).subdivide().subdivide()
    box.apply_translation([c[0], c[1], 0.5 * (z0 + z1)])
    zc = box.triangles_center[:, 2]
    patch = submesh(box, zc > z0 + 0.01 if up else zc < z1 - 0.01)
    o = np.array([0.0, 0.0, 1.0 if up else -1.0])
    tip, base = (z1, z0) if up else (z0, z1)
    side = np.sign(c[0])
    posterior = number % 10 >= 4
    m = np.array([0.0, 1.0, 0.0]) if posterior else np.array([-side, 0.0, 0.0])
    half = d / 2 if posterior else w / 2
    mid = np.array([c[0], c[1], 0.5 * (z0 + z1)])
    t = Tooth(
        fdi=number,
        center=mid,
        o=o,
        m=m,
        b=np.cross(o, m) * (1 if np.cross(o, m) @ [side, 1.0, 0.0] > 0 else -1),
        cr=np.array([c[0], c[1], base]) - 5.0 * o,
        margin_center=np.array([c[0], c[1], base]),
        mesial_contact=mid + half * m,
        distal_contact=mid - half * m,
        cusp=np.array([c[0], c[1], tip]),
        fa=mid,
        width=w,
        crown_height=z1 - z0,
        root_length=12.0,
        root_from_ct=False,
    )
    return t, patch


def _anatomy(jaw: str, teeth: dict) -> JawAnatomy:
    gum = GumField(np.full((10, 10), -50.0, dtype=np.float32), -100.0, -100.0, 20.0)
    return JawAnatomy(
        jaw,
        ArchCurve(np.array([[-20.0, -27.0], [0.0, 0.0], [20.0, -27.0]])),
        {n: t for n, (t, _) in teeth.items()},
        {n: close_crown(p, t.o) for n, (t, p) in teeth.items()},
        {},
        gum,
        {n: p for n, (_, p) in teeth.items()},
    )


def _posed_copy(t: Tooth, patch: trimesh.Trimesh, T: np.ndarray):
    from aligners.analyze import posed_tooth

    return posed_tooth(t, T), trimesh.Trimesh(transform_points(T, patch.vertices), patch.faces)


def _dentitions() -> tuple[JawAnatomy, JawAnatomy]:
    lower, upper = {}, {}
    to_upper = np.linalg.inv(X_TRUE)
    for right in (True, False):
        s = 1.0 if right else -1.0
        for pos, (x, y, w, d) in LOWER.items():
            lower[(40 if right else 30) + pos] = _box_tooth(
                (40 if right else 30) + pos, np.array([s * x, y]), w, d, 0.0, 6.0, True
            )
            low = lower[(40 if right else 30) + pos][0]
            if pos == 1:  # overjet: upper edge 2.5 mm labial of the lower labial face
                c, z0 = np.array([s * x, y + d / 2 + 2.5]), 4.0
            elif pos == 3:  # canine cusp over the lower canine's distal contact
                c, z0 = np.array([s * (x + 1.0), low.distal_contact[1]]), 6.0
            else:  # half a cusp distal (Class I), a little buccal
                c, z0 = np.array([s * (x + 1.5), y - 2.5]), 6.0
            n = (10 if right else 20) + pos
            t, patch = _box_tooth(n, c, w, d, z0, z0 + 6.0, False)
            upper[n] = _posed_copy(t, patch, to_upper)
    return _anatomy("upper", upper), _anatomy("lower", lower)


def test_occlusion_recovers_the_ideal_relation():
    upper, lower = _dentitions()
    occ = Occlusion(upper, {}, lower, {}, overjet_mm=2.5)
    r = occ.relation(X_TRUE)
    assert abs(r["overjet_mm"] - 2.5) < 1e-6 and abs(r["midline_mm"]) < 1e-6
    assert abs(r["molar_right_mm"]) < 1e-6 and abs(r["canine_left_mm"]) < 1e-6
    off = np.eye(4)
    off[:3, :3] = Rotation.from_euler("z", 3, degrees=True).as_matrix()
    off[:3, 3] = [1.5, -2.0, 1.0]
    X, metrics = occ.solve(off @ X_TRUE)
    assert np.abs(X[:3, 3] - X_TRUE[:3, 3]).max() < 0.3
    angle = np.degrees(Rotation.from_matrix(X[:3, :3] @ X_TRUE[:3, :3].T).magnitude())
    assert angle < 0.5
    assert metrics["max_penetration_mm"] < 0.2 and not metrics["warnings"]
