import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from aligners.anatomy import ArchCurve, GumField, JawAnatomy, Tooth, close_crown, submesh
from aligners.case import CaseError
from aligners.geometry import rigid_fit, transform_points
from aligners.surgery import (
    DentalFrames,
    Occlusion,
    movement_angles,
    movements_for_relation,
    table_to_transform,
    table_vector,
    to_frame,
    transform_to_table,
)

PRE = {  # scanned upper landmarks, movement frame (x right, y anterior, z up)
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


def _movement(t: list[float], pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Rz(yaw) Ry(roll) Rx(pitch) about the incisor midline (the frame origin), then t."""
    M = np.eye(4)
    M[:3, :3] = Rotation.from_euler("ZYX", [yaw, roll, pitch], degrees=True).as_matrix()
    M[:3, 3] = t
    return M


def test_table_signs_follow_the_movement_frame():
    assert np.allclose(table_vector({"forward_mm": 1, "up_mm": 0, "left_mm": 0}), [0, 1, 0])
    assert np.allclose(table_vector({"forward_mm": 0, "up_mm": 1, "left_mm": 0}), [0, 0, 1])
    assert np.allclose(table_vector({"forward_mm": 0, "up_mm": 0, "left_mm": 1}), [-1, 0, 0])


def test_movement_table_round_trip():
    M = _movement([1.0, 4.0, -2.0], 1.7, -1.1, 2.3)
    table = transform_to_table(M, PRE)
    back, residuals = table_to_transform(table, PRE, "maxilla")
    assert max(residuals.values()) < 0.1
    moved = np.array(list(to_frame(back, PRE).values()))
    assert np.abs(moved - np.array(list(to_frame(M, PRE).values()))).max() < 0.1
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


def test_movement_angle_signs():
    front, right = np.array([[0.0, 10.0, 0.0]]), np.array([[10.0, 0.0, 0.0]])
    up = _movement([0, 0, 0], 3.0, 0.0, 0.0)
    assert transform_points(up, front)[0, 2] > 0 and movement_angles(up)["pitch_deg"] > 2.9
    roll = _movement([0, 0, 0], 0.0, 3.0, 0.0)  # + = patient-left side up = right side down
    assert transform_points(roll, right)[0, 2] < 0 and movement_angles(roll)["roll_deg"] > 2.9
    yaw = _movement([0, 0, 0], 0.0, 0.0, 3.0)  # + = front toward the patient's left (-x)
    assert transform_points(yaw, front)[0, 0] < 0 and movement_angles(yaw)["yaw_deg"] > 2.9


def _frames() -> DentalFrames:
    P_u = np.eye(4)
    P_u[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    P_l = np.eye(4)
    P_l[:3, 3] = [0.0, -3.0, -12.0]
    to_bite = {"upper": np.linalg.inv(P_u), "lower": np.linalg.inv(P_u) @ P_l}
    return DentalFrames({"upper": P_u, "lower": P_l}, {}, to_bite, 0.0)


def test_movements_realize_the_relation_moving_one_jaw():
    from aligners.surgery import arch_to_bite_after

    frames = _frames()
    X = np.eye(4)
    X[:3, :3] = Rotation.from_euler("yz", [180, 4], degrees=True).as_matrix()
    X[:3, 3] = [1.0, -2.0, 11.0]
    for move in ("mandible", "maxilla"):
        M = movements_for_relation(X, frames, move)
        fixed = "maxilla" if move == "mandible" else "mandible"
        assert np.allclose(M[fixed], np.eye(4))
        after = arch_to_bite_after(frames, M)
        assert np.allclose(np.linalg.inv(after["lower"]) @ after["upper"], X, atol=1e-9)


def test_dental_frame_pitch_tilts_the_horizontal_anterior_up(tmp_path):
    import json

    from aligners.surgery import dental_frames

    (tmp_path / "work").mkdir()
    identity = np.eye(4).tolist()
    jaws = {
        jaw: {"scan_to_arch": identity, "ct_to_scan": identity, "arch_to_bite": identity}
        for jaw in ("upper", "lower")
    }
    (tmp_path / "work" / "frames.json").write_text(
        json.dumps({"bite_source": "scans", "jaws": jaws})
    )
    from aligners.case import Case

    frames = dental_frames(Case(tmp_path), {"surgery": {"pitch_deg": 10.0}})
    anterior = frames.P["upper"][:3, :3] @ np.array([0.0, 1.0, 0.0])  # upper arch +y
    occlusal = frames.P["upper"][:3, :3] @ np.array([0.0, 0.0, 1.0])  # toward the lower arch
    assert anterior[1] > 0.98 and anterior[2] < -0.17  # the occlusal plane's front lies lower
    assert occlusal[2] < -0.98  # z superior
    left = frames.P["upper"][:3, :3] @ np.array([1.0, 0.0, 0.0])  # upper arch +x = patient left
    assert left[0] < -0.99


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


def test_landmarks_contract_reproduces_the_movement_table():
    """Another planner following landmarks.json's rule writes aligners' own table, also when
    the CT mandible is in another bite than the occlusion of the scans."""
    from aligners.surgery import landmarks_document

    rng = np.random.default_rng(5)

    def rigid(angle: float, shift: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * angle).as_matrix()
        T[:3, 3] = rng.normal(size=3) * shift
        return T

    # CT -> frame of the upper teeth; CT-image lower teeth -> frame in the scanned occlusion
    C = {"upper": rigid(0.3, 50.0), "lower": rigid(0.3, 50.0)}
    M = {"maxilla": rigid(0.05, 3.0), "mandible": rigid(0.05, 3.0)}
    scanned = {"upper": PRE, "lower": {k: v - [0.0, 2.0, 10.0] for k, v in PRE.items()}}
    report = {"ct_to_frame": C, "pitch_deg": 0.0, "landmarks_scanned_frame": scanned}
    doc = landmarks_document(report)
    frame = doc["head_frame"]
    frame_to_ct = np.linalg.inv(C["upper"])
    # what the planner holds: CT-image -> post-op LPS transforms of its jaw segments
    planner = {
        "maxilla": frame_to_ct @ M["maxilla"] @ C["upper"],
        "mandible": frame_to_ct @ M["mandible"] @ C["lower"],
    }
    for jaw, bone in (("upper", "maxilla"), ("lower", "mandible")):
        expected = transform_to_table(M[bone], scanned[jaw])
        for name, entry in doc["jaws"][bone].items():
            d = transform_points(planner[bone], entry["ct_lps"][None])[0] - entry["reference_lps"]
            got = {
                "forward_mm": round(float(d @ frame["y_anterior"]), 1) + 0.0,
                "up_mm": round(float(d @ frame["z_superior"]), 1) + 0.0,
                "left_mm": round(float(-(d @ frame["x_patient_right"])), 1) + 0.0,
            }
            assert got == expected[name]
