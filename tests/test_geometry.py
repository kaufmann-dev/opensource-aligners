import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from aligners.geometry import ICPTarget, transform_points


def test_icp_recovers_small_rigid_transform():
    ring = trimesh.creation.annulus(r_min=15, r_max=25, height=8, sections=64).subdivide()
    half = ring.triangles_center[:, 1] < 5.0  # U-shaped like an arch: no rotational symmetry
    mesh = trimesh.Trimesh(ring.vertices, ring.faces[half])
    mesh.remove_unreferenced_vertices()
    pts = np.asarray(mesh.vertices)
    target = ICPTarget(pts, np.asarray(mesh.vertex_normals))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.02, -0.03, 0.05]).as_matrix()
    T[:3, 3] = [0.8, -0.5, 0.4]
    src = transform_points(np.linalg.inv(T), pts)
    est, rmse = target.icp(src, iterations=60, trim=1.0)
    assert rmse < 0.05
    assert np.abs(transform_points(est, src) - pts).max() < 0.05
