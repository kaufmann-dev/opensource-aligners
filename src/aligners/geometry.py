"""Shared geometry: mesh cleanup, rigid transforms, ICP, signed distance."""

import igl
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

Array = NDArray[np.float64]


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Weld vertices, drop degenerate/duplicate faces, keep the largest connected component."""
    m = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    m.update_faces(m.nondegenerate_faces(height=1e-9))
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: len(p.faces))
    return m


def boundary_loops(faces: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    """Vertex-index loops of every mesh boundary, longest first."""
    loops = [np.asarray(loop, dtype=np.int64) for loop in igl.boundary_loop_all(faces)]
    return sorted(loops, key=len, reverse=True)


def transform_points(T: Array, pts: Array) -> Array:
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_vectors(T: Array, vecs: Array) -> Array:
    return vecs @ T[:3, :3].T


def transformed(mesh: trimesh.Trimesh, T: Array) -> trimesh.Trimesh:
    return trimesh.Trimesh(transform_points(T, mesh.vertices), mesh.faces.copy(), process=False)


def frame_matrix(origin: Array, x: Array, y: Array, z: Array) -> Array:
    """Local->world matrix of an orthonormal frame given by its axes and origin."""
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
    return T


def orthonormal_frame(z: Array, y_hint: Array) -> tuple[Array, Array, Array]:
    """Right-handed (x, y, z) with z exact and y as close as possible to y_hint."""
    z = z / np.linalg.norm(z)
    y = y_hint - np.dot(y_hint, z) * z
    y /= np.linalg.norm(y)
    x = np.asarray(np.cross(y, z), dtype=np.float64)
    return x, y, z


def rotation_about(axis: Array, angle_rad: float, center: Array | None = None) -> Array:
    T = np.eye(4)
    axis = axis / np.linalg.norm(axis)
    T[:3, :3] = Rotation.from_rotvec(axis * angle_rad).as_matrix()
    if center is not None:
        T[:3, 3] = center - T[:3, :3] @ center
    return T


def translation(t: Array) -> Array:
    T = np.eye(4)
    T[:3, 3] = t
    return T


def rotation_angle_deg(T: Array) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3]).as_rotvec())))


def pca(points: Array) -> tuple[Array, Array, Array]:
    """Centroid, eigenvalues (descending) and eigenvectors (columns) of a point set."""
    c = points.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((points - c).T))
    order = np.argsort(w)[::-1]
    return c, np.asarray(w[order], dtype=np.float64), np.asarray(v[:, order], dtype=np.float64)


def vertex_normals(mesh: trimesh.Trimesh) -> Array:
    return np.asarray(mesh.vertex_normals, dtype=np.float64)


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int = 0) -> tuple[Array, Array]:
    """Area-uniform surface samples and their face normals."""
    result = trimesh.sample.sample_surface(mesh, count, seed=seed)
    pts, face_idx = result[0], result[1]
    return np.asarray(pts, dtype=np.float64), np.asarray(mesh.face_normals[face_idx])


def subsample(points: Array, count: int, seed: int = 0) -> NDArray[np.int64]:
    if len(points) <= count:
        return np.arange(len(points))
    return np.random.default_rng(seed).choice(len(points), count, replace=False)


def _point_to_plane_step(src: Array, dst: Array, normals: Array) -> Array:
    A = np.hstack([np.cross(src, normals), normals])
    b = np.einsum("ij,ij->i", dst - src, normals)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:]
    return T


class ICPTarget:
    """Target point set with normals, indexed once for repeated ICP runs."""

    def __init__(self, points: Array, normals: Array):
        self.points = points
        self.normals = normals
        self.tree = KDTree(points)

    def icp(
        self,
        src: Array,
        init: Array | None = None,
        iterations: int = 30,
        trim: float = 0.8,
        max_dist: float = np.inf,
    ) -> tuple[Array, float]:
        """Trimmed point-to-plane ICP. Returns (src->target transform, trimmed RMSE)."""
        T = np.eye(4) if init is None else init.copy()
        rmse = np.inf
        for _ in range(iterations):
            moved = transform_points(T, src)
            dist, idx = self.tree.query(moved, distance_upper_bound=max_dist)
            ok = np.isfinite(dist)
            if ok.sum() < 6:
                return T, np.inf
            d, i, p = dist[ok], idx[ok], moved[ok]
            keep = d <= np.quantile(d, trim)
            rmse = float(np.sqrt(np.mean(d[keep] ** 2)))
            step = _point_to_plane_step(p[keep], self.points[i[keep]], self.normals[i[keep]])
            T = step @ T
            if np.linalg.norm(step[:3, 3]) < 1e-5 and rotation_angle_deg(step) < 1e-4:
                break
        return T, rmse


class MeshSDF:
    """Signed distance to a (nearly) closed triangle mesh; negative inside.

    The sign comes from the fast generalized winding number, so small gaps are tolerated.
    """

    def __init__(self, vertices: Array, faces: NDArray[np.int64]):
        self.V = np.ascontiguousarray(vertices, dtype=np.float64)
        self.F = np.ascontiguousarray(faces, dtype=np.int64)
        self.lo = self.V.min(axis=0)
        self.hi = self.V.max(axis=0)

    @classmethod
    def of(cls, mesh: trimesh.Trimesh) -> "MeshSDF":
        return cls(np.asarray(mesh.vertices), np.asarray(mesh.faces))

    def __call__(self, points: Array, bound: float = np.inf) -> Array:
        """Signed distance; with a finite bound, |d| > bound is clamped to +-bound."""
        if len(points) == 0:
            return np.zeros(0)
        P = np.ascontiguousarray(points, dtype=np.float64)
        S, *_ = igl.signed_distance(
            P,
            self.V,
            self.F,
            sign_type=igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
            lower_bound=-bound,
            upper_bound=bound,
        )
        S = np.asarray(S, dtype=np.float64)
        far = np.isnan(S)
        if far.any():
            # igl leaves points beyond the bound undefined; their sign still matters.
            wn = np.asarray(igl.fast_winding_number(self.V, self.F, P[far]))
            S[far] = np.where(wn > 0.5, -bound, bound)
        return np.clip(S, -bound, bound)

    def unsigned(self, points: Array) -> Array:
        d2, *_ = igl.point_mesh_squared_distance(
            np.ascontiguousarray(points, dtype=np.float64), self.V, self.F
        )
        return np.sqrt(d2)


def face_adjacency_graph(mesh: trimesh.Trimesh) -> tuple[NDArray[np.int64], Array]:
    """Face adjacency pairs and the signed dihedral angle across each shared edge.

    Positive angles are convex edges, negative are concave (creases like gingival margins).
    """
    pairs = np.asarray(mesh.face_adjacency, dtype=np.int64)
    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    convex = np.asarray(mesh.face_adjacency_convex, dtype=bool)
    return pairs, np.where(convex, angles, -angles)


def decimate(mesh: trimesh.Trimesh, target_faces: int, max_error: float = 0.025) -> trimesh.Trimesh:
    """Topology-preserving quadric decimation (MeshLib): closed stays closed, genus is kept.

    Stops at target_faces or when the next collapse would exceed max_error (mm)."""
    import meshlib.mrmeshnumpy as mn
    import meshlib.mrmeshpy as mr

    if len(mesh.faces) <= target_faces:
        return mesh
    m = mn.meshFromFacesVerts(
        np.asarray(mesh.faces, np.int32), np.asarray(mesh.vertices, np.float32)
    )
    settings = mr.DecimateSettings()
    settings.maxError = max_error
    settings.maxDeletedFaces = len(mesh.faces) - target_faces
    settings.packMesh = True
    mr.decimateMesh(m, settings)
    return trimesh.Trimesh(
        mn.getNumpyVerts(m).astype(np.float64), mn.getNumpyFaces(m.topology), process=True
    )
