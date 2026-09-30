"""Intraoral scan loading and arch-frame detection.

Arch frame: right-handed, origin at the dental midline (incisal level), +z occlusal (toward the
opposing arch), +y anterior, +x = y x z. So +x is patient-right in the lower jaw and
patient-left in the upper jaw.
"""

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from aligners.case import CaseError
from aligners.fdi import Jaw
from aligners.geometry import (
    Array,
    ICPTarget,
    boundary_loops,
    clean_mesh,
    frame_matrix,
    orthonormal_frame,
    pca,
    subsample,
    transform_points,
)

CUSP_BAND_QUANTILE = 0.75
MIDLINE_BAND_MM = 3.0


def load_scan(path: Path) -> trimesh.Trimesh:
    try:
        raw = trimesh.load(path, force="mesh")
    except Exception as e:
        raise CaseError(f"cannot read scan {path}: {e}") from e
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise CaseError(f"{path} contains no triangles")
    mesh = clean_mesh(raw)
    extent = np.ptp(mesh.vertices, axis=0).max()
    if not 20.0 < extent < 200.0:
        raise CaseError(f"{path}: extent {extent:.1f} is not a dental arch in mm")
    return mesh


def patient_right_sign(jaw: Jaw) -> float:
    """Sign of the arch-frame x axis that points to the patient's right."""
    return 1.0 if jaw == "lower" else -1.0


def _mirror_plane(points: Array, normals: Array, occlusal: Array) -> tuple[Array, Array, float]:
    """Best sagittal symmetry plane (normal, point) of a cusp band, by mirror-ICP search."""
    target = ICPTarget(points, normals)
    c = points.mean(axis=0)
    u, v, _ = orthonormal_frame(occlusal, np.cross(occlusal, [1.0, 0.0, 0.0]) + 1e-9)
    best: tuple[float, Array] | None = None
    for theta in np.radians(np.arange(0, 180, 5)):
        s = np.cos(theta) * u + np.sin(theta) * v
        R = np.eye(4)
        R[:3, :3] -= 2 * np.outer(s, s)
        R[:3, 3] = 2 * np.dot(c, s) * s
        T, rmse = target.icp(transform_points(R, points), iterations=10, trim=0.7)
        if best is None or rmse < best[0]:
            best = (rmse, T @ R)
    assert best is not None
    M = best[1]
    w, vecs = np.linalg.eig(M[:3, :3])
    normal = np.real(vecs[:, np.argmin(np.abs(w + 1))])
    normal -= np.dot(normal, occlusal) * occlusal
    normal /= np.linalg.norm(normal)
    point = (c + transform_points(M, c[None])[0]) / 2
    return normal, point, best[0]


def arch_frame(mesh: trimesh.Trimesh) -> tuple[Array, dict[str, Any]]:
    """World->arch transform of a scan and diagnostics."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    N = np.asarray(mesh.vertex_normals, dtype=np.float64)
    loops = boundary_loops(np.asarray(mesh.faces, dtype=np.int64))
    if not loops:
        raise CaseError("scan has no open boundary; expected an intraoral surface scan")
    c, _, axes = pca(V)
    occlusal = axes[:, 2]
    base = V[loops[0]].mean(axis=0)
    if np.dot(c - base, occlusal) < 0:
        occlusal = -occlusal

    heights = (V - c) @ occlusal
    band = np.flatnonzero(heights >= np.quantile(heights, CUSP_BAND_QUANTILE))
    band = band[subsample(V[band], 4000)]
    normal, point, rmse = _mirror_plane(V[band], N[band], occlusal)

    anterior = np.cross(occlusal, normal)
    P = V[band]
    near = P[np.abs((P - point) @ normal) < MIDLINE_BAND_MM]
    if len(near) < 10:
        raise CaseError("could not find incisors at the arch midline")
    if np.dot(near.mean(axis=0) - P.mean(axis=0), anterior) < 0:
        anterior = -anterior
    tip = near[np.argmax((near - point) @ anterior)]
    origin = tip - np.dot(tip - point, normal) * normal
    x, y, z = orthonormal_frame(occlusal, anterior)
    to_arch = np.linalg.inv(frame_matrix(origin, x, y, z))
    return to_arch, {"symmetry_rmse_mm": rmse, "boundary_loops": len(loops)}


def scanner_bite(
    upper: Any, lower: Any, upper_to_arch: Array, lower_to_arch: Array
) -> tuple[bool, dict[str, float]]:
    """Do the two scans share a bite-registered frame (teeth in occlusion)?

    `upper`/`lower` are JawAnatomy objects (arch frames). In the scans' common frame the arches
    must face each other, touch, and not interpenetrate beyond scanner registration noise."""
    from aligners.geometry import MeshSDF, transform_points

    up_occlusal = np.linalg.inv(upper_to_arch)[:3, 2]
    low_occlusal = np.linalg.inv(lower_to_arch)[:3, 2]
    facing = float(np.dot(up_occlusal, low_occlusal))
    up_to_low = lower_to_arch @ np.linalg.inv(upper_to_arch)
    depths = []
    samples = np.vstack([lower.enamel_samples(n, 600, seed=5) for n in lower.teeth])
    d_min = np.full(len(samples), np.inf)
    for solid in upper.contact_solids.values():
        sdf = MeshSDF(
            transform_points(up_to_low, np.asarray(solid.vertices)), np.asarray(solid.faces)
        )
        near = np.all((samples > sdf.lo - 2.0) & (samples < sdf.hi + 2.0), axis=1)
        if near.any():
            d_min[near] = np.minimum(d_min[near], sdf(samples[near], bound=3.0))
    touching = d_min[np.isfinite(d_min)]
    depths = -touching[touching < 0]
    metrics = {
        "facing": facing,
        "contact_points": int((np.abs(touching) < 0.3).sum()),
        "max_penetration_mm": float(depths.max()) if len(depths) else 0.0,
        "penetration_p99_mm": float(np.quantile(-touching, 0.99)) if len(touching) else 0.0,
    }
    ok = facing < -0.8 and metrics["contact_points"] >= 20 and metrics["max_penetration_mm"] < 0.8
    return ok, metrics
