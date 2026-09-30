"""Tooth movements in anatomical terms, pivoting about the center of resistance.

Sign conventions (tooth frame): mesial + toward the midline, buccal + labial/buccal,
extrusion + occlusal, rotation + mesial edge moves buccally, tip + crown tips mesially,
torque + crown moves buccally (root lingually).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from aligners.anatomy import Tooth
from aligners.geometry import Array, transform_points


def _unit(v: Array) -> Array:
    return v / np.linalg.norm(v)


def rotation_axes(tooth: Tooth) -> tuple[Array, Array, Array]:
    """Axes (rotation, tip, torque) whose positive right-hand rotation matches the conventions."""
    return (
        _unit(np.cross(tooth.m, tooth.b)),
        _unit(np.cross(tooth.o, tooth.m)),
        _unit(np.cross(tooth.o, tooth.b)),
    )


@dataclass
class Movement:
    mesial: float = 0.0
    buccal: float = 0.0
    extrusion: float = 0.0
    rotation: float = 0.0  # degrees
    tip: float = 0.0
    torque: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "mesial_mm": self.mesial,
            "buccal_mm": self.buccal,
            "extrusion_mm": self.extrusion,
            "rotation_deg": self.rotation,
            "tip_deg": self.tip,
            "torque_deg": self.torque,
        }


def compose(tooth: Tooth, mv: Movement) -> Array:
    """Pose (4x4) realizing a movement about the tooth's center of resistance."""
    ax_rot, ax_tip, ax_torque = rotation_axes(tooth)
    rotvec = (
        np.radians(mv.rotation) * ax_rot
        + np.radians(mv.tip) * ax_tip
        + np.radians(mv.torque) * ax_torque
    )
    R = Rotation.from_rotvec(rotvec).as_matrix()
    t = mv.mesial * tooth.m + mv.buccal * tooth.b + mv.extrusion * tooth.o
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tooth.cr + t - R @ tooth.cr
    return T


def decompose(tooth: Tooth, T: Array) -> Movement:
    """Anatomical components of a pose, measured in the tooth's initial frame about its CR."""
    t = transform_points(T, tooth.cr[None])[0] - tooth.cr
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    ax_rot, ax_tip, ax_torque = rotation_axes(tooth)
    return Movement(
        mesial=float(t @ tooth.m),
        buccal=float(t @ tooth.b),
        extrusion=float(t @ tooth.o),
        rotation=float(np.degrees(rotvec @ ax_rot)),
        tip=float(np.degrees(rotvec @ ax_tip)),
        torque=float(np.degrees(rotvec @ ax_torque)),
    )


def interpolate(tooth: Tooth, T: Array, s: float) -> Array:
    """Pose at fraction s of the rigid motion T: CR translates linearly, rotation scales."""
    if s <= 0:
        return np.eye(4)
    if s >= 1:
        return T.copy()
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    R = Rotation.from_rotvec(s * rotvec).as_matrix()
    t = transform_points(T, tooth.cr[None])[0] - tooth.cr
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = tooth.cr + s * t - R @ tooth.cr
    return out
