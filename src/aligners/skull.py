"""Head reference frame from the CT: the facial midsagittal plane by mirror symmetry of the
cranium, and a coarse bone mask for the surgical bone segments.

Head frame (right-handed, mm): +x patient right, +y anterior, +z superior. x is the normal of
the midsagittal plane; z is the CT's superior axis projected onto that plane and tilted about x
by `head_pitch_deg` (+ = anterior up); y = z x x. CT world coordinates are DICOM LPS.
"""

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from skimage.measure import marching_cubes

from aligners import fdi
from aligners.case import Case, CaseError, read_json, write_json
from aligners.ct import Volume
from aligners.geometry import Array, frame_matrix, mirror_plane, rotation_about, subsample

BONE_RES_MM = 1.0  # isotropic grid of the bone mask
JAW_EXCLUSION_MM = 5.0  # jaw/teeth labels dilated by this are not cranium
ABOVE_TEETH_MM = 15.0  # cranium starts this far above the highest upper tooth (no spine)
SKULL_POINTS = 8000
MIN_SKULL_POINTS = 2000
MAX_SYMMETRY_RMSE_MM = 1.5
CT_SUPERIOR = np.array([0.0, 0.0, 1.0])  # LPS +z
PATIENT_RIGHT = np.array([-1.0, 0.0, 0.0])  # LPS -x


def _resample(
    image: sitk.Image, reference: sitk.Image, nearest: bool, default: float, pixel: int
) -> sitk.Image:
    interpolator = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    return sitk.Resample(image, reference, sitk.Transform(), interpolator, default, pixel)


def _iso_reference(image: sitk.Image, res: float) -> sitk.Image:
    spacing = np.asarray(image.GetSpacing(), float)
    size = np.ceil(np.asarray(image.GetSize()) * spacing / res).astype(int)
    ref = sitk.Image([int(v) for v in size], sitk.sitkUInt8)
    ref.SetOrigin(image.GetOrigin())
    ref.SetDirection(image.GetDirection())
    ref.SetSpacing([res] * 3)
    return ref


def _volume(image: sitk.Image) -> Volume:
    return Volume(
        data=sitk.GetArrayFromImage(image).transpose(2, 1, 0),
        origin=np.asarray(image.GetOrigin(), float),
        spacing=np.asarray(image.GetSpacing(), float),
        direction=np.asarray(image.GetDirection(), float).reshape(3, 3),
    )


def bone_threshold(values: NDArray[Any]) -> float:
    """Bone/soft-tissue boundary: the upper of three Otsu classes (air, soft tissue, bone).
    Padding outside the scanner's field of view (the minimum value) is ignored, so the rule
    works for calibrated CT and uncalibrated CBCT alike."""
    v = values[values > values.min()]
    return float(threshold_multiotsu(v, classes=3)[1])


def build_head_frame(case: Case, seg: Volume, labels: NDArray[np.int32]) -> dict[str, Any]:
    """Bone mask on a 1 mm grid (`bone.npz`) and the midsagittal plane (`skull.json`)."""
    image = sitk.ReadImage(str(case.ct_dir / "ct.nii.gz"))
    ref = _iso_reference(image, BONE_RES_MM)
    lowest = float(sitk.GetArrayViewFromImage(image).min())
    ct = _volume(_resample(image, ref, False, lowest, sitk.sitkFloat32))
    seg_image = sitk.GetImageFromArray(labels.transpose(2, 1, 0).astype(np.uint8))
    seg_image.SetOrigin(tuple(seg.origin))
    seg_image.SetSpacing(tuple(seg.spacing))
    seg_image.SetDirection(tuple(seg.direction.ravel()))
    lab = _volume(_resample(seg_image, ref, True, 0.0, sitk.sitkUInt8)).data
    sample = ct.data[::3, ::3, ::3].ravel()
    bone = ct.data > bone_threshold(sample)
    np.savez_compressed(
        case.ct_dir / "bone.npz",
        bone=bone,
        labels=lab,
        origin=ct.origin,
        spacing=ct.spacing,
        direction=ct.direction,
    )

    jaws = np.isin(lab, [fdi.TS_LOWER_JAWBONE, fdi.TS_UPPER_JAWBONE, *fdi.TS_RESTORATIONS])
    jaws |= lab >= min(fdi.TS_TOOTH_LABELS)
    jaw_distance = np.asarray(ndimage.distance_transform_edt(~jaws, sampling=ct.spacing))
    near_jaws = jaw_distance < JAW_EXCLUSION_MM
    upper_teeth = np.isin(
        lab, [k for k, n in {**fdi.TS_TOOTH_LABELS, **fdi.TS_PULP_LABELS}.items() if n < 30]
    )
    if not upper_teeth.any():
        raise CaseError("CT segmentation has no upper teeth; cannot place the cranium")
    z_world = ct.to_world(np.argwhere(upper_teeth).astype(float)) @ CT_SUPERIOR
    cranium = bone & ~near_jaws
    ijk = np.argwhere(cranium)
    above = ct.to_world(ijk.astype(float)) @ CT_SUPERIOR > z_world.max() + ABOVE_TEETH_MM
    cranium[:] = False
    cranium[tuple(ijk[above].T)] = True
    if cranium.sum() < MIN_SKULL_POINTS:
        raise CaseError(
            "the CT shows too little cranium above the teeth to find the facial midline"
        )
    verts, _, normals, _ = marching_cubes(np.pad(cranium, 1).astype(np.float32), 0.5, step_size=2)
    verts = ct.to_world(verts - 1.0)
    normals = normals @ ct.direction.T
    keep = subsample(verts, SKULL_POINTS, seed=7)
    normal, point, rmse = mirror_plane(verts[keep], normals[keep], CT_SUPERIOR, contain_axis=False)
    if normal @ PATIENT_RIGHT < 0:
        normal = -normal
    warnings = []
    if rmse > MAX_SYMMETRY_RMSE_MM:
        warnings.append(
            f"facial midline fit is poor (symmetry RMSE {rmse:.1f} mm): check face metrics"
        )
    info = {
        "plane_normal": normal,
        "plane_point": point,
        "symmetry_rmse_mm": rmse,
        "ct_superior": CT_SUPERIOR,
        "coordinates": "CT world, DICOM LPS mm; the normal points to the patient's right",
    }
    write_json(case.ct_dir / "skull.json", info)
    return {**info, "warnings": warnings}


def load_skull(case: Case) -> dict[str, Array]:
    path = case.ct_dir / "skull.json"
    if not path.exists():
        raise CaseError(
            "no head frame: surgical modes need the CT image (case.toml inputs.ct); "
            "run `aligners ct` again"
        )
    info = read_json(path)
    return {k: np.asarray(info[k], float) for k in ("plane_normal", "plane_point", "ct_superior")}


def ct_to_head(skull: dict[str, Array], head_pitch_deg: float, origin_ct: Array) -> Array:
    """CT world (LPS) -> head frame, with its origin at `origin_ct` projected onto the plane."""
    x = skull["plane_normal"] / np.linalg.norm(skull["plane_normal"])
    z = skull["ct_superior"] - (skull["ct_superior"] @ x) * x
    z /= np.linalg.norm(z)
    # + pitch tilts the horizontal anterior-up: the head frame turns about +x (patient right)
    z = rotation_about(x, np.radians(head_pitch_deg))[:3, :3] @ z
    y = np.cross(z, x)
    origin = origin_ct - ((origin_ct - skull["plane_point"]) @ x) * x
    return np.linalg.inv(frame_matrix(origin, x, y, z))


def load_bone(case: Case) -> tuple[Volume, NDArray[np.uint8]]:
    path = case.ct_dir / "bone.npz"
    if not Path(path).exists():
        raise CaseError("no CT bone mask; run `aligners ct` again with the CT image")
    d = np.load(path)
    vol = Volume(d["bone"], d["origin"], d["spacing"], d["direction"])
    return vol, d["labels"]
