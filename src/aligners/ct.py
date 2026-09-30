"""CT import: DICOM series selection, TotalSegmentator `teeth`, per-tooth meshes."""

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pydicom
import SimpleITK as sitk
import trimesh
from numpy.typing import NDArray
from scipy import ndimage
from skimage.measure import marching_cubes

from aligners import fdi
from aligners.case import Case, CaseError, read_json, write_json, write_mesh
from aligners.geometry import decimate, pca

MESH_VOXEL_MM = 0.25
MASK_SIGMA_MM = 0.4
TOOTH_FACES = 20000
BONE_FACES = 60000
CROWN_BONE_DISTANCE_MM = 1.0


@dataclass
class SeriesInfo:
    uid: str
    number: str
    description: str
    modality: str
    files: list[str]
    rows: int
    cols: int
    pixel_spacing: tuple[float, float]
    slice_spacing: float
    axial: bool

    @property
    def voxel_volume(self) -> float:
        return self.pixel_spacing[0] * self.pixel_spacing[1] * self.slice_spacing

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d["files"] = len(self.files)
        return d


def discover_series(root: Path) -> list[SeriesInfo]:
    groups: dict[str, list[tuple[Path, Any]]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if uid:
            groups.setdefault(uid, []).append((path, ds))
    out = []
    for uid, items in groups.items():
        ds0 = items[0][1]
        iop = np.asarray(getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
        normal = np.cross(iop[:3], iop[3:])
        positions = [
            float(np.dot(normal, np.asarray(ds.ImagePositionPatient, float)))
            for _, ds in items
            if hasattr(ds, "ImagePositionPatient")
        ]
        order = np.argsort(positions) if len(positions) == len(items) else np.arange(len(items))
        steps = np.diff(np.sort(positions)) if len(positions) > 1 else np.array([0.0])
        steps = steps[steps > 1e-4]
        spacing = [float(v) for v in getattr(ds0, "PixelSpacing", [0.0, 0.0])]
        out.append(
            SeriesInfo(
                uid=uid,
                number=str(getattr(ds0, "SeriesNumber", "")),
                description=str(getattr(ds0, "SeriesDescription", "")),
                modality=str(getattr(ds0, "Modality", "")),
                files=[str(items[i][0]) for i in order],
                rows=int(getattr(ds0, "Rows", 0)),
                cols=int(getattr(ds0, "Columns", 0)),
                pixel_spacing=(spacing[0], spacing[1]),
                slice_spacing=float(np.median(steps)) if len(steps) else 0.0,
                axial=bool(abs(normal[2]) > 0.9),
            )
        )
    return out


def select_series(series: list[SeriesInfo], override: str) -> SeriesInfo:
    if override:
        for s in series:
            if override in (s.uid, s.number):
                return s
        raise CaseError(f"series '{override}' not found; candidates: {_series_list(series)}")
    usable = [
        s
        for s in series
        if s.modality == "CT" and len(s.files) >= 30 and s.axial and s.slice_spacing > 0
    ]
    if not usable:
        raise CaseError(f"no axial CT series with >= 30 slices; found: {_series_list(series)}")
    return min(usable, key=lambda s: (round(s.voxel_volume, 4), -len(s.files)))


def _series_list(series: list[SeriesInfo]) -> str:
    return "; ".join(
        f"#{s.number} {s.modality} '{s.description}' {len(s.files)} files" for s in series
    )


def _import_ct(case: Case, source: Path, series_override: str) -> dict[str, Any]:
    dst = case.ct_dir / "ct.nii.gz"
    if source.is_dir():
        series = discover_series(source)
        chosen = select_series(series, series_override)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(chosen.files)
        image = reader.Execute()
        info = {"source": str(source), "series": chosen.summary()}
    else:
        image = sitk.ReadImage(str(source))
        info = {"source": str(source), "series": None}
    sitk.WriteImage(image, str(dst))
    return info


# TotalSegmentator <= 2.18 passes device=None to the crop model of the `teeth` task when run on
# CPU (`convert_device_to_string` ignores plain strings). Patch that function in-process.
_TOTALSEG_SHIM = """
import sys
import totalsegmentator.python_api as api
_orig = api.convert_device_to_string
api.convert_device_to_string = lambda d: d if isinstance(d, str) else _orig(d)
from totalsegmentator.bin.TotalSegmentator import main
sys.argv = ["TotalSegmentator", *sys.argv[1:]]
main()
"""


def _totalseg_python() -> str:
    exe = shutil.which("TotalSegmentator")
    if exe is None:
        raise CaseError(
            "TotalSegmentator is not installed; install it with `pip install TotalSegmentator` "
            "or pass an existing `teeth` segmentation with --ct-seg"
        )
    first = Path(exe).read_bytes()[:512].split(b"\n", 1)[0].decode(errors="replace")
    if not first.startswith("#!"):
        raise CaseError(f"cannot find the Python interpreter of {exe}")
    return first[2:].strip()


def _run_totalsegmentator(ct_path: Path, seg_path: Path, device: str, log: Path) -> None:
    args = ["-i", str(ct_path), "-o", str(seg_path), "--ml", "-ta", "teeth", "--device", device]
    cmd = [_totalseg_python(), "-c", _TOTALSEG_SHIM, *args]
    with log.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not seg_path.exists():
        tail = log.read_text()[-2000:]
        raise CaseError(f"TotalSegmentator failed (exit {proc.returncode}); log {log}:\n{tail}")


@dataclass
class Volume:
    """Label or intensity volume in (i, j, k) index order with its index->world mapping."""

    data: NDArray[Any]
    origin: NDArray[np.float64]
    spacing: NDArray[np.float64]
    direction: NDArray[np.float64]

    @classmethod
    def read(cls, path: Path) -> "Volume":
        img = sitk.ReadImage(str(path))
        return cls(
            data=sitk.GetArrayFromImage(img).transpose(2, 1, 0),
            origin=np.asarray(img.GetOrigin(), float),
            spacing=np.asarray(img.GetSpacing(), float),
            direction=np.asarray(img.GetDirection(), float).reshape(3, 3),
        )

    def to_world(self, ijk: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.origin + (ijk * self.spacing) @ self.direction.T

    def to_index(self, xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        return ((xyz - self.origin) @ np.linalg.inv(self.direction).T) / self.spacing


def merge_labels(seg: NDArray[Any]) -> tuple[NDArray[np.uint8], list[str]]:
    """Map TotalSegmentator `teeth` labels to an FDI-valued volume (0 = background).

    Pulps join their tooth; each bridge/crown/implant component joins the tooth it touches most.
    """
    warnings: list[str] = []
    out = np.zeros(seg.shape, dtype=np.uint8)
    for label, number in fdi.TS_TOOTH_LABELS.items():
        out[seg == label] = number
    for label, number in fdi.TS_PULP_LABELS.items():
        out[seg == label] = number
    restorations = np.isin(seg, fdi.TS_RESTORATIONS)
    if restorations.any():
        components, count = cast(tuple[NDArray[np.int32], int], ndimage.label(restorations))
        for comp in range(1, count + 1):
            mask = components == comp
            ring = ndimage.binary_dilation(mask) & ~mask
            touching = out[ring]
            touching = touching[touching > 0]
            if len(touching) == 0:
                warnings.append(
                    f"restoration component of {int(mask.sum())} voxels touches no tooth"
                )
                continue
            values, counts = np.unique(touching, return_counts=True)
            out[mask] = values[np.argmax(counts)]
    return out, warnings


def _mask_mesh(vol: Volume, mask: NDArray[np.bool_], faces: int) -> trimesh.Trimesh:
    """Smooth surface of a binary mask, in world mm."""
    margin = np.ceil(3.0 / vol.spacing).astype(int)
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, mask.shape)
    crop = mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].astype(np.float32)
    crop = np.pad(crop, 1)
    smooth = ndimage.gaussian_filter(crop, MASK_SIGMA_MM / vol.spacing)
    step = MESH_VOXEL_MM / vol.spacing
    axes = [np.arange(0, n - 1 + 1e-6, s) for n, s in zip(smooth.shape, step, strict=True)]
    grid = np.meshgrid(*axes, indexing="ij")
    fine = ndimage.map_coordinates(smooth, grid, order=1)
    verts, tris, _, _ = marching_cubes(fine, 0.5)
    native = verts * step + lo - 1
    mesh = trimesh.Trimesh(vol.to_world(native), tris, process=True)
    mesh = decimate(mesh, faces)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def run_ct(case: Case, force: bool = False, device: str | None = None) -> dict[str, Any]:
    cfg = case.config()
    inputs = cfg["inputs"]
    if not (inputs["ct"] or inputs["ct_seg"]):
        raise CaseError("case has no CT input (--ct or --ct-seg)")
    case.ct_dir.mkdir(parents=True, exist_ok=True)
    info_path = case.ct_dir / "ct.json"
    seg_path = case.ct_dir / "teeth_seg.nii.gz"
    source_key = f"{inputs['ct']}|{inputs['ct_seg']}|{inputs['series']}"
    previous = read_json(info_path) if info_path.exists() else {}
    reuse = not force and seg_path.exists() and previous.get("source_key") == source_key

    info: dict[str, Any] = {"source_key": source_key, "warnings": []}
    if reuse:
        info["import"] = previous.get("import")
        info["segmentation"] = "reused"
    elif inputs["ct_seg"]:
        shutil.copyfile(inputs["ct_seg"], seg_path)
        info["import"] = {"source": inputs["ct_seg"], "series": None}
        info["segmentation"] = "provided"
    else:
        info["import"] = _import_ct(case, Path(inputs["ct"]), inputs["series"])
        _run_totalsegmentator(
            case.ct_dir / "ct.nii.gz",
            seg_path,
            device or cfg["ct"]["device"],
            case.ct_dir / "totalseg.log",
        )
        info["segmentation"] = "TotalSegmentator teeth"

    seg = Volume.read(seg_path)
    labels = seg.data.astype(np.int32)
    if labels.max() > fdi.TS_MAX_LABEL:
        raise CaseError(f"{seg_path} does not look like a TotalSegmentator `teeth` labelmap")
    teeth_vol, warnings = merge_labels(labels)
    info["warnings"] += warnings
    info["spacing"] = seg.spacing

    teeth_dir = case.ct_dir / "teeth"
    if teeth_dir.exists():
        shutil.rmtree(teeth_dir)
    bones: dict[str, NDArray[np.bool_]] = {
        "lower": labels == fdi.TS_LOWER_JAWBONE,
        "upper": labels == fdi.TS_UPPER_JAWBONE,
    }
    bone_distance = {
        jaw: ndimage.distance_transform_edt(~mask, sampling=seg.spacing)
        for jaw, mask in bones.items()
    }
    for jaw, mask in bones.items():
        if mask.any():
            write_mesh(case.ct_dir / f"jawbone_{jaw}.ply", _mask_mesh(seg, mask, BONE_FACES))
    np.savez_compressed(
        case.ct_dir / "jawbone.npz",
        upper=bones["upper"],
        lower=bones["lower"],
        origin=seg.origin,
        spacing=seg.spacing,
        direction=seg.direction,
    )

    teeth = []
    for number in sorted(int(v) for v in np.unique(teeth_vol) if v):
        mask = teeth_vol == number
        voxels = int(mask.sum())
        voxel_mm3 = float(np.prod(seg.spacing))
        if voxels * voxel_mm3 < 20.0:
            info["warnings"].append(f"tooth {number} ignored: only {voxels * voxel_mm3:.1f} mm^3")
            continue
        mesh = _mask_mesh(seg, mask, TOOTH_FACES)
        jaw = fdi.jaw_of(number)
        dist = ndimage.map_coordinates(
            bone_distance[jaw], seg.to_index(np.asarray(mesh.vertices)).T, order=1, mode="nearest"
        )
        crown = np.flatnonzero(dist > CROWN_BONE_DISTANCE_MM)
        center, _, axes = pca(seg.to_world(np.argwhere(mask).astype(float)))
        axis = axes[:, 0]
        if len(crown) and np.dot(np.asarray(mesh.vertices)[crown].mean(0) - center, axis) < 0:
            axis = -axis
        write_mesh(teeth_dir / f"{number}.ply", mesh)
        np.save(teeth_dir / f"{number}_bone_dist.npy", dist.astype(np.float32))
        teeth.append(
            {
                "fdi": number,
                "jaw": jaw,
                "volume_mm3": voxels * voxel_mm3,
                "centroid": center,
                "axis": axis,
                "crown_vertices": int(len(crown)),
            }
        )
        if len(crown) < 50:
            info["warnings"].append(
                f"tooth {number}: almost no exposed crown (unerupted or bone label)"
            )
    if not teeth:
        raise CaseError("segmentation contains no teeth")
    info["teeth"] = teeth
    write_json(info_path, info)
    return info
