"""Offscreen PNG renders (VTK) of arch-frame geometry: occlusal, frontal and side views."""

import colorsys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import vtk
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from aligners import fdi

VIEW_SIZE = (600, 450)
VIEWS = ("occlusal", "frontal", "right", "left")


@dataclass
class Item:
    mesh: trimesh.Trimesh
    color: tuple[float, float, float] = (0.85, 0.82, 0.75)
    face_colors: NDArray[np.uint8] | None = None
    opacity: float = 1.0


def tooth_color(number: int) -> tuple[float, float, float]:
    if number == 0:
        return (0.93, 0.62, 0.62)
    hue = ((fdi.position(number) - 1) / 8.0 + (0.5 if fdi.quadrant(number) % 2 == 0 else 0.0)) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.55, 0.95)


def label_colors(labels: NDArray[np.int64]) -> NDArray[np.uint8]:
    out = np.zeros((len(labels), 3), dtype=np.uint8)
    for value in np.unique(labels):
        out[labels == value] = (np.array(tooth_color(int(value))) * 255).astype(np.uint8)
    return out


def _polydata(mesh: trimesh.Trimesh, face_colors: NDArray[np.uint8] | None) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64), deep=True))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    offsets = numpy_to_vtkIdTypeArray(
        np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True
    )
    conn = numpy_to_vtkIdTypeArray(faces.ravel(), deep=True)
    cells = vtk.vtkCellArray()
    cells.SetData(offsets, conn)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    if face_colors is not None:
        arr = numpy_to_vtk(np.ascontiguousarray(face_colors), deep=True)
        arr.SetName("colors")
        poly.GetCellData().SetScalars(arr)
    return poly


def _camera(
    view: str, right: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Camera direction (from focus toward camera) and view-up for a named view."""
    z = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])
    if view == "occlusal":
        return z, y
    if view == "frontal":
        return y, z
    if view == "right":
        return right, z
    if view == "left":
        return -right, z
    raise ValueError(view)


def render(
    path: Path,
    items: Sequence[Item],
    labels: Sequence[tuple[NDArray[np.float64], str]] = (),
    right: NDArray[np.float64] | None = None,
    title: str = "",
    views: Sequence[str] = VIEWS,
) -> Path:
    """Render items into one PNG with one panel per view."""
    right = np.array([1.0, 0.0, 0.0]) if right is None else right
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    for item in items:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(_polydata(item.mesh, item.face_colors))
        if item.face_colors is not None:
            mapper.SetColorModeToDirectScalars()
            mapper.SetScalarModeToUseCellData()
        else:
            mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*item.color)
        actor.GetProperty().SetOpacity(item.opacity)
        ren.AddActor(actor)
    for pos, text in labels:
        t = vtk.vtkBillboardTextActor3D()
        t.SetInput(text)
        t.SetPosition(*pos)
        t.GetTextProperty().SetFontSize(14)
        t.GetTextProperty().SetColor(0.05, 0.05, 0.3)
        t.GetTextProperty().SetBold(True)
        t.GetTextProperty().SetJustificationToCentered()
        ren.AddActor(t)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(*VIEW_SIZE)
    win.AddRenderer(ren)
    bounds = np.array(ren.ComputeVisiblePropBounds()).reshape(3, 2)
    center = bounds.mean(axis=1)
    panels = []
    for view in views:
        direction, up = _camera(view, right)
        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center + 200 * direction))
        cam.SetViewUp(*up)
        cam.SetParallelProjection(True)
        ren.ResetCamera()
        cam.Zoom(1.25)
        win.Render()
        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(win)
        grab.Update()
        img = grab.GetOutput()
        w, h, _ = img.GetDimensions()
        arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3]
        panel = Image.fromarray(np.ascontiguousarray(arr))
        ImageDraw.Draw(panel).text((8, 6), f"{title}  [{view}]", fill=(0, 0, 0))
        panels.append(panel)
    cols = 2 if len(panels) > 1 else 1
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * VIEW_SIZE[0], rows * VIEW_SIZE[1]), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * VIEW_SIZE[0], (i // cols) * VIEW_SIZE[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path
