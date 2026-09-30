"""Per-face tooth labeling of intraoral scans (FDI numbers, 0 = gingiva)."""

import shutil
from pathlib import Path
from typing import Any

import maxflow
import numpy as np
import trimesh
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import KDTree

from aligners import fdi
from aligners.anatomy import build_anatomy, submesh
from aligners.case import Case, CaseError, read_json, read_mesh, write_json
from aligners.fdi import JAWS, Jaw
from aligners.geometry import Array, MeshSDF, face_adjacency_graph, transformed

TAU_MM = 0.4  # CT surface uncertainty
SURFACE_SLACK_MM = 0.3  # tooth faces this close to a CT crown cost nothing
GUM_CLEAR_MM = 0.9  # gingiva is at least this far from any CT crown
BAND_MM = 1.5  # graph-cut refinement band around initial label boundaries
PAIRWISE_LAMBDA = 2.0
CREASE_SIGMA_RAD = 0.3
MIN_ISLAND_MM2 = 5.0
MIN_GUM_REGION_MM2 = 20.0
MIN_CROWN_AREA_MM2 = 40.0  # smallest normal visible crown (lower incisor) is ~100 mm^2
CROWN_BONE_MM = 2.0  # CT tooth surface closer to bone than this is root, not visible
CREASE_SMOOTH_PASSES = 8  # Laplacian passes of the concavity measure
GUMLINE_SIGMA_MM = 0.03  # concavity at which cutting along a crease becomes almost free
SNAP_TOOTH_MM = 1.2  # deeper crown faces stay tooth (a cut along a fissure could drop a cusp)
SNAP_GUM_MM = 1.5  # faces farther out on the gingiva than this stay gingiva

Labels = NDArray[np.int64]


def ct_unaries(
    mesh: trimesh.Trimesh, teeth: list[tuple[int, trimesh.Trimesh, trimesh.Trimesh]]
) -> tuple[list[int], Array]:
    """Unary costs per face for [gum, *teeth] from registered CT teeth (full solid, exposed crown).

    A face's distance to tooth k is 0 inside the CT tooth (fissures the smoothed CT surface
    bridges over) and otherwise its distance to the exposed crown, so roots under thin mucosa
    do not attract gingiva.
    """
    centers = np.asarray(mesh.triangles_center)
    labels = [0] + [n for n, _, _ in teeth]
    dist = np.full((len(centers), len(teeth)), np.inf)
    for k, (_, full, crown) in enumerate(teeth):
        solid = MeshSDF.of(full)
        near = np.all((centers > solid.lo - 2.0) & (centers < solid.hi + 2.0), axis=1)
        inside = solid(centers[near], bound=1.0) < 0
        d = (
            MeshSDF.of(crown).unsigned(centers[near])
            if len(crown.faces)
            else np.full(near.sum(), np.inf)
        )
        dist[near, k] = np.where(inside, 0.0, d)
    nearest = dist.min(axis=1) if len(teeth) else np.full(len(centers), np.inf)
    cap = 9.0
    unary = np.empty((len(centers), len(labels)))
    unary[:, 0] = np.minimum(cap, (np.maximum(0.0, GUM_CLEAR_MM - nearest) / TAU_MM) ** 2)
    unary[:, 1:] = np.minimum(cap, (np.maximum(0.0, dist - SURFACE_SLACK_MM) / TAU_MM) ** 2)
    return labels, unary


def _pairwise(mesh: trimesh.Trimesh) -> tuple[NDArray[np.int64], Array]:
    pairs, angles = face_adjacency_graph(mesh)
    v = np.asarray(mesh.vertices)
    e = np.asarray(mesh.face_adjacency_edges)
    length = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    weight = (
        PAIRWISE_LAMBDA
        * (length / length.mean())
        * np.exp(-np.maximum(0.0, -angles) / CREASE_SIGMA_RAD)
    )
    return pairs, weight


def alpha_expansion(
    unary: Array,
    pairs: NDArray[np.int64],
    weights: Array,
    init: Labels,
    active: NDArray[np.bool_],
    cycles: int = 2,
) -> Labels:
    """Potts alpha-expansion over `active` nodes; inactive nodes keep their labels."""
    lab = init.copy()
    n_labels = unary.shape[1]
    nodes = np.flatnonzero(active)
    index = np.full(len(lab), -1)
    index[nodes] = np.arange(len(nodes))
    ai, aj = active[pairs[:, 0]], active[pairs[:, 1]]
    inner = pairs[ai & aj]
    w_inner = weights[ai & aj]
    # Edges with one fixed endpoint: (active node, fixed node, weight)
    b1 = ai & ~aj
    b2 = aj & ~ai
    border_a = np.concatenate([pairs[b1, 0], pairs[b2, 1]])
    border_f = np.concatenate([pairs[b1, 1], pairs[b2, 0]])
    w_border = np.concatenate([weights[b1], weights[b2]])

    for _ in range(cycles):
        changed = False
        for alpha in range(n_labels):
            cur = lab[nodes]
            u0 = unary[nodes, cur].copy()
            u1 = unary[nodes, alpha].copy()
            np.add.at(u0, index[border_a], w_border * (lab[border_a] != lab[border_f]))
            np.add.at(u1, index[border_a], w_border * (alpha != lab[border_f]))
            li, lj = lab[inner[:, 0]], lab[inner[:, 1]]
            A = w_inner * (li != lj)
            B = w_inner * (li != alpha)
            C = w_inner * (alpha != lj)
            ii, jj = index[inner[:, 0]], index[inner[:, 1]]
            u0 += np.bincount(ii, A, len(nodes))
            u1 += np.bincount(ii, C, len(nodes))
            u1 += np.bincount(jj, -C, len(nodes))
            cap = B + C - A
            g = maxflow.Graph[float]()
            ids = g.add_nodes(len(nodes))
            g.add_edges(ii, jj, np.maximum(cap, 0.0), np.zeros_like(cap))
            m = np.minimum(u0, u1)
            g.add_grid_tedges(ids, u1 - m, u0 - m)
            g.maxflow()
            switch = g.get_grid_segments(ids)
            if switch.any():
                new = np.where(switch, alpha, cur)
                if np.any(new != cur):
                    changed = True
                lab[nodes] = new
        if not changed:
            break
    return lab


def _adjacency(n: int, pairs: NDArray[np.int64]) -> sparse.csr_matrix:
    data = np.ones(len(pairs) * 2)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def cleanup(mesh: trimesh.Trimesh, labels: Labels) -> tuple[Labels, list[str]]:
    """Keep one connected region per label, absorb small islands into their surroundings."""
    warnings: list[str] = []
    pairs = np.asarray(mesh.face_adjacency, dtype=np.int64)
    area = np.asarray(mesh.area_faces)
    out = labels.copy()
    same = out[pairs[:, 0]] == out[pairs[:, 1]]
    adj_same = _adjacency(len(out), pairs[same])
    _, comp = connected_components(adj_same, directed=False)
    comp_area = np.bincount(comp, area)
    unassigned = np.zeros(len(out), dtype=bool)
    for value in np.unique(out):
        comps = np.unique(comp[out == value])
        if len(comps) <= 1:
            continue
        keep = comps[np.argmax(comp_area[comps])]
        for c in comps:
            # gingiva may legitimately be several regions (lingual/buccal split by teeth)
            if c == keep or (value == 0 and comp_area[c] >= MIN_GUM_REGION_MM2):
                continue
            if value != 0 and comp_area[c] >= MIN_ISLAND_MM2:
                warnings.append(f"tooth {value}: extra region of {comp_area[c]:.1f} mm^2 removed")
            unassigned |= comp == c
    adj = _adjacency(len(out), pairs)
    out[unassigned] = -1
    for _ in range(200):
        todo = np.flatnonzero(out < 0)
        if len(todo) == 0:
            break
        sub = adj[todo]
        filled = False
        for row, face in zip(range(len(todo)), todo, strict=True):
            nbr = out[sub.indices[sub.indptr[row] : sub.indptr[row + 1]]]
            nbr = nbr[nbr >= 0]
            if len(nbr):
                vals, counts = np.unique(nbr, return_counts=True)
                out[face] = vals[np.argmax(counts)]
                filled = True
        if not filled:
            break
    out[out < 0] = 0
    return out, warnings


def refine(mesh: trimesh.Trimesh, label_values: list[int], unary: Array) -> Labels:
    """Graph-cut refinement in a band around the boundaries of the unary argmin labeling."""
    init = np.argmin(unary, axis=1)
    pairs, weights = _pairwise(mesh)
    boundary = np.zeros(len(init), dtype=bool)
    diff = init[pairs[:, 0]] != init[pairs[:, 1]]
    boundary[pairs[diff].ravel()] = True
    centers = np.asarray(mesh.triangles_center)
    active = np.zeros(len(init), dtype=bool)
    if boundary.any():
        d, _ = KDTree(centers[boundary]).query(centers, distance_upper_bound=BAND_MM)
        active = np.isfinite(d)
    lab = alpha_expansion(unary, pairs, weights, init, active) if active.any() else init
    return np.asarray(label_values)[lab]


def label_report(mesh: trimesh.Trimesh, labels: Labels, jaw: Jaw) -> dict[str, Any]:
    area = np.asarray(mesh.area_faces)
    teeth = sorted(int(v) for v in np.unique(labels) if v)
    report: dict[str, Any] = {"teeth": teeth, "warnings": []}
    present = {fdi.position(t) for t in teeth if fdi.is_patient_right(t)} | {
        fdi.position(t) for t in teeth if not fdi.is_patient_right(t)
    }
    for side in (True, False):
        for pos in range(1, 8):
            number = fdi.fdi_number(jaw, side, pos)
            if number not in teeth and pos in present:
                report["warnings"].append(f"tooth {number} not found in the {jaw} scan")
    report["areas_mm2"] = {t: float(area[labels == t].sum()) for t in teeth}
    return report


# ---------------------------------------------------------------------------------------------
# Case step


def _save_ct_teeth(directory: Path, teeth: dict[int, tuple[Array, trimesh.Trimesh]]) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for number, (_, mesh) in teeth.items():
        mesh.export(directory / f"{number}.ply")
    write_json(directory / "axes.json", {str(n): axis for n, (axis, _) in teeth.items()})


def _load_ct_teeth(directory: Path) -> dict[int, tuple[Array, trimesh.Trimesh]] | None:
    if not (directory / "axes.json").exists():
        return None
    axes = read_json(directory / "axes.json")
    return {
        int(n): (np.asarray(a, float), read_mesh(directory / f"{n}.ply")) for n, a in axes.items()
    }


def render_segmentation(case: Case, jaw: Jaw) -> Path:
    from aligners.render import Item, label_colors, render
    from aligners.scan import patient_right_sign

    mesh = read_mesh(case.scan_dir / f"{jaw}.ply")
    labels = np.load(case.scan_dir / f"{jaw}_labels.npy")
    info = read_json(case.anatomy_dir(jaw) / "teeth.json")
    tags = [(np.asarray(t["cusp"]) + np.array([0, 0, 2.0]), str(t["fdi"])) for t in info["teeth"]]
    right = np.array([patient_right_sign(jaw), 0.0, 0.0])
    return render(
        case.renders / f"segmentation_{jaw}.png",
        [Item(mesh, face_colors=label_colors(labels))],
        tags,
        right,
        f"{jaw} segmentation",
    )


def concavity(mesh: trimesh.Trimesh) -> Array:
    """Per-vertex H = (v - smooth(v)) . n in mm (uniform Laplacian smoothing); negative in
    concave creases such as the gingival margin and interproximal lines."""
    v = np.asarray(mesh.vertices)
    e = np.asarray(mesh.edges_unique)
    adj = _adjacency(len(v), e)
    degree = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1.0)
    smooth = v.copy()
    for _ in range(CREASE_SMOOTH_PASSES):
        smooth = (adj @ smooth) / degree[:, None]
    return np.einsum("ij,ij->i", v - smooth, np.asarray(mesh.vertex_normals))


def snap_gumline(mesh: trimesh.Trimesh, labels: Labels) -> Labels:
    """Move the tooth/gingiva boundary onto the gingival crease.

    The CT-derived labels put the boundary about 0.5 mm onto the gingiva (the CT blurs the
    tooth neck). A binary min-cut over the faces near the boundary, with edge costs that
    almost vanish along concave creases, snaps it onto the real gumline. Faces deep inside
    a crown stay tooth and faces far out on the gingiva stay gingiva."""
    pairs = np.asarray(mesh.face_adjacency, dtype=np.int64)
    edges = np.asarray(mesh.face_adjacency_edges)
    v = np.asarray(mesh.vertices)
    centers = np.asarray(mesh.triangles_center)
    h = concavity(mesh)[edges].mean(axis=1)
    length = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    cost = length * (np.exp(-((np.minimum(h, 0.0) / GUMLINE_SIGMA_MM) ** 2)) + 1e-3)
    n = len(labels)
    step = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
    graph = sparse.csr_matrix(
        (np.r_[step, step], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
        shape=(n, n),
    )
    tooth = labels > 0
    crossing = tooth[pairs[:, 0]] != tooth[pairs[:, 1]]
    if not crossing.any():
        return labels
    reach = max(SNAP_TOOTH_MM, SNAP_GUM_MM) + 0.5
    dist = dijkstra(graph, indices=np.unique(pairs[crossing]), min_only=True, limit=reach)
    signed = np.where(tooth, dist, -dist)  # + inside a crown
    band = np.isfinite(signed)
    keep_tooth = band & (signed >= SNAP_TOOTH_MM)
    for value in np.unique(labels[tooth]):
        own = band & (labels == value)
        if own.any() and not (own & keep_tooth).any():
            # a small crown keeps its deepest part, so it cannot vanish in the cut
            keep_tooth |= own & (signed >= 0.6 * signed[own].max())
    keep_gum = band & (signed <= -SNAP_GUM_MM)
    nodes = np.flatnonzero(band)
    index = np.full(n, -1)
    index[nodes] = np.arange(len(nodes))
    inner = band[pairs[:, 0]] & band[pairs[:, 1]]
    g = maxflow.Graph[float]()
    ids = g.add_nodes(len(nodes))
    g.add_edges(index[pairs[inner, 0]], index[pairs[inner, 1]], cost[inner], cost[inner])
    hard = 1e6
    g.add_grid_tedges(
        ids, np.where(keep_tooth[nodes], hard, 0.0), np.where(keep_gum[nodes], hard, 0.0)
    )
    g.maxflow()
    snapped = tooth.copy()
    snapped[nodes] = ~g.get_grid_segments(ids)
    out = np.where(snapped, labels, 0)
    grown = snapped & ~tooth
    if grown.any():
        # new crown faces take the label of the nearest face that was already that crown
        _, _, nearest = dijkstra(
            graph,
            indices=np.flatnonzero(tooth & band),
            min_only=True,
            limit=reach,
            return_predecessors=True,
        )
        out[grown] = labels[nearest[grown]]
    return out


def run_segment(case: Case) -> dict[str, Any]:
    from aligners.register import bite_disagreement, load_ct_teeth, register_jaw
    from aligners.scan import arch_frame, cusp_plane_leveling, load_scan, scanner_bite

    inputs = case.config()["inputs"]
    case.require(case.ct_dir / "ct.json", "ct")
    case.scan_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, Any] = {"jaws": {}}
    registration: dict[str, Any] = {}
    result: dict[str, Any] = {"jaws": {}, "warnings": []}
    scans: dict[str, trimesh.Trimesh] = {}
    for jaw in JAWS:
        mesh = load_scan(Path(inputs[jaw]))
        scans[jaw] = mesh
        scan_to_arch, frame_info = arch_frame(mesh)
        arch_mesh = transformed(mesh, scan_to_arch)
        teeth = load_ct_teeth(case, jaw)
        if not teeth:
            raise CaseError(f"CT contains no {jaw} teeth")
        reg = register_jaw(teeth, jaw, mesh, scan_to_arch)
        registration[jaw] = reg
        by_fdi = {t.fdi: t for t in teeth}
        ct_teeth: dict[int, tuple[Array, trimesh.Trimesh]] = {}
        for t in reg["teeth"]:
            if not t["visible"]:
                continue
            T = scan_to_arch @ t["transform"]
            ct_teeth[t["fdi"]] = (
                T[:3, :3] @ by_fdi[t["fdi"]].axis,
                transformed(by_fdi[t["fdi"]].mesh, T),
            )
            if t["rmse_mm"] > 0.35:
                result["warnings"].append(
                    f"{jaw} tooth {t['fdi']}: CT-to-scan fit RMSE {t['rmse_mm']:.2f} mm"
                )
        crowns = []
        for n, (_, m) in ct_teeth.items():
            exposed = by_fdi[n].bone_dist[np.asarray(m.faces)].min(axis=1) > CROWN_BONE_MM
            crowns.append((n, m, submesh(m, exposed)))
        values, unary = ct_unaries(arch_mesh, crowns)
        labels = snap_gumline(arch_mesh, refine(arch_mesh, values, unary))
        labels, warnings = cleanup(arch_mesh, labels)
        labels, dropped = drop_partial_crowns(arch_mesh, labels)
        labels = fill_enclosed_gum(arch_mesh, labels)
        level, level_deg = cusp_plane_leveling(arch_mesh, labels)
        scan_to_arch = level @ scan_to_arch
        arch_mesh = transformed(arch_mesh, level)
        ct_teeth = {
            n: (level[:3, :3] @ ax, transformed(m, level)) for n, (ax, m) in ct_teeth.items()
        }
        frame_info["occlusal_leveling_deg"] = level_deg
        for n in dropped:
            ct_teeth.pop(n, None)
            warnings.append(
                f"tooth {n}: only partly visible in the scan (erupting/covered); treated as gingiva"
            )
        arch_mesh.export(case.scan_dir / f"{jaw}.ply")
        np.save(case.scan_dir / f"{jaw}_labels.npy", labels)
        _save_ct_teeth(case.scan_dir / f"ct_{jaw}", ct_teeth)
        anatomy = build_anatomy(case.anatomy_dir(jaw), jaw, arch_mesh, labels, ct_teeth)
        report = label_report(arch_mesh, labels, jaw)
        render_segmentation(case, jaw)
        frames["jaws"][jaw] = {
            "scan_to_arch": scan_to_arch,
            "ct_to_scan": reg["global"],
            **frame_info,
        }
        result["jaws"][jaw] = {"teeth": report["teeth"]}
        result["warnings"] += [
            f"{jaw}: {w}" for w in warnings + report["warnings"] + anatomy["warnings"]
        ]

    # Bite: the scanner's bite registration if the two scans share one, else the CT's.
    J = frames["jaws"]
    from aligners.anatomy import JawAnatomy

    anat = {jaw: JawAnatomy.load(case.anatomy_dir(jaw), jaw) for jaw in JAWS}
    ok, metrics = scanner_bite(
        anat["upper"], anat["lower"], J["upper"]["scan_to_arch"], J["lower"]["scan_to_arch"]
    )
    upper_crowns = np.array(
        [
            t.crown_points().mean(axis=0)
            for t in load_ct_teeth(case, "upper")
            if len(t.crown_points())
        ]
    )
    disagreement = bite_disagreement(
        registration["upper"]["global"], registration["lower"]["global"], upper_crowns
    )
    frames["scan_bite_check"] = metrics
    if ok:
        frames["bite_source"] = "scans"
        for entry in J.values():
            entry["arch_to_bite"] = np.linalg.inv(entry["scan_to_arch"])
        frames["ct_bite_difference_mm"] = disagreement
        if disagreement > 1.0:
            result["warnings"].append(
                f"the CT bite differs from the scanner bite by {disagreement:.1f} mm (bite block or "
                "open mouth during the CT?); using the scanner bite"
            )
    else:
        frames["bite_source"] = "ct"
        for entry in J.values():
            entry["arch_to_bite"] = np.linalg.inv(entry["scan_to_arch"] @ entry["ct_to_scan"])
        result["warnings"].append(
            "the scans are not bite-registered, so the bite comes from the CT; if the CT was taken "
            "with a bite block or open mouth, bite metrics are wrong (export bite-aligned scans)"
        )
    write_json(case.work / "frames.json", frames)
    write_json(case.work / "registration.json", registration)
    result["bite"] = frames["bite_source"]
    return result


def drop_partial_crowns(mesh: trimesh.Trimesh, labels: Labels) -> tuple[Labels, list[int]]:
    """Relabel as gingiva teeth whose visible crown is far smaller than a normal crown."""
    area = np.asarray(mesh.area_faces)
    out = labels.copy()
    dropped = []
    for n in (int(v) for v in np.unique(labels) if v):
        if area[labels == n].sum() < MIN_CROWN_AREA_MM2:
            out[labels == n] = 0
            dropped.append(n)
    return out, dropped


def fill_enclosed_gum(mesh: trimesh.Trimesh, labels: Labels) -> Labels:
    """Gingiva islands inside one crown (e.g. where a dropped partial crown overlapped its
    neighbor) become that crown; gingiva touching the scan's open edge or several teeth stays."""
    pairs = np.asarray(mesh.face_adjacency, dtype=np.int64)
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    _, comp = connected_components(_adjacency(len(labels), pairs[same]), directed=False)
    open_edge = np.zeros(len(labels), dtype=bool)
    edges = np.sort(np.asarray(mesh.edges), axis=1)
    _, index, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    open_edge[np.asarray(mesh.edges_face)[counts[index.ravel()] == 1]] = True
    out = labels.copy()
    cross = pairs[~same]
    for c in np.unique(comp[labels == 0]):
        inside = comp == c
        if open_edge[inside].any():
            continue
        rim = cross[inside[cross[:, 0]] | inside[cross[:, 1]]].ravel()
        around = np.unique(labels[rim[~inside[rim]]])
        if len(around) == 1 and around[0] != 0:
            out[inside] = around[0]
    return out


def relabel(case: Case, jaw: Jaw, mapping: dict[int, int]) -> dict[str, Any]:
    for old, new in mapping.items():
        if new != 0 and (not fdi.is_valid(new) or fdi.jaw_of(new) != jaw):
            raise CaseError(f"{new} is not a valid {jaw} FDI number")
        if not fdi.is_valid(old):
            raise CaseError(f"{old} is not a valid FDI number")
    mesh = read_mesh(case.require(case.scan_dir / f"{jaw}.ply", "segment"))
    labels = np.load(case.scan_dir / f"{jaw}_labels.npy")
    new_labels = labels.copy()
    for old, new in mapping.items():
        new_labels[labels == old] = new
    new_labels, warnings = cleanup(mesh, new_labels)
    np.save(case.scan_dir / f"{jaw}_labels.npy", new_labels)
    ct_teeth = _load_ct_teeth(case.scan_dir / f"ct_{jaw}")
    if ct_teeth:
        ct_teeth = {mapping.get(n, n): v for n, v in ct_teeth.items() if mapping.get(n, n) != 0}
        _save_ct_teeth(case.scan_dir / f"ct_{jaw}", ct_teeth)
    anatomy = build_anatomy(case.anatomy_dir(jaw), jaw, mesh, new_labels, ct_teeth)
    render_segmentation(case, jaw)
    report = label_report(mesh, new_labels, jaw)
    return {
        "jaw": jaw,
        "teeth": report["teeth"],
        "warnings": warnings + report["warnings"] + anatomy["warnings"],
    }
