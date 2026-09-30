"""Self-contained HTML 3D preview of the staged treatment (three.js from a CDN)."""

import base64
import json
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import KDTree

from aligners.analyze import arch_to_bite, posed_tooth
from aligners.anatomy import JawAnatomy, submesh
from aligners.case import Case, read_json, read_mesh, to_jsonable
from aligners.fdi import JAWS
from aligners.geometry import decimate
from aligners.motion import decompose

TOOTH_FACES = 5000
ROOT_FACES = 2500
GUM_FACES = 60000
GUM_FOLLOW_MM = 3.0


def _pack(mesh: trimesh.Trimesh) -> dict[str, str]:
    return {
        "v": base64.b64encode(np.asarray(mesh.vertices, np.float32).tobytes()).decode(),
        "f": base64.b64encode(np.asarray(mesh.faces, np.uint32).tobytes()).decode(),
    }


def build_preview(case: Case) -> str:
    staged = read_json(case.require(case.work / "stages.json", "stage"))
    checks = (
        read_json(case.work / "checks.json")
        if (case.work / "checks.json").exists()
        else {"findings": []}
    )
    export_plan = (
        read_json(case.work / "export_plan.json")
        if (case.work / "export_plan.json").exists()
        else {}
    )
    to_bite = arch_to_bite(case)
    data: dict[str, Any] = {"jaws": {}, "findings": checks["findings"]}
    for jaw in JAWS:
        anatomy = JawAnatomy.load(case.anatomy_dir(jaw), jaw)
        scan = read_mesh(case.scan_dir / f"{jaw}.ply")
        labels = np.load(case.scan_dir / f"{jaw}_labels.npy")
        gum = decimate(submesh(scan, labels == 0), GUM_FACES, max_error=0.1)
        info = staged["jaws"][jaw]
        poses = [{int(n): np.asarray(T, float) for n, T in st.items()} for st in info["poses"]]
        teeth = {}
        for n in anatomy.order():
            t = anatomy.teeth[n]
            movement = [
                {k: round(v, 2) for k, v in decompose(t, p[n]).as_dict().items()} for p in poses
            ]
            entry: dict[str, Any] = {
                "crown": _pack(decimate(anatomy.patches[n], TOOTH_FACES, max_error=0.05)),
                "poses": [
                    p[n].T.ravel().round(5).tolist() for p in poses
                ],  # column-major for three.js
                "movement": movement,
                "start": info["teeth"][str(n)]["start"],
                "stages": info["teeth"][str(n)]["stages"],
                "label": (posed_tooth(t, np.eye(4)).cusp + t.o * 2.0).round(3).tolist(),
            }
            if n in anatomy.roots:
                entry["root"] = _pack(decimate(anatomy.roots[n], ROOT_FACES, max_error=0.1))
            teeth[str(n)] = entry
        group = np.eye(4)
        if to_bite:
            group = np.linalg.inv(to_bite["lower"]) @ to_bite[jaw]
        # gingiva follows the teeth near it with weight (1 - d/3mm)^2, blending the two nearest
        # crowns by inverse squared distance: a papilla between two moving teeth stretches
        # instead of tearing, and the margin at a crown still follows that crown exactly
        order = anatomy.order()
        gum_v = np.asarray(gum.vertices)
        dist = np.column_stack(
            [KDTree(np.asarray(anatomy.patches[n].vertices)).query(gum_v)[0] for n in order]
        )
        two = np.argsort(dist, axis=1)[:, :2] if len(order) > 1 else np.zeros((len(gum_v), 2), int)
        d2 = np.take_along_axis(dist, two, axis=1)
        inverse = 1.0 / (d2 + 0.05) ** 2
        share = inverse[:, 0] / inverse.sum(axis=1)
        weight = np.clip(1.0 - d2[:, 0] / GUM_FOLLOW_MM, 0.0, 1.0) ** 2
        data["jaws"][jaw] = {
            "gum": _pack(gum),
            "gumTooth": [order[i] for i in two[:, 0]],
            "gumTooth2": [order[i] for i in two[:, 1]],
            "gumShare": share.round(3).tolist(),
            "gumWeight": weight.round(3).tolist(),
            "teeth": teeth,
            "stages": info["stages"],
            "group": group.T.ravel().round(6).tolist(),
            "ipr": info["ipr_mm"],
            "attachments": export_plan.get(jaw, {}).get("attachments", {}),
        }
    html = TEMPLATE.replace("/*DATA*/", json.dumps(to_jsonable(data)))
    path = case.out / "preview.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return str(path)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aligner Preview</title>
<style>
  :root { --bg:#f4f5f7; --panel:#ffffff; --text:#1d2330; --muted:#667085; --accent:#2f6fde; --warn:#b54708; --err:#c01048; }
  @media (prefers-color-scheme: dark) { :root { --bg:#12151b; --panel:#1b2029; --text:#e6e9ef; --muted:#98a2b3; --accent:#6ea0ff; --warn:#fdb022; --err:#f97066; } }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--text); font:14px/1.4 system-ui,sans-serif; }
  #app { display:grid; grid-template-columns: 1fr 320px; height:100%; }
  #view { position:relative; }
  canvas { display:block; }
  aside { background:var(--panel); padding:16px; overflow:auto; border-left:1px solid #0001; }
  h1 { font-size:16px; margin:0 0 4px; } .muted { color:var(--muted); font-size:12px; }
  .row { display:flex; gap:8px; align-items:center; margin:10px 0; flex-wrap:wrap; }
  input[type=range] { width:100%; } button { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:6px 12px; cursor:pointer; }
  label { user-select:none; } table { width:100%; border-collapse:collapse; font-size:12px; } td { padding:2px 4px; border-bottom:1px solid #0001; }
  .warning { color:var(--warn); } .error { color:var(--err); }
  #tip { position:absolute; pointer-events:none; background:var(--panel); color:var(--text); padding:6px 8px; border-radius:6px; font-size:12px; box-shadow:0 2px 8px #0003; display:none; }
  @media (max-width: 800px) { #app { grid-template-columns: 1fr; grid-template-rows: 60vh auto; } aside { border-left:0; } }
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"}}</script>
</head>
<body>
<div id="app">
  <div id="view"><div id="tip"></div></div>
  <aside>
    <h1>Aligner treatment preview</h1>
    <div class="muted">Experimental, not for clinical use.</div>
    <div class="row"><b>Stage <span id="stageLabel">0</span></b> / <span id="stageMax"></span>
      <button id="play">Play</button></div>
    <input id="stage" type="range" min="0" value="0" step="1">
    <div class="row">
      <button data-view="front">Front</button>
      <button data-view="right">Right</button>
      <button data-view="left">Left</button>
      <button data-view="upper">Upper occlusal</button>
      <button data-view="lower">Lower occlusal</button>
    </div>
    <div class="row">
      <label><input type="checkbox" id="upper" checked> Upper</label>
      <label><input type="checkbox" id="lower" checked> Lower</label>
      <label><input type="checkbox" id="gum" checked> Gingiva</label>
      <label><input type="checkbox" id="roots"> Roots (CT)</label>
      <label><input type="checkbox" id="ghost"> Initial ghost</label>
    </div>
    <h1>IPR (before stage 1)</h1><div id="ipr" class="muted"></div>
    <h1>Attachments</h1><div id="att" class="muted"></div>
    <h1>Checks</h1><div id="checks"></div>
    <p class="muted">Hover a tooth for its movement so far (mm / degrees, tooth frame).</p>
  </aside>
</div>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const DATA = /*DATA*/;
const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
camera.up.set(0, 0, 1); camera.position.set(0, 110, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, -20, 0);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.6));
const dir = new THREE.DirectionalLight(0xffffff, 1.2); dir.position.set(30, 80, 60); scene.add(dir);
function decode(p) {
  const v = new Float32Array(Uint8Array.from(atob(p.v), c => c.charCodeAt(0)).buffer);
  const f = new Uint32Array(Uint8Array.from(atob(p.f), c => c.charCodeAt(0)).buffer);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(v, 3)); g.setIndex(new THREE.BufferAttribute(f, 1));
  g.computeVertexNormals(); return g;
}
const toothMat = new THREE.MeshStandardMaterial({ color: 0xf2eee4, roughness: 0.45 });
const gumMat = new THREE.MeshStandardMaterial({ color: 0xe8a0a0, roughness: 0.7, side: THREE.DoubleSide });
const rootMat = new THREE.MeshStandardMaterial({ color: 0xd8c8a0, transparent: true, opacity: 0.5 });
const ghostMat = new THREE.MeshStandardMaterial({ color: 0x8899aa, transparent: true, opacity: 0.25, depthWrite: false });
const hotMat = new THREE.MeshStandardMaterial({ color: 0xffc36b, roughness: 0.45 });
const groups = {}, teeth = [], gums = [], roots = [], ghosts = [], skins = [];
let maxStage = 0;
for (const [jaw, J] of Object.entries(DATA.jaws)) {
  const group = new THREE.Group(); group.matrixAutoUpdate = false; group.matrix.fromArray(J.group);
  scene.add(group); groups[jaw] = group; maxStage = Math.max(maxStage, J.stages);
  const gum = new THREE.Mesh(decode(J.gum), gumMat); group.add(gum); gums.push(gum);
  const rest = gum.geometry.attributes.position.array.slice();
  skins.push({ gum, rest, tooth: J.gumTooth, tooth2: J.gumTooth2, share: J.gumShare, weight: J.gumWeight, jaw });
  for (const [n, T] of Object.entries(J.teeth)) {
    const geo = decode(T.crown);
    const tooth = new THREE.Group(); tooth.matrixAutoUpdate = false;
    const mesh = new THREE.Mesh(geo, toothMat); mesh.userData = { n, T }; tooth.add(mesh);
    if (T.root) { const r = new THREE.Mesh(decode(T.root), rootMat); r.visible = false; tooth.add(r); roots.push(r); }
    group.add(tooth); teeth.push({ tooth, mesh, T, jaw });
    const ghost = new THREE.Mesh(geo, ghostMat); ghost.visible = false; group.add(ghost); ghosts.push(ghost);
  }
}
const slider = document.getElementById('stage'); slider.max = maxStage;
const VIEWS = {  // camera position, view-up; lower arch frame: +z toward the upper arch, +y anterior
  front: [[0, 110, 5], [0, 0, 1], null],
  right: [[-110, -15, 5], [0, 0, 1], null],
  left: [[110, -15, 5], [0, 0, 1], null],
  upper: [[0, -18, -110], [0, 1, 0], 'upper'],
  lower: [[0, -18, 110], [0, 1, 0], 'lower'],
};
function setView(name) {
  const [pos, up, only] = VIEWS[name];
  camera.position.set(...pos); camera.up.set(...up); controls.target.set(0, -18, 0); controls.update();
  for (const jaw of ['upper', 'lower']) {
    const show = !only || only === jaw;
    groups[jaw].visible = show; document.getElementById(jaw).checked = show;
  }
}
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => setView(b.dataset.view));
document.getElementById('stageMax').textContent = maxStage;
function setStage(s) {
  document.getElementById('stageLabel').textContent = s;
  const mats = {};
  for (const t of teeth) {
    const k = Math.min(s, t.T.poses.length - 1); t.tooth.matrix.fromArray(t.T.poses[k]); t.stage = k;
    mats[t.jaw + ':' + t.mesh.userData.n] = t.tooth.matrix;
  }
  // blend gingiva vertices toward the poses of their two nearest teeth
  const v = new THREE.Vector3(), u = new THREE.Vector3();
  for (const sk of skins) {
    const pos = sk.gum.geometry.attributes.position, a = pos.array, r = sk.rest;
    for (let i = 0; i < sk.weight.length; i++) {
      const w = sk.weight[i];
      if (w <= 0) { a[3*i] = r[3*i]; a[3*i+1] = r[3*i+1]; a[3*i+2] = r[3*i+2]; continue; }
      v.set(r[3*i], r[3*i+1], r[3*i+2]).applyMatrix4(mats[sk.jaw + ':' + sk.tooth[i]]);
      u.set(r[3*i], r[3*i+1], r[3*i+2]).applyMatrix4(mats[sk.jaw + ':' + sk.tooth2[i]]);
      v.lerp(u, 1 - sk.share[i]);
      a[3*i] = r[3*i] + w * (v.x - r[3*i]); a[3*i+1] = r[3*i+1] + w * (v.y - r[3*i+1]); a[3*i+2] = r[3*i+2] + w * (v.z - r[3*i+2]);
    }
    pos.needsUpdate = true; sk.gum.geometry.computeVertexNormals();
  }
}
slider.oninput = () => setStage(+slider.value); setStage(0);
window.preview = { setStage: (s) => { slider.value = s; setStage(s); }, setView, maxStage, camera, controls, groups, gums };
let timer = null;
document.getElementById('play').onclick = (e) => {
  if (timer) { clearInterval(timer); timer = null; e.target.textContent = 'Play'; return; }
  e.target.textContent = 'Pause';
  timer = setInterval(() => { slider.value = (+slider.value + 1) % (maxStage + 1); setStage(+slider.value); }, 700);
};
for (const id of ['upper', 'lower']) document.getElementById(id).onchange = (e) => { groups[id].visible = e.target.checked; };
document.getElementById('gum').onchange = (e) => gums.forEach(g => g.visible = e.target.checked);
document.getElementById('roots').onchange = (e) => roots.forEach(g => g.visible = e.target.checked);
document.getElementById('ghost').onchange = (e) => ghosts.forEach(g => g.visible = e.target.checked);
const list = (obj, fmt) => Object.entries(obj).map(fmt).join('<br>') || 'none';
document.getElementById('ipr').innerHTML = Object.entries(DATA.jaws).map(([j, J]) => `<b>${j}</b>: ` + list(J.ipr, ([k, v]) => `${k}: ${v.toFixed(2)} mm`)).join('<br>');
document.getElementById('att').innerHTML = Object.entries(DATA.jaws).map(([j, J]) => `<b>${j}</b>: ` + list(J.attachments, ([k, v]) => `${k} ${v}`)).join('<br>');
document.getElementById('checks').innerHTML = DATA.findings.length ? DATA.findings.map(f => `<div class="${f.severity}">${f.message}</div>`).join('') : '<div class="muted">no findings</div>';
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), tip = document.getElementById('tip');
let hot = null;
renderer.domElement.addEventListener('pointermove', (e) => {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(teeth.map(t => t.mesh))[0];
  if (hot) hot.material = toothMat;
  if (!hit) { tip.style.display = 'none'; hot = null; return; }
  hot = hit.object; hot.material = hotMat;
  const t = teeth.find(x => x.mesh === hot), m = t.T.movement[t.stage];
  tip.innerHTML = `<b>${hot.userData.n}</b> stage ${t.stage}<table>` + Object.entries(m).map(([k, v]) => `<tr><td>${k.replace('_', ' ')}</td><td>${v}</td></tr>`).join('') + '</table>';
  tip.style.display = 'block'; tip.style.left = (e.clientX - r.left + 14) + 'px'; tip.style.top = (e.clientY - r.top + 14) + 'px';
});
function resize() { const w = view.clientWidth, h = view.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resize); resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>
</body>
</html>
"""
