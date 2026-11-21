---
name: aligner-planning
description: Plan, review and iterate a clear-aligner case with the `aligners` CLI in this repository - choose treatment goals from the analysis, fix tooth numbering, judge renders, and loop setup/stage/check until the plan passes, then export. Use when asked to plan, adjust, review or export an aligner case.
---

# Aligner planning with the `aligners` CLI

Experimental, not for clinical use. Say so whenever you present a plan or its outputs.

## Inputs a case needs

- Upper and lower intraoral scans (STL/PLY/OBJ). Bite-registered scans (exported in occlusion)
  give the most accurate bite; otherwise the bite comes from the CT.
- A head/jaw CT (DICOM folder or NIfTI) or an existing TotalSegmentator `teeth` labelmap.

```sh
uv run aligners init cases/<name> --upper U.stl --lower L.stl --ct <dicom-dir>
```

## Loop

Run every command with `--json` and read the JSON; view PNGs in `work/renders/` with your
image tool.

1. `uv run aligners ct cases/<name> --json` (first run: TotalSegmentator on CPU, ~15 min).
2. `uv run aligners segment cases/<name> --json`. Read `bite` (`scans` or `ct`) and warnings.
   The tooth/gingiva boundary is snapped onto the gingival crease, and the arch frames are
   leveled to the occlusal plane through the tooth tips (`occlusal_leveling_deg` in
   `frames.json`). View `segmentation_upper.png` / `segmentation_lower.png`: every crown one
   color (no colored collar on the gingiva, no gingiva patches inside a crown), labels on
   the right teeth (FDI: 1x upper right, 2x upper left, 3x lower left, 4x lower right; the
   render's "right" panel is the patient's right side). Fix wrong numbers with
   `uv run aligners relabel cases/<name> --jaw upper --map 21=22,22=21` (`N=0` makes a region
   gingiva).
3. `uv run aligners analyze cases/<name> --json`, then `aligners render ... --what bite`.
4. Edit `cases/<name>/plan.toml` (goals + per-tooth overrides; every key is documented there).
5. `uv run aligners setup cases/<name> --json`; view `setup_upper.png`, `setup_lower.png`,
   and `aligners render ... --what bite:final`. Read warnings and `bite_after`. In surgical
   modes also view `surgery.png` and read `surgery` (see Surgical cases).
6. `uv run aligners stage cases/<name> --json`, then `uv run aligners check cases/<name> --json`
   (exit code 2 = errors). Repeat from 4 until `ok` is true and warnings are understood.
7. `uv run aligners export cases/<name> --json` writes `out/<jaw>/aligner_NN.stl`,
   `model_NN.stl`, `template.stl`, `out/manifest.json` and `out/preview.html`.
8. Summarize for the user: stage counts, IPR per contact (done before stage 1), attachments
   (bonded before stage 1 with `template.stl`), open warnings, and the disclaimer.

## Reading `analysis.json`

- `jaws.<jaw>.discrepancy_mm`: + crowding / - spacing on the ideal arch (mm).
- `rotation_deg`: + mesial edge buccal, relative to the ideal arch; `level_deviation_mm`: cusp
  height off the smooth occlusal curve; `spee_depth_mm`; `intercanine_mm`, `intermolar_mm`;
  `midline_offset_left_mm` (dental midline vs arch symmetry plane, + = patient left);
  `contacts_mm` (+ space, - overlap).
- `bite`: `overjet_mm` (normal 2-3), `overbite_mm` (normal 1.5-3),
  `midline_upper_vs_lower_left_mm`, `molar_relationship.<side>.class` (I/II/III, offset mm),
  `posterior_crossbite` (upper FDI numbers), `interferences` (lower/upper pairs, depth mm).
- `bolton_anterior_pct` (norm 77.2 %): large deviations mean the arches cannot both be ideal.

## Choosing goals (`plan.toml`)

- How the solver makes space: the ideal arch hugs the outermost teeth (lingually displaced
  teeth come out, well-placed ones are not pulled in), incisors advance up to
  `max_incisor_advance_mm`, and IPR is used only for what remains (and by `stage` where a
  derotating tooth cannot pass its neighbor otherwise). Bucco-lingual moves are done by
  tipping (`tipping`, `tipping_pivot`), so inclinations change; `harmonize_inclination`
  gives left/right homologous teeth the same inclination, and `level_incisal_edges` tips the
  incisors so their incisal edges are horizontal in the frontal view.
- `anchors` (default: 6s-8s) keep their place along the arch and their occlusion; they only
  follow the arch transversely (expansion). Remove a tooth from the list to align it fully.
- Mild crowding (< 4 mm per arch): the defaults usually suffice. If `setup` reports crowding
  left, add transverse expansion where the arch is narrow (`expand_premolar_mm` <= 3 total,
  `expand_molar_mm` <= 2 total) before raising IPR limits.
- `bite.min_overjet_mm` and `bite.min_overbite_mm` are always kept: proclining incisors
  flattens the bite, so the solver extrudes incisors if the overbite would drop below.
  Lower incisors that would hit the upper ones push the upper front forward (up to
  `max_incisor_advance_mm`), then open the bite by intruding incisors (not below
  `min_overbite_mm`); new posterior interferences intrude the pair and its homologs a little.
  If the upper front then has spaces (lower anterior teeth too wide, see `bolton_anterior_pct`),
  the lower incisors advance less and lower anterior IPR takes over.
- Spacing: `close_spaces = true`; retraction is capped by `max_incisor_retract_mm`. With a small
  overjet, do not retract upper incisors (keep `bite.min_overjet_mm`).
- Deep bite: `spee_flatten` 0.3-0.6 on the lower arch, or `bite.enforce = true` with an
  `overbite_mm` target. Open bite: `bite.enforce` with a positive overbite target.
- Overjet: `bite.enforce = true`, `overjet_mm` 2-3.
- Midlines: `bite.align_midlines` (default on) moves the upper dental midline onto the lower
  one; `midline_shift_mm` per jaw (+ = patient left) moves a jaw's midline.
- Posterior crossbite: expand the upper arch (`expand_premolar_mm`, `expand_molar_mm`).
- Class II/III molar relationships need elastics, extractions or surgery. Do not plan elastics or
  extractions; report them. A skeletal cause can be planned with surgery (see Surgical cases).
- Implants, crowned or ankylosed teeth: `[teeth."NN"] fixed = true`.
- Individual corrections: `[teeth."NN"]` with `mesial_mm`, `buccal_mm`, `extrusion_mm`,
  `rotation_deg`, `tip_deg`, `torque_deg` (tooth frame, about the center of resistance).
- Rotations of round teeth (canines/premolars) > 5 deg and extrusions > 0.5 mm get
  attachments automatically (`[attachments]` to add/remove).

## Surgical cases (`[surgery]`)

Orthodontics come first and surgery follows the last stage: the aligners move the teeth to fit
the jaw relation after surgery. This needs the CT image (`inputs.ct`) showing the cranium above
the teeth; `analyze` then reports `face` in the head frame (+x patient right, +y anterior, +z up).

- Consider surgery when `face` or `bite` shows a skeletal problem that orthodontic compensation
  would hide:
  - occlusal cant > 2 deg or `cant_molars_mm` > 2 mm;
  - chin (`chin_mm`) or dental midlines > 2-3 mm off the facial midline;
  - Class II/III > 3 mm or a negative overjet.
  The surgeon decides; you only plan the teeth for their plan.
- `face.symmetry_rmse_mm` > 1.5 means the facial midline is uncertain: say so.
- `head_pitch_deg` aligns the head horizontal with the surgeon's software. The measured
  `face.maxilla.occlusal_pitch_deg` drops by `head_pitch_deg`. Set it to (measured − the
  report's "Maxillary Occlusal Plane" angle), after confirming both use + = anterior up.
- `mode = "manual"` (the surgeon's plan): copy the dental rows of the plan's point table
  (ProPlan CMF: "Maxilla/Mandible Incisor Midline", "Canine L/R", "1st Molar L/R") into
  `[surgery.maxilla]` / `[surgery.mandible]`.
  - Signs: Ant = + `forward_mm`, Post = −; Up = + `up_mm`, Down = −; Left = + `left_mm`,
    Right = −.
  - Skip the skeletal rows (A, ANS, B, Pog, Me).
  - Give one landmark for a pure translation, otherwise at least three.
  - `setup` refuses entries that are more than 1.5 mm off one rigid movement (sign or side
    mix-up) and warns above 0.3 mm.
  - The teeth are then aligned with the usual goals against the post-surgical bite.
- `mode = "auto"`:
  - Each arch is set up on its own jaw: no compensation for the other jaw, incisors torqued to
    `upper_incisor_deg` (to the horizontal, norm ~110) and `lower_incisor_deg` (IMPA, norm
    ~93). If setup warns "incisor decompensation needs … limited", raise
    `max_incisor_advance_mm` / `max_incisor_retract_mm` or adjust the targets.
  - The solver then places the mandible for Class I and `bite.overjet_mm`. It levels and
    centers the maxilla (cant, yaw, midline), using only `osteotomies`.
  - Maxilla `forward/up/pitch` stay the surgeon's choice (looks, airway): keep them at 0 unless
    told otherwise.
  - `["bsso"]` alone cannot correct the maxilla. With `["lefort1"]` alone the mandible only
    autorotates; residual asymmetry is reported.
- Review:
  - `surgery.png` (grey = before surgery) and `bite_final.png`;
  - the `setup` movement table and angles: after surgery cant and yaw ≈ 0;
  - `split_gap_mm` (+ gap, − overlap at the sagittal splits) and `chin_mm`;
  - `check` warnings: `surgery_magnitude`, `chin` (genioplasty is not planned),
    `split_overlap`, `surgery_fit`.
- Hand the surgeon `out/surgery/movements.toml` (the same format as the manual input),
  `movements.json` and the STLs (DICOM LPS coordinates, overlay on the CT). If they change the
  plan, paste their values into `plan.toml` with `mode = "manual"` and rerun.
- Skeletal landmarks (condyles, gonion, menton, pogonion) and split gaps are approximate.

## Judging results

- More than ~25 stages per arch, or warnings of > 4 mm translation / > 30 deg rotation per
  tooth: goals are too ambitious; relax them.
- `check` errors must be fixed before export: `stage_limit`, `collision`, `registration`,
  `fabrication`. Warnings (`root_proximity`, `root_outside_bone`, `occlusion`, `overjet`,
  `crossbite`, `molar_class`) need a decision you explain to the user.
- `setup` warnings about unresolved crowding mean the goals cannot create enough space.
