# aligners

An experimental, open-source clear-aligner pipeline. It is a command-line tool that turns
intraoral jaw scans and a head CT into a staged orthodontic treatment and 3D-printable
aligners for every stage.

> **Experimental, not for clinical use.** This is a research/hobby project. Nothing it produces
> has been validated, and it must not be used to treat anyone.

## What it does

1. **CT**: picks the thin axial series from a DICOM folder and segments every tooth and both
   jawbones with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (`teeth` task).
   The result provides crown shapes, roots and bone.
2. **Segment**: finds each scan's arch frame, levels it to the occlusal plane, registers the CT
   teeth onto the scan and labels every scan face with its FDI tooth number (0 = gingiva).
   - The tooth/gingiva boundary is snapped onto the gingival crease.
   - The bite comes from bite-registered scans, else from the CT.
3. **Analyze**: measures crowding/spacing, rotations, leveling, arch widths, midlines, overjet,
   overbite, molar class and Bolton ratio.
4. **Setup**: solves the target tooth positions from treatment goals in `plan.toml`.
   - It uses an ideal arch that hugs the outermost teeth, with incisor advancement before IPR.
   - It corrects inclination by tipping, levels the incisal edges and aligns the midlines.
   - It keeps minimum overjet/overbite and clears occlusal interferences.
   - It coordinates both arches: lower IPR replaces proclination when the upper front would be
     left with spaces.
5. **Surgery (optional)**: plans the teeth for the jaw position after orthognathic surgery
   (Le Fort I, BSSO), with orthodontics before surgery.
   - `auto` sets up each arch on its own jaw (decompensation) and solves the jaw movements for
     an ideal occlusion and a level, centered maxilla.
   - `manual` takes the surgeon's planned movements as a landmark table, e.g. the rows of a
     ProPlan CMF report, and fits the teeth to that jaw relation.
   - The facial midline comes from the mirror symmetry of the cranium in the CT.
   - Both write the jaw movements as a table, angles, transforms, landmarks and bone/teeth STLs
     in DICOM coordinates.
6. **Stage**: splits the movement into aligner stages within per-stage limits and schedules
   teeth so neighbors never collide.
7. **Check**: validates stage limits, collisions, registration, roots (proximity, bone), bite
   (after surgery in the surgical modes), jaw movement sizes and the fabricated parts.
8. **Export**: writes, per stage, a trimmed aligner shell for direct printing and a model with
   base and trim-line groove for thermoforming. It also writes an attachment bonding template,
   `manifest.json` and an interactive 3D `preview.html`.

All fabrication geometry is built from signed-distance fields (no mesh booleans), so every shell
is one watertight part.

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 and the locked dependencies).
- Only when starting from DICOM: TotalSegmentator on your `PATH`
  (`pip install TotalSegmentator`; tested with 2.15.0). It runs on the CPU. The first run
  downloads the `teeth` model and takes about 15 minutes. It is not needed when you pass an
  existing TotalSegmentator `teeth` labelmap with `--ct-seg`.

```sh
uv sync --locked
uv run aligners --help
```

## Inputs

A case needs:
- **Upper and lower intraoral scans** (STL/PLY/OBJ). Scans exported in occlusion
  (bite-registered) give the most accurate bite.
- **A CT of the jaws**: a DICOM folder (`--ct`) or an existing TotalSegmentator `teeth`
  labelmap (`--ct-seg`).
- **Surgical modes only**: the CT image (`--ct`, also when `--ct-seg` is given) covering the
  skull above the teeth (orbits, cranial base), so the facial midline can be found.

## Usage

```sh
uv run aligners init cases/me --upper upper.stl --lower lower.stl --ct path/to/dicom
uv run aligners run cases/me            # ct, segment, analyze, setup, stage, check, export
```

Or step by step. Every command takes `--json` for machine-readable output and writes PNG renders
to `cases/me/work/renders/`.

```sh
uv run aligners ct cases/me
uv run aligners segment cases/me       # fix numbering with `aligners relabel` if needed
uv run aligners analyze cases/me
uv run aligners setup cases/me         # edit cases/me/plan.toml and repeat as needed
uv run aligners stage cases/me
uv run aligners check cases/me         # exit code 2 = errors
uv run aligners export cases/me --jobs 4
```

Treatment goals, per-tooth overrides and the surgery mode live in `cases/me/plan.toml`; every key
is documented inline. Fabrication and staging parameters live in `cases/me/case.toml`.

For a surgical case, set `[surgery] mode = "auto"` or `"manual"` in `plan.toml`:
- `auto` chooses the jaw movements. `[surgery.auto]` sets which osteotomies it may use and the
  maxilla's forward/vertical/pitch.
- `manual` reads the movements from `[surgery.maxilla]` and `[surgery.mandible]`: the
  displacement of each landmark in mm, `forward_mm`, `up_mm` and `left_mm`. The landmarks are
  the incisor midline, both canines and both first molars.

`setup` then renders `work/renders/surgery.png`, and `analyze` reports the face metrics (cant,
yaw, midlines, chin, incisor angles).

Outputs are written to `cases/me/out/`:
- `<jaw>/aligner_NN.stl`, `<jaw>/model_NN.stl` and `<jaw>/template.stl`
- `manifest.json`: wear order, IPR before stage 1, attachments
- `preview.html`, with a before/after surgery view in the surgical modes
- surgical modes, in `surgery/`:
  - `movements.toml`: the planned movements in the `plan.toml` input format;
  - `movements.json`: tables, angles, transforms, split gaps;
  - `landmarks.json`: named landmarks in CT coordinates, the head frame and the rule for turning
    another planner's plan into a movement table. bimax-simulation's *Export aligner movements*
    uses it;
  - STLs of the maxilla, mandible segments, cranium and teeth before/after surgery, in CT DICOM
    (LPS) coordinates.

`cases/` is git-ignored: case folders contain medical data.

## AI agents

The CLI is built to be driven by an agent. `.claude/skills/aligner-planning/SKILL.md` is a
[Claude Code](https://claude.com/claude-code) skill with the planning loop and the orthodontic
heuristics for choosing goals and judging results.

## Reproducibility

Results are not stored in the repository; they are recreated from the inputs.
- Dependencies are pinned in `uv.lock` (`uv sync --locked`), and all random sampling uses fixed
  seeds.
- The same inputs, the same `plan.toml`/`case.toml` and the same code produce the same results.
  A rerun from scratch reproduces every file bit for bit on the same machine.
- On another computer, pass the same TotalSegmentator labelmap with `--ct-seg` (copy
  `work/ct/teeth_seg.nii.gz`). Re-running the neural network on other hardware or with another
  TotalSegmentator version can change the segmentation slightly, and with it everything after.
- Different CPUs can round the last digits of floating-point math differently, so bit-identical
  files across machines are likely but not guaranteed. Compare `work/setup.json` and
  `work/stages.json` to confirm the plan is the same.

## Development

```sh
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright
```
