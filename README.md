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
   - The tooth/gingiva boundary is snapped onto the gingival crease, and every boundary between
     two touching crowns onto their interproximal crease.
   - Each tooth gets a frame from its own crown: incisors along the incisal edge, canines along
     the cusp ridges, premolars and molars between their contacts. Where a crowded canine
     touches its neighbors says little about how it is turned.
   - Crown surface the scan could not see, for example enamel hidden behind a crowded
     neighbor, is completed from the CT tooth fitted to the scan. Contacts, collisions, the
     printed models and the preview use these completed crowns.
   - The bite comes from bite-registered scans, else from the CT.
3. **Analyze**: measures crowding/spacing (with the setup's own contact walk), rotations,
   leveling, arch widths, midlines, incisor inclinations, overjet, overbite, molar class and
   Bolton ratio.
4. **Setup**: solves the target tooth positions from treatment goals in `plan.toml`.
   - It uses a smooth, symmetric ideal arch through the current incisors and the held molars,
     fitted robustly so that displaced teeth come onto the arch of the others. Each tooth type
     keeps its natural in/out offset.
   - It torques the incisors to their target inclinations (CT long axis) and runs the arch
     through them. IPR resolves the crowding that remains.
   - It corrects other inclinations by tipping and mirrors the incisal edges left to right. It
     aligns the midlines, each arch moving half.
   - It keeps minimum overjet/overbite and clears occlusal interferences. When the upper front
     cannot come far enough forward, the lower front stays back by the rest.
5. **Surgery (optional)**: plans the teeth for a jaw movement after orthognathic surgery, with
   orthodontics before surgery. Only the teeth are used.
   - `auto` aligns each arch on its own jaw, with the incisors at their target inclinations
     (decompensation). It then finds how far one jaw has to move (forward/back, sideways,
     up/down, rotated) for the two arches to fit ideally.
   - `manual` takes the surgeon's planned movements as a landmark table, e.g. the rows of a
     ProPlan CMF report, and fits the teeth to that jaw relation.
   - Both write the jaw movements as tables and angles in a frame of the upper occlusal plane,
     plus transforms, landmarks and the teeth before/after surgery as STLs in DICOM
     coordinates.
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
- `auto` chooses the movement. `[surgery.auto]` sets which jaw moves (`move`, mandible by
  default) and the decompensated incisor inclinations.
- `manual` reads the movements from `[surgery.maxilla]` and `[surgery.mandible]`: the
  displacement of each landmark in mm, `forward_mm`, `up_mm` and `left_mm`. The landmarks are
  the incisor midline, both canines and both first molars.
- Movements are given in a frame of the teeth: origin at the upper incisal midline, horizontal
  along the upper occlusal plane. `[surgery] pitch_deg` tilts that horizontal, e.g. to match a
  surgeon's planning software.

`setup` then renders `work/renders/surgery.png` (teeth only).

Outputs are written to `cases/me/out/`:
- `<jaw>/aligner_NN.stl`, `<jaw>/model_NN.stl` and `<jaw>/template.stl`
- `manifest.json`: wear order, IPR before stage 1, attachments
- `preview.html`, with a before/after surgery toggle in the surgical modes
- surgical modes, in `surgery/`:
  - `movements.toml`: the planned movements in the `plan.toml` input format;
  - `movements.json`: tables, rotations, transforms, the occlusion reached;
  - `landmarks.json`: named landmarks in CT coordinates, the movement frame's axes and the rule
    for turning another planner's plan into a movement table. bimax-simulation's *Export
    aligner movements* uses it;
  - `<jaw>_teeth_before_surgery.stl` / `<jaw>_teeth_after_surgery.stl`: the planned teeth in CT
    DICOM (LPS) coordinates.

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
