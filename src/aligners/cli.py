"""`aligners` command-line interface."""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from aligners.case import Case, CaseError, init_case, to_jsonable

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Experimental clear-aligner pipeline: scans and/or CT to printable aligners. "
    "Not for clinical use.",
)
console = Console(stderr=True)

CaseArg = Annotated[Path, typer.Argument(help="Case directory")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print a JSON document on stdout")]


def _emit(as_json: bool, result: dict[str, Any], summary: Callable[[dict[str, Any]], None]) -> None:
    if as_json:
        sys.stdout.write(json.dumps(to_jsonable(result), indent=2) + "\n")
    else:
        summary(result)
        for w in result.get("warnings", []):
            console.print(f"[yellow]warning:[/] {w}")


def _fail(as_json: bool, message: str, code: int = 1) -> None:
    if as_json:
        sys.stdout.write(json.dumps({"error": message}) + "\n")
    else:
        console.print(f"[red]error:[/] {message}")
    raise typer.Exit(code)


def _run(
    as_json: bool, fn: Callable[[], dict[str, Any]], summary: Callable[[dict[str, Any]], None]
):
    try:
        with console.status("working…", spinner="dots") if not as_json else _null():
            result = fn()
    except CaseError as e:
        _fail(as_json, str(e))
        return
    _emit(as_json, result, summary)
    return result


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


@app.command()
def init(
    case: CaseArg,
    upper: Annotated[str, typer.Option(help="Upper jaw scan (STL/PLY/OBJ)")] = "",
    lower: Annotated[str, typer.Option(help="Lower jaw scan (STL/PLY/OBJ)")] = "",
    ct: Annotated[str, typer.Option(help="DICOM directory or CT NIfTI")] = "",
    ct_seg: Annotated[str, typer.Option(help="Existing TotalSegmentator `teeth` labelmap")] = "",
    series: Annotated[str, typer.Option(help="DICOM series UID or number override")] = "",
    as_json: JsonOpt = False,
) -> None:
    """Create a case directory with case.toml and a default plan.toml."""

    def fn() -> dict[str, Any]:
        c = init_case(case, upper, lower, ct, ct_seg, series)
        return {"case": str(c.root), "config": str(c.config_path), "plan": str(c.plan_path)}

    _run(as_json, fn, lambda r: console.print(f"created case [bold]{r['case']}[/]"))


@app.command("ct")
def ct_cmd(
    case: CaseArg,
    force: Annotated[bool, typer.Option(help="Re-run TotalSegmentator even if cached")] = False,
    device: Annotated[str | None, typer.Option(help="cpu or gpu (default from case.toml)")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Import the CT, segment teeth with TotalSegmentator, and mesh every tooth."""
    from aligners.ct import run_ct

    def summary(r: dict[str, Any]) -> None:
        teeth = [t["fdi"] for t in r["teeth"]]
        console.print(f"CT teeth ({len(teeth)}): {' '.join(map(str, teeth))}")

    _run(as_json, lambda: run_ct(Case(case), force, device), summary)


@app.command()
def segment(case: CaseArg, as_json: JsonOpt = False) -> None:
    """Find arch frames, register CT teeth, label every scan face, and build tooth anatomy."""
    from aligners.segment import run_segment

    def summary(r: dict[str, Any]) -> None:
        for jaw, info in r["jaws"].items():
            console.print(f"{jaw} ({info['method']}): {' '.join(map(str, info['teeth']))}")
        console.print(f"bite: {r['bite'] or 'unavailable'}")

    _run(as_json, lambda: run_segment(Case(case)), summary)


@app.command("relabel")
def relabel_cmd(
    case: CaseArg,
    jaw: Annotated[str, typer.Option(help="upper or lower")],
    map_: Annotated[str, typer.Option("--map", help="Comma list old=new, e.g. 21=22,22=21,17=0")],
    as_json: JsonOpt = False,
) -> None:
    """Correct tooth numbers of a segmented scan (0 = gingiva) and rebuild its anatomy."""
    from aligners.segment import relabel

    def fn() -> dict[str, Any]:
        if jaw not in ("upper", "lower"):
            raise CaseError("--jaw must be upper or lower")
        try:
            mapping = {int(a): int(b) for a, b in (p.split("=") for p in map_.split(",") if p)}
        except ValueError as e:
            raise CaseError(f"bad --map '{map_}'") from e
        return relabel(Case(case), jaw, mapping)  # type: ignore[arg-type]

    _run(as_json, fn, lambda r: console.print(f"{r['jaw']}: {' '.join(map(str, r['teeth']))}"))


@app.command("render")
def render_cmd(
    case: CaseArg,
    what: Annotated[str, typer.Option(help="segmentation | setup | stage:N | bite")],
    as_json: JsonOpt = False,
) -> None:
    """Render PNGs (occlusal, frontal, right, left) into work/renders/."""
    from aligners.views import render_what

    _run(
        as_json,
        lambda: render_what(Case(case), what),
        lambda r: console.print("\n".join(r["files"])),
    )


@app.command()
def analyze(case: CaseArg, as_json: JsonOpt = False) -> None:
    """Measure crowding/spacing, rotations, leveling, arch widths, midlines and bite."""
    from aligners.analyze import run_analyze

    def summary(r: dict[str, Any]) -> None:
        for jaw, a in r["jaws"].items():
            d = a["discrepancy_mm"]
            kind = "crowding" if d > 0 else "spacing"
            console.print(f"{jaw}: {abs(d):.1f} mm {kind}, Spee {a['spee_depth_mm']:.1f} mm")
        if r.get("bite"):
            b = r["bite"]
            console.print(
                f"overjet {b.get('overjet_mm', float('nan')):.1f} mm, "
                f"overbite {b.get('overbite_mm', float('nan')):.1f} mm, "
                f"molars {', '.join(f'{k} {v["class"]}' for k, v in b['molar_relationship'].items())}"
            )

    _run(as_json, lambda: run_analyze(Case(case)), summary)


@app.command()
def setup(case: CaseArg, as_json: JsonOpt = False) -> None:
    """Solve target tooth positions from plan.toml goals and per-tooth overrides."""
    from aligners.setup import run_setup
    from aligners.views import render_what

    def fn() -> dict[str, Any]:
        result = run_setup(Case(case))
        result["renders"] = render_what(Case(case), "setup")["files"]
        return result

    def summary(r: dict[str, Any]) -> None:
        for jaw, j in r["jaws"].items():
            ipr = ", ".join(f"{k}: {v:.2f}" for k, v in j["ipr_mm"].items()) or "none"
            console.print(f"{jaw}: incisors {j['incisor_change_mm']:+.1f} mm, IPR {ipr}")

    _run(as_json, fn, summary)


@app.command()
def stage(case: CaseArg, as_json: JsonOpt = False) -> None:
    """Split the setup into aligner stages within per-stage limits, avoiding collisions."""
    from aligners.staging import run_stage

    def fn() -> dict[str, Any]:
        result = run_stage(Case(case))
        for j in result["jaws"].values():
            j.pop("poses")
        return result

    def summary(r: dict[str, Any]) -> None:
        for jaw, j in r["jaws"].items():
            console.print(f"{jaw}: {j['stages']} stages")

    _run(as_json, fn, summary)


@app.command()
def check(case: CaseArg, as_json: JsonOpt = False) -> None:
    """Validate the plan: stage limits, collisions, roots, bite, fabricated parts. Exit 2 on errors."""
    from aligners.checks import run_check

    def summary(r: dict[str, Any]) -> None:
        for f in r["findings"]:
            color = "red" if f["severity"] == "error" else "yellow"
            console.print(f"[{color}]{f['severity']}[/] {f['message']}")
        console.print("[green]ok[/]" if r["ok"] else f"[red]{r['errors']} error(s)[/]")

    result = _run(as_json, lambda: run_check(Case(case)), summary)
    if result is not None and not result["ok"]:
        raise typer.Exit(2)


@app.command()
def export(
    case: CaseArg,
    stages: Annotated[str, typer.Option(help="Only these stages, e.g. 3 or 1-5")] = "",
    jobs: Annotated[int, typer.Option(help="Parallel worker processes")] = 4,
    as_json: JsonOpt = False,
) -> None:
    """Write aligner shells, thermoforming models, bonding templates, manifest and preview."""
    from aligners.fabricate import run_export
    from aligners.preview import build_preview

    def fn() -> dict[str, Any]:
        result = run_export(Case(case), stages, jobs)
        result["preview"] = build_preview(Case(case))
        return result

    def summary(r: dict[str, Any]) -> None:
        console.print(f"{len(r['files'])} files in {Path(case) / 'out'}; preview: {r['preview']}")

    _run(as_json, fn, summary)


@app.command()
def preview(case: CaseArg, as_json: JsonOpt = False) -> None:
    """Write out/preview.html, an interactive 3D view of every stage."""
    from aligners.preview import build_preview

    _run(
        as_json,
        lambda: {"preview": build_preview(Case(case))},
        lambda r: console.print(r["preview"]),
    )


@app.command()
def run(
    case: CaseArg,
    force: Annotated[bool, typer.Option(help="Export even if checks report errors")] = False,
    jobs: Annotated[int, typer.Option(help="Parallel worker processes for export")] = 4,
    as_json: JsonOpt = False,
) -> None:
    """Run every step: ct, segment, analyze, setup, stage, check, export."""
    from aligners.analyze import run_analyze
    from aligners.checks import run_check
    from aligners.ct import run_ct
    from aligners.fabricate import run_export
    from aligners.preview import build_preview
    from aligners.segment import run_segment
    from aligners.setup import run_setup
    from aligners.staging import run_stage
    from aligners.views import render_what

    def fn() -> dict[str, Any]:
        c = Case(case)
        steps: dict[str, Any] = {}
        steps["ct"] = run_ct(c)
        steps["segment"] = run_segment(c)
        steps["analyze"] = run_analyze(c)
        steps["setup"] = run_setup(c)
        render_what(c, "setup")
        render_what(c, "bite")
        render_what(c, "bite:final")
        stage_result = run_stage(c)
        steps["stage"] = {j: {"stages": v["stages"]} for j, v in stage_result["jaws"].items()}
        steps["check"] = run_check(c)
        if steps["check"]["ok"] or force:
            steps["export"] = run_export(c, "", jobs)
            steps["check"] = run_check(c)
            steps["preview"] = build_preview(c)
        warnings = [w for s in steps.values() if isinstance(s, dict) for w in s.get("warnings", [])]
        return {"steps": steps, "warnings": warnings, "ok": steps["check"]["ok"]}

    def summary(r: dict[str, Any]) -> None:
        stages = r["steps"]["stage"]
        console.print(
            f"stages: upper {stages['upper']['stages']}, lower {stages['lower']['stages']}; "
            f"checks {'ok' if r['ok'] else 'FAILED'}"
        )
        if "preview" in r["steps"]:
            console.print(f"preview: {r['steps']['preview']}")

    result = _run(as_json, fn, summary)
    if result is not None and not result["ok"]:
        raise typer.Exit(2)
