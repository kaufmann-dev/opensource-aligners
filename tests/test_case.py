import pytest

from aligners.case import PLAN_TEMPLATE, Case, CaseError


def _plan(tmp_path, extra: str) -> dict:
    (tmp_path / "plan.toml").write_text(PLAN_TEMPLATE + extra)
    return Case(tmp_path).plan()


def test_plan_template_defaults_to_no_surgery(tmp_path):
    plan = _plan(tmp_path, "")
    assert plan["surgery"]["mode"] == "none"
    assert plan["surgery"]["maxilla"] == {} and plan["surgery"]["mandible"] == {}
    assert plan["surgery"]["auto"]["move"] == "mandible"


def test_surgery_tables_merge_defaults(tmp_path):
    text = PLAN_TEMPLATE.replace(
        "# incisor_midline = { forward_mm = 4.0, up_mm = 0.0, left_mm = 0.0 }",
        "incisor_midline = { forward_mm = 4.0 }\nmolar_left = { up_mm = -1.5, left_mm = 0.5 }",
    ).replace('mode = "none"', 'mode = "manual"')
    (tmp_path / "plan.toml").write_text(text)
    plan = Case(tmp_path).plan()
    assert plan["surgery"]["maxilla"]["incisor_midline"] == {
        "forward_mm": 4.0,
        "up_mm": 0.0,
        "left_mm": 0.0,
    }
    assert plan["surgery"]["maxilla"]["molar_left"]["up_mm"] == -1.5


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('mode = "none"', 'mode = "both"', "surgery.mode"),
        ('move = "mandible"', 'move = "chin"', "surgery.auto.move"),
        (
            "# incisor_midline = { forward_mm = 4.0, up_mm = 0.0, left_mm = 0.0 }",
            "pogonion = { forward_mm = 4.0 }",
            "unknown landmark",
        ),
        (
            "# incisor_midline = { forward_mm = 4.0, up_mm = 0.0, left_mm = 0.0 }",
            "incisor_midline = { back_mm = 4.0 }",
            "unknown key",
        ),
    ],
)
def test_bad_surgery_entries_are_rejected(tmp_path, old, new, message):
    (tmp_path / "plan.toml").write_text(PLAN_TEMPLATE.replace(old, new))
    with pytest.raises(CaseError, match=message):
        Case(tmp_path).plan()
