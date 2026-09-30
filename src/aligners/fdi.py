"""FDI tooth numbering tables and the TotalSegmentator `teeth` label map."""

Jaw = str  # "upper" | "lower"
JAWS: tuple[Jaw, Jaw] = ("upper", "lower")

TYPE_NAMES = {
    1: "central incisor",
    2: "lateral incisor",
    3: "canine",
    4: "first premolar",
    5: "second premolar",
    6: "first molar",
    7: "second molar",
    8: "third molar",
}

# Mean root length (mm) by jaw and position; used for the center of resistance without CT.
ROOT_LENGTH_MM = {
    "upper": {1: 13.0, 2: 13.0, 3: 17.0, 4: 14.0, 5: 14.0, 6: 13.0, 7: 13.0, 8: 11.0},
    "lower": {1: 12.5, 2: 14.0, 3: 16.0, 4: 14.0, 5: 14.5, 6: 14.0, 7: 13.0, 8: 11.0},
}

# Mean mesiodistal crown width (mm); plausibility range is +-40 %.
CROWN_WIDTH_MM = {
    "upper": {1: 8.5, 2: 6.6, 3: 7.6, 4: 7.0, 5: 6.6, 6: 10.2, 7: 9.2, 8: 8.6},
    "lower": {1: 5.3, 2: 5.8, 3: 6.8, 4: 7.0, 5: 7.1, 6: 11.1, 7: 10.4, 8: 10.0},
}

# TotalSegmentator `teeth` task (v2.15): label values of interest.
TS_LOWER_JAWBONE = 1
TS_UPPER_JAWBONE = 2
TS_RESTORATIONS = (8, 9, 10)  # bridge, crown, implant
TS_TOOTH_LABELS = {
    11 + 8 * (q - 1) + (p - 1): 10 * q + p for q in (1, 2, 3, 4) for p in range(1, 9)
}
TS_PULP_LABELS = {46 + 8 * (q - 1) + (p - 1): 10 * q + p for q in (1, 2, 3, 4) for p in range(1, 9)}
TS_MAX_LABEL = 77


def quadrant(fdi: int) -> int:
    return fdi // 10


def position(fdi: int) -> int:
    return fdi % 10


def jaw_of(fdi: int) -> Jaw:
    return "upper" if quadrant(fdi) in (1, 2) else "lower"


def is_patient_right(fdi: int) -> bool:
    return quadrant(fdi) in (1, 4)


def is_valid(fdi: int) -> bool:
    return quadrant(fdi) in (1, 2, 3, 4) and 1 <= position(fdi) <= 8


def fdi_number(jaw: Jaw, patient_right: bool, pos: int) -> int:
    if jaw == "upper":
        q = 1 if patient_right else 2
    else:
        q = 4 if patient_right else 3
    return 10 * q + pos


def is_anterior(fdi: int) -> bool:
    return position(fdi) <= 3


def is_canine_or_premolar(fdi: int) -> bool:
    return position(fdi) in (3, 4, 5)


def jaw_teeth(jaw: Jaw) -> list[int]:
    """All 16 FDI numbers of a jaw, ordered from patient-right distal to patient-left distal."""
    right = [fdi_number(jaw, True, p) for p in range(8, 0, -1)]
    left = [fdi_number(jaw, False, p) for p in range(1, 9)]
    return right + left


def arch_order(teeth: list[int]) -> list[int]:
    """Sort present teeth of one jaw from patient-right distal to patient-left distal."""
    if not teeth:
        return []
    order = jaw_teeth(jaw_of(teeth[0]))
    return sorted(teeth, key=order.index)


def describe(fdi: int) -> str:
    side = "right" if is_patient_right(fdi) else "left"
    return f"{jaw_of(fdi)} {side} {TYPE_NAMES[position(fdi)]}"
