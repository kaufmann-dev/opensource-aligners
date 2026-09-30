import numpy as np

from aligners.ct import SeriesInfo, merge_labels, select_series


def test_merge_labels_joins_pulp_and_restorations():
    seg = np.zeros((20, 20, 20), dtype=np.int32)
    seg[2:8, 2:8, 2:8] = 11  # tooth 11
    seg[4:6, 4:6, 4:6] = 46  # its pulp
    seg[8:10, 2:8, 2:8] = 9  # a crown restoration touching it
    seg[15:17, 15:17, 15:17] = 9  # a stray restoration touching nothing
    out, warnings = merge_labels(seg)
    assert set(np.unique(out)) == {0, 11}
    assert (out[4:6, 4:6, 4:6] == 11).all() and (out[8:10, 2:8, 2:8] == 11).all()
    assert len(warnings) == 1


def _series(
    number: str, files: int, spacing: float, slice_mm: float, axial: bool = True
) -> SeriesInfo:
    return SeriesInfo(
        number, number, "", "CT", ["f"] * files, 512, 512, (spacing, spacing), slice_mm, axial
    )


def test_select_series_prefers_finest_axial():
    chosen = select_series(
        [
            _series("2", 92, 0.34, 2.0),
            _series("6", 231, 0.315, 0.8),
            _series("9", 202, 0.33, 0.8, axial=False),
        ],
        "",
    )
    assert chosen.number == "6"
