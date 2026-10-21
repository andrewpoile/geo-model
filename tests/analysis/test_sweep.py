from dataclasses import asdict

import pandas as pd
import pytest
from frames import default_intake, synthetic_frames

from geo_model.analysis.sweep import GRID, default_cell, sweep_tests, swept
from geo_model.measures.statistics import holm, routes_t_test
from geo_model.model.settings import DEFAULTS

# --------------------------------------------------------------------------
# GRID
# --------------------------------------------------------------------------


def test_every_grid_varies_its_own_parameter_alone_and_holds_the_default():
    for parameter, cells in GRID.items():
        assert DEFAULTS in cells
        for cell in cells:
            others = {k: v for k, v in asdict(cell).items() if k != parameter}
            defaults = {k: v for k, v in asdict(DEFAULTS).items() if k != parameter}
            assert others == defaults


@pytest.mark.parametrize("disadvantage", ["score", "decile"])
def test_swept_leaves_the_other_students_route_discount_out_unless_they_ride(
    disadvantage,
):
    assert swept(GRID, disadvantage) == [
        parameter for parameter in GRID if parameter != "route_discount"
    ]


def test_swept_runs_every_grid_when_routes_are_open_to_every_student():
    assert swept(GRID, "score-open") == list(GRID)


# --------------------------------------------------------------------------
# sweep_tests
# --------------------------------------------------------------------------


def test_sweep_tests_tests_every_cell_and_corrects_over_them_all():
    results, *_ = synthetic_frames()

    tests = sweep_tests(results)

    cells = results[["parameter", "value"]].drop_duplicates()
    expected = pd.DataFrame(
        [
            routes_t_test(
                results[
                    (results["parameter"] == parameter) & (results["value"] == value)
                ]
            )
            for parameter, value in zip(cells["parameter"], cells["value"])
        ],
        index=pd.MultiIndex.from_frame(cells),
    ).sort_index()
    pd.testing.assert_frame_equal(tests.drop(columns="p_holm"), expected)
    pd.testing.assert_series_equal(tests["p_holm"], holm(tests["p"]), check_names=False)


# --------------------------------------------------------------------------
# default_cell
# --------------------------------------------------------------------------


def test_default_cell_picks_the_rows_of_the_first_parameters_default():
    _, intake, _, _ = synthetic_frames()

    pd.testing.assert_frame_equal(default_cell(intake, GRID), default_intake(intake))


def test_default_cell_rejects_rows_without_it():
    _, intake, _, _ = synthetic_frames()

    with pytest.raises(ValueError, match="no default cell of"):
        default_cell(intake.drop(default_intake(intake).index), GRID)
