import numpy as np
import pandas as pd
import pytest
from frames import local_rows, route_rows

from geo_model.measures.summaries import (
    local_acceptance,
    local_difference,
    local_summary,
    route_utilisation,
    unassigned_rows,
    utilisation_summary,
)
from geo_model.model.settings import SCENARIOS

# --------------------------------------------------------------------------
# unassigned_rows
# --------------------------------------------------------------------------


def test_unassigned_rows_splits_each_scenario_into_its_two_groups():
    results = pd.DataFrame(
        {
            "scenario": ["with routes", "without routes"],
            "n_matched": [900, 800],
            "n_unmatched": [100, 200],
            "n_disadvantaged": [400, 400],
            "unassigned_disadvantaged": [40, 150],
        }
    )

    rows = unassigned_rows(results).set_index("series")

    assert set(rows.index) == {
        f"{group}, {scenario}"
        for group in ("disadvantaged", "other")
        for scenario in SCENARIOS
    }
    # Every unplaced student falls in exactly one group of their scenario.
    counts = rows["unassigned"]
    assert counts["disadvantaged, with routes"] == 40
    assert counts["other, with routes"] == 60
    assert counts["disadvantaged, without routes"] == 150
    assert counts["other, without routes"] == 50


# --------------------------------------------------------------------------
# route_utilisation and utilisation_summary
# --------------------------------------------------------------------------


def test_route_utilisation_averages_each_routes_riders_over_seeds():
    per_route = route_utilisation(route_rows())

    # Route 1 holds more seats, so it comes first.
    np.testing.assert_array_equal(per_route["route"], [1, 0])
    np.testing.assert_array_equal(per_route["capacity"], [5, 4])
    np.testing.assert_array_equal(per_route["disadvantaged"], [1.5, 0.5])
    np.testing.assert_array_equal(per_route["other"], [1.0, 1.0])
    np.testing.assert_array_equal(per_route["used"], [2.5, 1.5])
    np.testing.assert_array_equal(per_route["unused"], [2.5, 2.5])
    np.testing.assert_array_equal(per_route["utilisation"], [0.5, 0.375])


def test_utilisation_summary_totals_the_seats_and_those_used():
    assert utilisation_summary(route_rows()) == (
        "2 routes provide 9 seats, 4.0 of them used, the mean over seeds (44.4%)."
    )


# --------------------------------------------------------------------------
# local_acceptance, local_summary and local_difference
# --------------------------------------------------------------------------


def test_local_acceptance_averages_each_schools_locals_over_seeds():
    per_school = local_acceptance(local_rows())

    # Beta has the most local applicants without routes, so it comes first,
    # and Gamma, with none, has no acceptance rate.
    assert list(per_school["school"]) == ["Beta"] * 2 + ["Alpha"] * 2 + ["Gamma"] * 2
    assert list(per_school["scenario"]) == list(SCENARIOS) * 3
    np.testing.assert_array_equal(per_school["local_applicants"], [8, 8, 5, 5, 0, 0])
    np.testing.assert_array_equal(per_school["accepted"], [7.5, 8, 3, 4, 0, 0])
    np.testing.assert_array_equal(per_school["rejected"], [0.5, 0, 2, 1, 0, 0])
    np.testing.assert_allclose(
        per_school["acceptance"], [0.9375, 1, 0.6, 0.8, np.nan, np.nan]
    )


def test_local_summary_totals_the_applicants_and_those_accepted():
    assert local_summary(local_rows()) == (
        "With routes: 13.0 local students applied to their own school, 10.5 of "
        "them accepted, the mean over seeds (80.8%).\n"
        "Without routes: 13.0 local students applied to their own school, 12.0 "
        "of them accepted, the mean over seeds (92.3%)."
    )


def test_local_difference_pairs_each_schools_accepted_by_seed():
    difference = local_difference(local_rows())

    assert list(difference.columns) == ["seed", "school", "difference"]
    pd.testing.assert_series_equal(
        difference.set_index(["seed", "school"])["difference"],
        pd.Series(
            [-1, -1, 0, -1, 0, 0],
            index=pd.MultiIndex.from_product(
                [[0, 1], ["Alpha", "Beta", "Gamma"]], names=["seed", "school"]
            ),
            name="difference",
        ),
    )


def test_local_difference_rejects_a_school_without_its_pair():
    rows = local_rows()
    unpaired = ~(
        (rows["seed"] == 1)
        & (rows["school"] == "Beta")
        & (rows["scenario"] == "with routes")
    )

    with pytest.raises(ValueError, match=r"\(1, 'Beta'\)"):
        local_difference(rows[unpaired])
