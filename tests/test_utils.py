import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from geo_model import utils
from geo_model.data.travel import NTS_BANDS
from geo_model.utils import (
    MILE,
    MODES,
    dissimilarity_index,
    dissimilarity_terms,
    expected_modes,
    gini,
    holm,
    local_intake,
    lorenz_curve,
    mode_change,
    routes_t_test,
    school_intake,
    trip_band,
)

# --------------------------------------------------------------------------
# school_intake
# --------------------------------------------------------------------------


def test_school_intake_counts_each_group_per_school_leaving_unmatched_out():
    matched = [0, 0, 1, 1, 1, -1, -1]
    disadvantaged = [True, True, False, True, False, True, False]

    group_a, group_b = school_intake(matched, disadvantaged, 3)

    np.testing.assert_array_equal(group_a, [2, 1, 0])
    np.testing.assert_array_equal(group_b, [0, 2, 0])


def test_school_intake_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        school_intake([0, 1], [True], 2)


def test_school_intake_rejects_a_school_beyond_the_count():
    with pytest.raises(ValueError, match="beyond the 2 schools"):
        school_intake([0, 2], [True, False], 2)


# --------------------------------------------------------------------------
# local_intake
# --------------------------------------------------------------------------


# School 0 sits in district 0 and school 1 in district 1, and route 0 runs
# from district 1 to school 0. Students 0 and 1 propose to school 1 first,
# students 2 and 3 to school 0 by the route.
LOCAL_PREFERENCES = np.array(
    [
        [[1, -1], [0, -1], [-1, -1]],
        [[1, -1], [0, -1], [-1, -1]],
        [[0, 0], [1, -1], [0, -1]],
        [[0, 0], [1, -1], [0, -1]],
    ]
)


def test_local_intake_counts_the_locals_who_proposed_and_those_seated():
    # Student 0 is seated at their own school after school 1 rejects them,
    # student 1 at theirs, student 2 is rejected everywhere, and student 3
    # rides to school 0 before ever proposing to their own.
    matching = np.array([[0, -1], [1, -1], [-1, -1], [0, 0]])

    applicants, group_a, group_b = local_intake(
        LOCAL_PREFERENCES, matching, [0, 1, 1, 1], [0, 1], [True, False, False, True]
    )

    np.testing.assert_array_equal(applicants, [1, 2])
    np.testing.assert_array_equal(group_a, [1, 0])
    np.testing.assert_array_equal(group_b, [0, 1])


def test_local_intake_rejects_a_bundle_missing_from_the_list():
    matching = np.array([[0, 0], [1, -1], [-1, -1], [0, 0]])

    with pytest.raises(ValueError, match="missing from their own list"):
        local_intake(LOCAL_PREFERENCES, matching, [0, 1, 1, 1], [0, 1], [False] * 4)


# --------------------------------------------------------------------------
# dissimilarity_terms
# --------------------------------------------------------------------------


def test_dissimilarity_terms_match_a_hand_worked_case():
    # a = [3, 1] of 4, b = [1, 3] of 4: 3/4 - 1/4 and 1/4 - 3/4.
    terms = dissimilarity_terms([3, 1], [1, 3])

    np.testing.assert_allclose(terms, [0.5, -0.5])


def test_dissimilarity_terms_cancel_and_their_half_absolute_sum_is_the_index():
    matched = [0, 0, 0, 1, 0, 1, 1, 2, 2]
    disadvantaged = [True, True, True, True, False, False, False, True, False]

    terms = dissimilarity_terms(*school_intake(matched, disadvantaged, 3))

    assert terms.sum() == pytest.approx(0)
    assert 0.5 * np.abs(terms).sum() == pytest.approx(
        dissimilarity_index(matched, disadvantaged, 3)
    )


@pytest.mark.parametrize("groups", [([0, 0], [1, 2]), ([1, 2], [0, 0])])
def test_dissimilarity_terms_reject_an_empty_group(groups):
    with pytest.raises(ValueError, match="hold a seat, so the index is undefined"):
        dissimilarity_terms(*groups)


def test_dissimilarity_terms_reject_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        dissimilarity_terms([1, 2], [1])


# --------------------------------------------------------------------------
# district_groups
# --------------------------------------------------------------------------


def test_district_groups_compares_each_districts_groups_with_the_regions(capsys):
    # District 0 holds 3 disadvantaged students and 1 advantaged, district 1
    # the reverse, of 4 each in all, and district 2 nobody.
    areas = pd.DataFrame(
        {
            "LSOA21CD": ["E0", "E1", "E2"],
            "LA (name)": ["Southampton"] * 3,
            "IDACI Decile": [1, 5, 9],
            "IDACI Score": [0.6, 0.2, 0.05],
        }
    )
    student_lsoa = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    disadvantaged = np.array([True, True, True, False, True, False, False, False])

    districts = utils.district_groups(areas, student_lsoa, disadvantaged)

    pd.testing.assert_frame_equal(districts.iloc[:, :4], areas)
    np.testing.assert_array_equal(districts["cohort"], [4, 4, 0])
    np.testing.assert_array_equal(districts["disadvantaged"], [3, 1, 0])
    np.testing.assert_array_equal(districts["advantaged"], [1, 3, 0])
    np.testing.assert_array_equal(
        districts["disadvantaged_share"], [0.75, 0.25, np.nan]
    )
    np.testing.assert_array_equal(districts["share_of_disadvantaged"], [0.75, 0.25, 0])
    np.testing.assert_array_equal(districts["share_of_advantaged"], [0.25, 0.75, 0])
    np.testing.assert_array_equal(districts["difference"], [0.5, -0.5, 0])
    assert (
        "No student sampled, so no disadvantaged share: E2" in capsys.readouterr().out
    )


# --------------------------------------------------------------------------
# dissimilarity_index
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matched", "disadvantaged", "n_schools", "expected"),
    [
        # Two schools, each with one disadvantaged student and two others.
        ([0, 0, 0, 1, 1, 1], [True, False, False, True, False, False], 2, 0.0),
        ([0, 0, 1, 1, 1], [True, True, False, False, False], 2, 1.0),
        # a = [3, 1] of 4, b = [1, 3] of 4: 1/2 * (|3/4 - 1/4| + |1/4 - 3/4|).
        ([0, 0, 0, 1, 0, 1, 1, 1], [True] * 4 + [False] * 4, 2, 0.5),
        # A third school with no intake contributes nothing, but the array
        # must still span it.
        ([0, 0, 1, 1, 1], [True, True, False, False, False], 3, 1.0),
    ],
    ids=["city-mix", "complete-segregation", "hand-worked", "unused-school"],
)
def test_dissimilarity_index_matches_known_cases(
    matched, disadvantaged, n_schools, expected
):
    index = dissimilarity_index(matched, disadvantaged, n_schools)

    assert index == pytest.approx(expected)


@pytest.mark.parametrize(
    "disadvantaged", [[True, True, True], [False, False, False], [True, False, False]]
)
def test_dissimilarity_index_rejects_an_empty_seated_group(disadvantaged):
    # The last case seats no disadvantaged student, only leaves one unmatched.
    with pytest.raises(ValueError, match="hold a seat, so the index is undefined"):
        dissimilarity_index([-1, 0, 1], disadvantaged, 2)


# --------------------------------------------------------------------------
# lorenz_curve
# --------------------------------------------------------------------------


def test_lorenz_curve_orders_schools_by_share_from_corner_to_corner():
    # a = [3, 1], b = [1, 3]: the second school, a quarter disadvantaged,
    # comes first and holds half the students and a quarter of group a.
    x, y = lorenz_curve([3, 1], [1, 3])

    np.testing.assert_allclose(x, [0, 0.5, 1])
    np.testing.assert_allclose(y, [0, 0.25, 1])


def test_lorenz_curve_leaves_out_a_school_nobody_was_matched_to():
    x, y = lorenz_curve([3, 0, 1], [1, 0, 3])

    np.testing.assert_allclose(x, [0, 0.5, 1])
    np.testing.assert_allclose(y, [0, 0.25, 1])


def test_lorenz_curve_rejects_no_seated_student_of_the_first_group():
    with pytest.raises(ValueError, match="so the curve is undefined"):
        lorenz_curve([0, 0], [1, 2])


def test_lorenz_curve_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        lorenz_curve([1, 2], [1])


# --------------------------------------------------------------------------
# gini
# --------------------------------------------------------------------------


def test_gini_matches_a_hand_worked_case():
    # Under the curve through (0, 0), (1/2, 1/4), (1, 1): 1/16 + 5/16 = 3/8.
    assert gini(*lorenz_curve([3, 1], [1, 3])) == pytest.approx(1 - 2 * 3 / 8)


def test_gini_is_zero_when_every_school_holds_the_city_mix():
    assert gini(*lorenz_curve([1, 2], [2, 4])) == pytest.approx(0)


def test_gini_under_complete_segregation_is_the_share_outside_the_group():
    # 4 of 10 students are in group a, so the curve reaches 1 - 4/10.
    assert gini(*lorenz_curve([4, 0], [0, 6])) == pytest.approx(0.6)


def test_gini_is_the_mean_difference_of_school_shares():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 50, 8), rng.integers(1, 50, 8)
    t, p = a + b, a / (a + b)

    pairs = t[:, None] * t[None, :] * np.abs(p[:, None] - p[None, :])
    expected = pairs.sum() / (2 * t.sum() ** 2 * (a.sum() / t.sum()))

    assert gini(*lorenz_curve(a, b)) == pytest.approx(expected)


# --------------------------------------------------------------------------
# trip_band
# --------------------------------------------------------------------------


def test_trip_band_bins_on_the_nts_edges():
    miles = np.array([0.0, 0.99, 1.0, 1.99, 2.0, 4.99, 5.0, 40.0])

    bands = trip_band(miles * MILE)

    np.testing.assert_array_equal(
        bands, [NTS_BANDS[i] for i in (0, 0, 1, 1, 2, 2, 3, 3)]
    )


def test_trip_band_scales_by_circuity_before_binning():
    # 0.8 straight-line miles is a 1.04 mile road trip at a circuity of 1.3.
    assert trip_band(np.array([0.8 * MILE]), circuity=1.3)[0] == NTS_BANDS[1]


def test_trip_band_rejects_a_non_positive_circuity():
    with pytest.raises(ValueError, match="circuity must be positive"):
        trip_band(np.array([100.0]), circuity=0)


# --------------------------------------------------------------------------
# expected_modes and mode_change
# --------------------------------------------------------------------------

SHARES = pd.DataFrame(
    {
        "walk": [0.8, 0.5, 0.1, 0.0],
        "cycle": [0.1, 0.1, 0.1, 0.0],
        "car": [0.1, 0.3, 0.5, 0.4],
        "bus": [0.0, 0.1, 0.2, 0.5],
        "other": [0.0, 0.0, 0.1, 0.1],
    },
    index=NTS_BANDS,
)


def test_expected_modes_counts_routed_routeless_and_drops_unseated():
    school_xy = np.array([[0.0, 0.0], [10_000.0, 0.0]])
    # Student 0 walks distance 0 to school 0, student 1 rides a route to
    # school 1, student 2 is 3 miles from school 1 without a route, student 3
    # holds no seat.
    student_xy = np.array(
        [[0.0, 0.0], [0.0, 0.0], [10_000 - 3 * MILE, 0.0], [5.0, 5.0]]
    )
    matching = np.array([[0, -1], [1, 0], [1, -1], [-1, -1]], dtype=np.int32)

    modes = expected_modes(matching, student_xy, school_xy, SHARES)

    expected = SHARES.loc[[NTS_BANDS[0], NTS_BANDS[2]]].sum()
    expected["route"] = 1
    assert list(modes.index) == MODES
    assert modes.to_dict() == pytest.approx(expected.to_dict())
    assert modes.sum() == pytest.approx(3)


def test_mode_change_subtracts_the_unrouted_count_within_every_key():
    results = pd.DataFrame(
        {
            "parameter": ["capacity_scale"] * 4,
            "value": [1, 1, 1, 1],
            "seed": [0, 0, 0, 0],
            "scenario": ["with routes", "without routes"] * 2,
            "mode": ["walk", "walk", "route", "route"],
            "students": [30.0, 50.0, 20.0, 0.0],
        }
    )

    change = mode_change(results)

    assert list(change.columns) == ["parameter", "value", "seed", "mode", "change"]
    assert change.set_index("mode")["change"].to_dict() == {
        "walk": -20.0,
        "route": 20.0,
    }


# --------------------------------------------------------------------------
# routes_t_test and holm
# --------------------------------------------------------------------------


def scenario_rows(with_routes: list[float], without_routes: list[float]):
    """Scenario rows of one setting, seed s scoring `with_routes[s]` with
    routes and `without_routes[s]` without."""
    return pd.DataFrame(
        {"seed": seed, "scenario": scenario, "dissimilarity": index}
        for seed, pair in enumerate(zip(with_routes, without_routes))
        for scenario, index in zip(("with routes", "without routes"), pair)
    )


def test_routes_t_test_is_the_paired_test_over_each_seeds_difference():
    with_routes, without_routes = [0.30, 0.28, 0.33], [0.35, 0.36, 0.34]
    difference = np.subtract(with_routes, without_routes)
    t = difference.mean() / (difference.std(ddof=1) / np.sqrt(3))

    test = routes_t_test(scenario_rows(with_routes, without_routes))

    assert test["seeds"] == 3
    assert test["difference"] == pytest.approx(difference.mean())
    assert test["t"] == pytest.approx(t)
    assert test["p"] == pytest.approx(2 * stats.t.sf(abs(t), 2))


def test_routes_t_test_pairs_the_scenarios_by_seed_not_by_row():
    rows = scenario_rows([0.30, 0.28, 0.33], [0.35, 0.36, 0.34])

    shuffled = rows.sample(frac=1, random_state=0)

    pd.testing.assert_series_equal(routes_t_test(shuffled), routes_t_test(rows))


def test_routes_t_test_is_undefined_where_routes_change_no_seed():
    test = routes_t_test(scenario_rows([0.30, 0.28], [0.30, 0.28]))

    assert test["difference"] == 0
    assert np.isnan(test["t"]) and np.isnan(test["p"])


def test_routes_t_test_is_undefined_over_one_seed_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test = routes_t_test(scenario_rows([0.30], [0.35]))

    assert test["seeds"] == 1
    assert test["difference"] == pytest.approx(-0.05)
    assert np.isnan(test["t"]) and np.isnan(test["p"])


def test_routes_t_test_rejects_a_seed_scored_in_one_scenario_alone():
    rows = scenario_rows([0.30, 0.28], [0.35, 0.36]).iloc[:-1]

    with pytest.raises(ValueError, match="one scenario alone have no pair to test: 1"):
        routes_t_test(rows)


def test_routes_t_test_rejects_a_seed_scored_twice():
    rows = scenario_rows([0.30, 0.28], [0.35, 0.36])

    with pytest.raises(ValueError, match="duplicate"):
        routes_t_test(pd.concat([rows, rows]))


def test_holm_steps_down_over_the_ordered_p_values():
    # Ordered 0.005, 0.01, 0.03, 0.04 they scale to 0.02, 0.03, 0.06, 0.04,
    # and the last takes the 0.06 before it.
    adjusted = holm(pd.Series([0.01, 0.04, 0.03, 0.005]))

    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_keeps_an_undefined_test_out_of_the_count():
    adjusted = holm(pd.Series([0.01, np.nan, 0.02], index=["a", "b", "c"]))

    assert list(adjusted.index) == ["a", "b", "c"]
    assert np.isnan(adjusted["b"])
    assert adjusted[["a", "c"]].tolist() == pytest.approx([0.02, 0.02])


def test_holm_caps_the_adjusted_p_values_at_one():
    assert holm(pd.Series([0.6, 0.9])).tolist() == [1.0, 1.0]
