import numpy as np
import pandas as pd
import pytest

from geo_model.measures.segregation import (
    dissimilarity_index,
    dissimilarity_terms,
    district_groups,
    fsm_share,
    fsm_term,
    gini,
    local_intake,
    lorenz_curve,
    school_intake,
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

    districts = district_groups(areas, student_lsoa, disadvantaged)

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
# fsm_share
# --------------------------------------------------------------------------


def test_fsm_share_scales_the_register_percentage_and_names_a_missing_one(capsys):
    schools = pd.DataFrame(
        {
            "EstablishmentName": ["Alpha School", "Beta School"],
            "PercentageFSM": [55.7, np.nan],
        }
    )

    fsm = fsm_share(schools)

    assert fsm["Alpha School"] == pytest.approx(0.557)
    assert np.isnan(fsm["Beta School"])
    assert "drawn blank: Beta School" in capsys.readouterr().out


# --------------------------------------------------------------------------
# fsm_term
# --------------------------------------------------------------------------


def test_fsm_term_takes_the_register_counts_over_the_schools_holding_them(capsys):
    # Alpha takes its 50% over 40 of its 60 pupils, as a school with a sixth
    # form does, so 20 are eligible and 20 are not; Gamma has 10 of 40.
    schools = pd.DataFrame(
        {
            "EstablishmentName": ["Alpha School", "Beta School", "Gamma School"],
            "FSM": [20, np.nan, 10],
            "PercentageFSM": [50.0, np.nan, 25.0],
        }
    )

    terms = fsm_term(schools)

    # 30 eligible and 50 not over Alpha and Gamma alone: Beta holds no figures.
    assert terms["Alpha School"] == pytest.approx(20 / 30 - 20 / 50)
    assert terms["Gamma School"] == pytest.approx(10 / 30 - 30 / 50)
    assert np.isnan(terms["Beta School"])
    assert "left out of the FSM totals: Beta School" in capsys.readouterr().out


def test_fsm_term_rejects_a_school_recording_no_pupil_eligible():
    schools = pd.DataFrame(
        {
            "EstablishmentName": ["Alpha School", "Beta School"],
            "FSM": [20, 0],
            "PercentageFSM": [50.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="cannot be recovered: Beta School"):
        fsm_term(schools)
