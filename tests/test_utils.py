import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from geo_model.load_data import NTS_BANDS
from geo_model.matching import fast_DAT
from geo_model.utils import (
    MILE,
    MODES,
    _spread,
    cohort_capacity,
    dissimilarity_index,
    district_index,
    expected_modes,
    mode_change,
    rank_bundles,
    reserve_routes,
    route_eligibility,
    sample_in_polygon,
    sample_students,
    school_intake,
    trip_band,
    unreserve,
    withhold_routes,
)

UNIT_SQUARE = shapely.box(0.0, 0.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# sample_in_polygon
# --------------------------------------------------------------------------


def test_sample_in_polygon_returns_requested_count_inside_geometry():
    points = sample_in_polygon(UNIT_SQUARE, (0.5, 0.5), 50, np.random.default_rng(0))
    assert points.shape == (50, 2)
    assert shapely.contains_xy(UNIT_SQUARE, points[:, 0], points[:, 1]).all()


def test_sample_in_polygon_is_reproducible_from_the_seed():
    a = sample_in_polygon(UNIT_SQUARE, (0.5, 0.5), 20, np.random.default_rng(7))
    b = sample_in_polygon(UNIT_SQUARE, (0.5, 0.5), 20, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "geom, size",
    [(UNIT_SQUARE, 0), (shapely.Polygon(), 10)],
    ids=["no-points-wanted", "empty-polygon"],
)
def test_sample_in_polygon_degenerate_cases_return_no_points(geom, size):
    points = sample_in_polygon(geom, (0.5, 0.5), size, np.random.default_rng(0))
    assert points.shape == (0, 2)


def test_sample_in_polygon_raises_when_centre_lies_far_outside():
    # The spread is set by the polygon's own bounds, so a centre 1000 widths
    # away never lands a candidate inside and rejection sampling cannot finish.
    with pytest.raises(RuntimeError, match="Rejection sampling failed"):
        sample_in_polygon(UNIT_SQUARE, (1000.0, 1000.0), 5, np.random.default_rng(0))


# --------------------------------------------------------------------------
# sample_students
# --------------------------------------------------------------------------


@pytest.fixture
def two_areas():
    borders = gpd.GeoSeries([shapely.box(0, 0, 1, 1), shapely.box(10, 0, 11, 1)])
    centroids = gpd.GeoSeries([shapely.Point(0.5, 0.5), shapely.Point(10.5, 0.5)])
    return borders, centroids


def test_sample_students_places_each_area_its_own_students(two_areas):
    borders, centroids = two_areas
    xy, area = sample_students(
        borders, centroids, np.array([3, 5]), np.random.default_rng(1)
    )

    assert xy.shape == (8, 2)
    np.testing.assert_array_equal(area, [0, 0, 0, 1, 1, 1, 1, 1])
    for i, geom in enumerate(borders):
        assert shapely.contains_xy(geom, xy[area == i, 0], xy[area == i, 1]).all()


def test_sample_students_with_no_students_returns_empty_arrays(two_areas):
    borders, centroids = two_areas
    xy, area = sample_students(
        borders, centroids, np.array([0, 0]), np.random.default_rng(1)
    )
    assert xy.shape == (0, 2)
    assert area.shape == (0,)


def test_sample_students_rejects_misaligned_inputs(two_areas):
    borders, centroids = two_areas
    with pytest.raises(ValueError, match="must align"):
        sample_students(
            borders, centroids, np.array([1, 2, 3]), np.random.default_rng(1)
        )


# --------------------------------------------------------------------------
# _spread
# --------------------------------------------------------------------------


def test_spread_is_the_standard_deviation():
    assert _spread(np.array([1.0, 3.0]), "Values") == pytest.approx(1.0)


def test_spread_rejects_a_constant_array():
    # Scaling by a zero spread yields NaN, which argsort orders arbitrarily
    # rather than failing, so the ranking would be silently meaningless.
    with pytest.raises(ValueError, match="zero spread"):
        _spread(np.full(4, 2.0), "Values")


# --------------------------------------------------------------------------
# rank_bundles: a two-district, two-school, one-route instance
# --------------------------------------------------------------------------
#
#   districts   0                     1
#   students    s0 (0, 0)             s1 (10, 0)
#   schools     c0 (0, 0)             c1 (20, 0)
#   routes      r0: district 1 -> c0
#
# Distances are s0: [0, 20] and s1: [10, 10].

STUDENT_XY = np.array([[0.0, 0.0], [10.0, 0.0]])
STUDENT_DISTRICT = np.array([0, 1])
SCHOOL_XY = np.array([[0.0, 0.0], [20.0, 0.0]])
SCHOOL_DISTRICT = np.array([0, 1])
ROUTE_DISTRICT = np.array([1])
ROUTE_SCHOOL = np.array([0])


def rank_two_districts(**kwargs):
    return rank_bundles(
        STUDENT_XY,
        STUDENT_DISTRICT,
        SCHOOL_XY,
        SCHOOL_DISTRICT,
        ROUTE_DISTRICT,
        ROUTE_SCHOOL,
        **kwargs,
    )


def test_rank_bundles_offers_routes_only_to_the_district_they_leave():
    preferences, _ = rank_two_districts(route_discount=0.5)

    # The first student's district has no route, so its option set is the two
    # schools alone and the third slot is padding. The second student takes
    # the discounted route first.
    np.testing.assert_array_equal(
        preferences,
        [[[0, -1], [1, -1], [-1, -1]], [[0, 0], [0, -1], [1, -1]]],
    )


def test_rank_bundles_zero_discount_leaves_a_route_below_its_own_school():
    # Cost ties, and the sort is stable with the routeless bundles laid out
    # first, so a seat is only taken on a route that earns it.
    preferences, _ = rank_two_districts(route_discount=0.0)
    np.testing.assert_array_equal(preferences[1], [[0, -1], [1, -1], [0, 0]])


def test_rank_bundles_full_route_discount_puts_routed_bundles_first():
    preferences, _ = rank_two_districts(route_discount=1.0)
    np.testing.assert_array_equal(preferences[1, 0], [0, 0])


def test_rank_bundles_priorities_bracket_local_and_routed_students_together():
    _, priorities = rank_two_districts(route_discount=0.5)

    assert priorities.shape == (2, 2, 2)  # (school, route + routeless, student)
    # c0 sits in district 0. Its top bracket holds local s0 (0m) and routed s1
    # (10m); s1 without a route falls to the bottom bracket. The (r0, s0) cell
    # is a bundle c0 never hears from, so it takes the filler rank 3.
    np.testing.assert_array_equal(priorities[0], [[3, 1], [0, 2]])
    # c1 has no route, and only s1 is local to its district.
    np.testing.assert_array_equal(priorities[1], [[2, 2], [1, 0]])


def test_rank_bundles_priorities_put_a_distant_local_above_a_near_outsider():
    _, priorities = rank_bundles(
        np.array([[100.0, 0.0], [1.0, 0.0]]),
        np.array([0, 1]),
        np.array([[0.0, 0.0]]),
        np.array([0]),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    np.testing.assert_array_equal(priorities[0, 0], [0, 1])


def test_rank_bundles_without_routes_ranks_schools_by_distance_alone():
    preferences, priorities = rank_bundles(
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([0, 0]),
        np.array([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0]]),
        np.array([0, 0, 0]),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )

    assert preferences.shape == (2, 3, 2)
    np.testing.assert_array_equal(preferences[:, :, 0], [[0, 1, 2], [1, 0, 2]])
    assert (preferences[:, :, 1] == -1).all()
    # A routeless instance still carries the route axis, of length one.
    assert priorities.shape == (3, 1, 2)


def test_rank_bundles_full_performance_weight_ranks_on_score_alone():
    preferences, _ = rank_bundles(
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([0, 0]),
        np.array([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0]]),
        np.array([0, 0, 0]),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        school_scores=np.array([0.0, 2.0, 1.0]),
        performance_weight=1.0,
    )
    np.testing.assert_array_equal(preferences[:, :, 0], [[1, 2, 0], [1, 2, 0]])


def test_rank_bundles_noise_perturbs_preferences_but_not_priorities():
    rng = np.random.default_rng(3)
    student_xy = rng.normal(size=(40, 2)) * 1000
    district = rng.integers(0, 3, size=40)
    school_xy = rng.normal(size=(6, 2)) * 1000
    school_district = np.arange(6) % 3

    quiet = rank_bundles(
        student_xy,
        district,
        school_xy,
        school_district,
        np.array([0, 1]),
        np.array([2, 4]),
    )
    noisy = rank_bundles(
        student_xy,
        district,
        school_xy,
        school_district,
        np.array([0, 1]),
        np.array([2, 4]),
        noise_scale=0.5,
        rng=np.random.default_rng(11),
    )

    assert not np.array_equal(quiet[0], noisy[0])
    # Priorities are always ranked on the unperturbed distances.
    np.testing.assert_array_equal(quiet[1], noisy[1])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"performance_weight": 1.5}, "performance_weight must lie"),
        ({"performance_weight": -0.1}, "performance_weight must lie"),
        ({"route_discount": 2.0}, "route_discount must lie"),
        ({"route_discount": -1.0}, "route_discount must lie"),
        ({"performance_weight": 0.5}, "school_scores is required"),
        (
            {"performance_weight": 0.5, "school_scores": np.array([1.0, 2.0, 3.0])},
            "one score per school",
        ),
        (
            {"performance_weight": 0.5, "school_scores": np.array([1.0, np.nan])},
            "non-finite",
        ),
    ],
)
def test_rank_bundles_rejects_bad_preference_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rank_two_districts(**kwargs)


def test_rank_bundles_rejects_a_district_per_student_of_the_wrong_length():
    with pytest.raises(ValueError, match="one district per student"):
        rank_bundles(
            STUDENT_XY,
            np.array([0, 1, 0]),
            SCHOOL_XY,
            SCHOOL_DISTRICT,
            ROUTE_DISTRICT,
            ROUTE_SCHOOL,
        )


def test_rank_bundles_rejects_a_district_per_school_of_the_wrong_length():
    with pytest.raises(ValueError, match="one district per school"):
        rank_bundles(
            STUDENT_XY,
            STUDENT_DISTRICT,
            SCHOOL_XY,
            np.array([0]),
            ROUTE_DISTRICT,
            ROUTE_SCHOOL,
        )


def test_rank_bundles_rejects_misaligned_route_axes():
    with pytest.raises(ValueError, match="must align"):
        rank_bundles(
            STUDENT_XY,
            STUDENT_DISTRICT,
            SCHOOL_XY,
            SCHOOL_DISTRICT,
            np.array([0, 1]),
            np.array([0]),
        )


def test_rank_bundles_rejects_a_route_to_a_school_that_does_not_exist():
    with pytest.raises(ValueError, match="outside the 2 schools"):
        rank_bundles(
            STUDENT_XY,
            STUDENT_DISTRICT,
            SCHOOL_XY,
            SCHOOL_DISTRICT,
            np.array([1]),
            np.array([5]),
        )


# --------------------------------------------------------------------------
# route_eligibility
# --------------------------------------------------------------------------


def test_route_eligibility_by_routeless_seat_reads_the_seat_held():
    school_xy = np.array([[0.0, 0.0], [20.0, 0.0]])
    # s0 sits at its school, s1 holds no seat, s2 is exactly the threshold
    # from its school and s3 is beyond it.
    student_xy = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [5.0, 0.0]])
    routeless = np.array([[0, -1], [-1, -1], [1, -1], [1, -1]])

    eligible = route_eligibility(student_xy, school_xy, routeless, 10.0, "routeless")

    np.testing.assert_array_equal(eligible, [False, True, False, True])


def test_route_eligibility_by_nearest_school_ignores_the_seat_held():
    school_xy = np.array([[0.0, 0.0], [20.0, 0.0]])
    # s0 is unseated but sits at a school, s1 is exactly the threshold from
    # its nearest school and s2 is beyond it.
    student_xy = np.array([[0.0, 0.0], [30.0, 0.0], [31.0, 0.0]])
    routeless = np.array([[-1, -1], [1, -1], [1, -1]])

    eligible = route_eligibility(student_xy, school_xy, routeless, 10.0, "nearest")

    np.testing.assert_array_equal(eligible, [False, False, True])


def test_route_eligibility_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="rule must be"):
        route_eligibility(
            np.zeros((1, 2)), np.zeros((1, 2)), np.array([[0, -1]]), 1.0, "district"
        )


# --------------------------------------------------------------------------
# withhold_routes, reserve_routes and unreserve: on the two-district instance
# --------------------------------------------------------------------------


def test_withhold_routes_pads_out_the_routed_bundles_of_ineligible_students():
    preferences, _ = rank_two_districts(route_discount=0.5)
    before = preferences.copy()

    withheld = withhold_routes(preferences, np.array([True, False]))

    # The second student loses its route in place, keeping the rest of the
    # list in order, and the first student's list is untouched.
    np.testing.assert_array_equal(withheld[0], before[0])
    np.testing.assert_array_equal(withheld[1], [[-1, -1], [0, -1], [1, -1]])
    np.testing.assert_array_equal(preferences, before)


def test_withhold_routes_leaves_eligible_students_alone():
    preferences, _ = rank_two_districts(route_discount=0.5)
    np.testing.assert_array_equal(
        withhold_routes(preferences, np.array([True, True])), preferences
    )


def test_withhold_routes_rejects_a_flag_per_student_of_the_wrong_length():
    preferences, _ = rank_two_districts()
    with pytest.raises(ValueError, match="one flag per student"):
        withhold_routes(preferences, np.array([True]))


def test_reserve_routes_makes_each_route_a_pool_at_its_school():
    preferences, priorities = rank_two_districts(route_discount=0.5)

    pooled, pool_priorities, pool_capacities = reserve_routes(
        preferences, priorities, np.array([1, 1]), ROUTE_SCHOOL, np.array([1])
    )

    # The routed bundle (c0, r0) becomes pool 2, and no pool carries a route.
    np.testing.assert_array_equal(pooled[0], preferences[0])
    np.testing.assert_array_equal(pooled[1], [[2, -1], [0, -1], [1, -1]])
    # Each school pool takes its routeless ranks, the route pool the ranks of
    # its riders at c0.
    assert pool_priorities.shape == (3, 1, 2)
    np.testing.assert_array_equal(pool_priorities[:, 0], [[0, 2], [1, 0], [3, 1]])
    np.testing.assert_array_equal(pool_capacities, [1, 1, 1])
    assert pool_capacities.dtype == np.int32


def test_reserve_routes_rejects_a_route_axis_of_the_wrong_length():
    preferences, priorities = rank_two_districts()
    with pytest.raises(ValueError, match="hold 1 routes"):
        reserve_routes(
            preferences, priorities, np.array([1, 1]), np.array([0, 0]), np.array([1])
        )


def test_unreserve_maps_a_route_pool_back_to_its_school_and_route():
    matching = np.array([[0, -1], [2, -1], [-1, -1]], dtype=np.int32)
    np.testing.assert_array_equal(
        unreserve(matching, 2, ROUTE_SCHOOL), [[0, -1], [0, 0], [-1, -1]]
    )


def test_unreserve_rejects_a_pool_beyond_the_routes():
    with pytest.raises(ValueError, match="beyond the 2 schools and 1 routes"):
        unreserve(np.array([[3, -1]]), 2, ROUTE_SCHOOL)


# A routed applicant outranks a local student at its school. c0 sits in
# district 0 with one seat; s0 is local to it at 5m, s1 is in district 1 at
# 2m and rides the one route there. Distances are s0: [5, 15], s1: [2, 18].
CONTESTED = {
    "student_xy": np.array([[5.0, 0.0], [2.0, 0.0]]),
    "student_district": STUDENT_DISTRICT,
    "school_xy": SCHOOL_XY,
    "school_district": SCHOOL_DISTRICT,
    "route_district": ROUTE_DISTRICT,
    "route_school": ROUTE_SCHOOL,
    "route_discount": 0.5,
}
ONE_SEAT_EACH = np.array([1, 1], dtype=np.int32)
ONE_RIDER = np.array([1], dtype=np.int32)


def test_a_routed_student_takes_the_local_students_seat_without_reserves():
    preferences, priorities = rank_bundles(**CONTESTED)
    matching = fast_DAT(preferences, priorities, ONE_SEAT_EACH, ONE_RIDER)
    np.testing.assert_array_equal(matching, [[1, -1], [0, 0]])


def test_a_routed_student_rides_on_a_reserved_seat_beside_the_local_student():
    preferences, priorities = rank_bundles(**CONTESTED)
    pooled = reserve_routes(
        preferences, priorities, ONE_SEAT_EACH, ROUTE_SCHOOL, ONE_RIDER
    )

    matching = unreserve(fast_DAT(*pooled, ONE_RIDER[:0]), 2, ROUTE_SCHOOL)

    # c0 now seats both: the local student on its one seat, the rider on
    # the route's.
    np.testing.assert_array_equal(matching, [[0, -1], [0, 0]])


def test_a_student_withheld_a_route_never_holds_one():
    preferences, priorities = rank_bundles(**CONTESTED)
    withheld = withhold_routes(preferences, np.array([True, False]))

    matching = fast_DAT(withheld, priorities, ONE_SEAT_EACH, ONE_RIDER)

    # On foot the rider falls to the bottom bracket at c0, so the local
    # student keeps the seat and the rider is deferred to c1.
    np.testing.assert_array_equal(matching, [[0, -1], [1, -1]])


# --------------------------------------------------------------------------
# district_index
# --------------------------------------------------------------------------


def test_district_index_maps_schools_onto_area_positions(capsys):
    areas = pd.DataFrame({"LSOA21CD": ["A", "B", "C"]})
    schools = pd.DataFrame(
        {
            "LSOA21CD": ["C", "A", "Z"],
            "EstablishmentName": ["Third", "First", "Over the border"],
        }
    )

    index = district_index(schools, areas)

    # A school outside every area the model holds takes -1, an index no
    # student carries, so nobody holds local priority there.
    np.testing.assert_array_equal(index, [2, 0, -1])
    assert index.dtype == np.int64
    assert "Over the border" in capsys.readouterr().out


def test_district_index_says_nothing_when_every_school_sits_inside(capsys):
    areas = pd.DataFrame({"LSOA21CD": ["A", "B"]})
    schools = pd.DataFrame({"LSOA21CD": ["B"], "EstablishmentName": ["Only"]})

    np.testing.assert_array_equal(district_index(schools, areas), [1])
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# cohort_capacity
# --------------------------------------------------------------------------


def make_schools(capacity, low, high, names=None):
    return pd.DataFrame(
        {
            "EstablishmentName": names or [f"School {i}" for i in range(len(capacity))],
            "SchoolCapacity": capacity,
            "StatutoryLowAge": low,
            "StatutoryHighAge": high,
        }
    )


def test_cohort_capacity_spreads_capacity_over_the_years_a_school_spans():
    schools = make_schools([500.0, 210.0], [4.0, 11.0], [11.0, 16.0])
    capacity = cohort_capacity(schools)

    np.testing.assert_array_equal(capacity, [71, 42])  # rint(500 / 7), 210 / 5
    assert capacity.dtype == np.int32


def test_cohort_capacity_rejects_a_school_with_nothing_published():
    schools = make_schools(
        [500.0, np.nan], [4.0, 11.0], [11.0, 16.0], ["Sized", "Unsized"]
    )
    with pytest.raises(ValueError, match="Unsized"):
        cohort_capacity(schools)


def test_cohort_capacity_rejects_an_age_range_covering_no_year_group():
    schools = make_schools([500.0], [11.0], [11.0], ["Ageless"])
    with pytest.raises(ValueError, match="no year group"):
        cohort_capacity(schools)


def test_cohort_capacity_rejects_a_cohort_of_less_than_one_seat():
    schools = make_schools([3.0], [11.0], [18.0], ["Tiny"])
    with pytest.raises(ValueError, match="fewer than one seat"):
        cohort_capacity(schools)


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
# dissimilarity_index
# --------------------------------------------------------------------------


def test_dissimilarity_index_is_zero_when_every_school_holds_the_city_mix():
    # Two schools, each with one disadvantaged student and two others.
    matched = [0, 0, 0, 1, 1, 1]
    disadvantaged = [True, False, False, True, False, False]
    assert dissimilarity_index(matched, disadvantaged, 2) == 0.0


def test_dissimilarity_index_is_one_under_complete_segregation():
    matched = [0, 0, 1, 1, 1]
    disadvantaged = [True, True, False, False, False]
    assert dissimilarity_index(matched, disadvantaged, 2) == 1.0


def test_dissimilarity_index_matches_a_hand_worked_case():
    # a = [3, 1] of 4, b = [1, 3] of 4: 1/2 * (|3/4 - 1/4| + |1/4 - 3/4|) = 1/2.
    matched = [0, 0, 0, 1, 0, 1, 1, 1]
    disadvantaged = [True, True, True, True, False, False, False, False]
    assert dissimilarity_index(matched, disadvantaged, 2) == pytest.approx(0.5)


def test_dissimilarity_index_counts_a_school_nobody_was_matched_to():
    # A third school with no intake contributes nothing, so the index is
    # unchanged by its presence but the array must still span it.
    matched = [0, 0, 1, 1, 1]
    disadvantaged = [True, True, False, False, False]
    assert dissimilarity_index(matched, disadvantaged, 3) == 1.0


def test_dissimilarity_index_leaves_unmatched_students_out():
    # The two unmatched students would make the mix even if counted.
    matched = [0, 0, 1, 1, 1, -1, -1]
    disadvantaged = [True, True, False, False, False, False, True]
    assert dissimilarity_index(matched, disadvantaged, 2) == 1.0


@pytest.mark.parametrize(
    "disadvantaged", [[True, True, True], [False, False, False], [True, False, False]]
)
def test_dissimilarity_index_rejects_an_empty_seated_group(disadvantaged):
    # The last case seats no disadvantaged student, only leaves one unmatched.
    with pytest.raises(ValueError, match="hold a seat, so the index is undefined"):
        dissimilarity_index([-1, 0, 1], disadvantaged, 2)


def test_dissimilarity_index_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        dissimilarity_index([0, 1], [True], 2)


def test_dissimilarity_index_rejects_a_school_beyond_the_count():
    with pytest.raises(ValueError, match="beyond the 2 schools"):
        dissimilarity_index([0, 2], [True, False], 2)


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
            "parameter": ["capacity"] * 4,
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
