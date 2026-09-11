import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from geo_model.utils import (
    _spread,
    cohort_capacity,
    district_index,
    rank_bundles,
    sample_in_polygon,
    sample_students,
)

UNIT_SQUARE = shapely.box(0.0, 0.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# sample_in_polygon
# --------------------------------------------------------------------------

def test_sample_in_polygon_returns_requested_count_inside_geometry():
    points = sample_in_polygon(
        UNIT_SQUARE, (0.5, 0.5), 50, np.random.default_rng(0)
    )
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
        STUDENT_XY, STUDENT_DISTRICT,
        SCHOOL_XY, SCHOOL_DISTRICT,
        ROUTE_DISTRICT, ROUTE_SCHOOL,
        **kwargs,
    )


def test_rank_bundles_offers_routes_only_to_the_district_they_leave():
    preferences, _ = rank_two_districts(route_discount=0.5)

    # The first student's district has no route, so its option set is the two
    # schools alone and the third slot is padding. The second student takes
    # the discounted route first.
    np.testing.assert_array_equal(
        preferences,
        [[[0, -1], [1, -1], [-1, -1]],
         [[0, 0], [0, -1], [1, -1]]],
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
        np.array([[100.0, 0.0], [1.0, 0.0]]), np.array([0, 1]),
        np.array([[0.0, 0.0]]), np.array([0]),
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
    )
    np.testing.assert_array_equal(priorities[0, 0], [0, 1])


def test_rank_bundles_without_routes_ranks_schools_by_distance_alone():
    preferences, priorities = rank_bundles(
        np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([0, 0]),
        np.array([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0]]), np.array([0, 0, 0]),
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
    )

    assert preferences.shape == (2, 3, 2)
    np.testing.assert_array_equal(preferences[:, :, 0], [[0, 1, 2], [1, 0, 2]])
    assert (preferences[:, :, 1] == -1).all()
    # A routeless instance still carries the route axis, of length one.
    assert priorities.shape == (3, 1, 2)


def test_rank_bundles_full_performance_weight_ranks_on_score_alone():
    preferences, _ = rank_bundles(
        np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([0, 0]),
        np.array([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0]]), np.array([0, 0, 0]),
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
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
        student_xy, district, school_xy, school_district,
        np.array([0, 1]), np.array([2, 4]),
    )
    noisy = rank_bundles(
        student_xy, district, school_xy, school_district,
        np.array([0, 1]), np.array([2, 4]),
        noise_scale=0.5, rng=np.random.default_rng(11),
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
            STUDENT_XY, np.array([0, 1, 0]),
            SCHOOL_XY, SCHOOL_DISTRICT, ROUTE_DISTRICT, ROUTE_SCHOOL,
        )


def test_rank_bundles_rejects_a_district_per_school_of_the_wrong_length():
    with pytest.raises(ValueError, match="one district per school"):
        rank_bundles(
            STUDENT_XY, STUDENT_DISTRICT,
            SCHOOL_XY, np.array([0]), ROUTE_DISTRICT, ROUTE_SCHOOL,
        )


def test_rank_bundles_rejects_misaligned_route_axes():
    with pytest.raises(ValueError, match="must align"):
        rank_bundles(
            STUDENT_XY, STUDENT_DISTRICT, SCHOOL_XY, SCHOOL_DISTRICT,
            np.array([0, 1]), np.array([0]),
        )


def test_rank_bundles_rejects_a_route_to_a_school_that_does_not_exist():
    with pytest.raises(ValueError, match="outside the 2 schools"):
        rank_bundles(
            STUDENT_XY, STUDENT_DISTRICT, SCHOOL_XY, SCHOOL_DISTRICT,
            np.array([1]), np.array([5]),
        )


# --------------------------------------------------------------------------
# district_index
# --------------------------------------------------------------------------

def test_district_index_maps_schools_onto_area_positions(capsys):
    areas = pd.DataFrame({"LSOA21CD": ["A", "B", "C"]})
    schools = pd.DataFrame({
        "LSOA21CD": ["C", "A", "Z"],
        "EstablishmentName": ["Third", "First", "Over the border"],
    })

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
    return pd.DataFrame({
        "EstablishmentName": names or [f"School {i}" for i in range(len(capacity))],
        "SchoolCapacity": capacity,
        "StatutoryLowAge": low,
        "StatutoryHighAge": high,
    })


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
