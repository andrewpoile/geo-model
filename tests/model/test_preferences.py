import numpy as np
import pandas as pd
import pytest

from geo_model.model.preferences import (
    _spread,
    cohort_capacity,
    district_index,
    rank_bundles,
    secondary_instance,
)
from geo_model.model.students import (
    disadvantaged_students,
    draw_cohort_sizes,
    sample_students,
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


def rank_two_districts(
    student_district=STUDENT_DISTRICT,
    school_district=SCHOOL_DISTRICT,
    route_district=ROUTE_DISTRICT,
    route_school=ROUTE_SCHOOL,
    **kwargs,
):
    return rank_bundles(
        STUDENT_XY,
        student_district,
        SCHOOL_XY,
        school_district,
        route_district,
        route_school,
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


def test_rank_bundles_ranks_bracket_local_and_routed_students_together():
    preferences, ranks = rank_two_districts(route_discount=0.5, priority="bracket")

    # Each rank sits beside its bundle on the student's list, s0's being
    # (c0, -1), (c1, -1) and padding, s1's (c0, r0), (c0, -1) and (c1, -1).
    assert ranks.shape == preferences.shape[:2]
    # c0 sits in district 0. Its top bracket holds local s0 (0m) and routed s1
    # (10m); s1 without a route falls to the bottom bracket. c1 has no route,
    # and only s1 is local to its district.
    np.testing.assert_array_equal(ranks, [[0, 1, -1], [1, 2, 0]])


def test_rank_bundles_offers_no_route_to_a_student_not_eligible_for_one():
    preferences, ranks = rank_two_districts(
        route_discount=0.5, route_eligible=[True, False], priority="bracket"
    )

    # s1 is the only student r0 serves, and without eligibility it ranks the
    # two schools alone, as a student whose district has no route does.
    np.testing.assert_array_equal(preferences[1], [[0, -1], [1, -1], [-1, -1]])
    # c0 no longer hears from (r0, s1), and s1 without a route stays in its
    # bottom bracket, behind local s0.
    np.testing.assert_array_equal(ranks[:, 0], [0, 1])


def test_rank_bundles_eligibility_leaves_a_student_no_route_serves_as_it_is():
    # s0's district has no route, so whether s0 is eligible changes nothing.
    for ranked, default in zip(
        rank_two_districts(route_discount=0.5, route_eligible=[False, True]),
        rank_two_districts(route_discount=0.5),
    ):
        np.testing.assert_array_equal(ranked, default)


@pytest.mark.parametrize(
    "priority, expected", [("bracket", [0, 1]), ("distance", [1, 0])]
)
def test_rank_bundles_ranks_a_distant_local_above_a_near_outsider_by_bracket_alone(
    priority, expected
):
    _, ranks = rank_bundles(
        np.array([[100.0, 0.0], [1.0, 0.0]]),
        np.array([0, 1]),
        np.array([[0.0, 0.0]]),
        np.array([0]),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        priority=priority,
    )
    np.testing.assert_array_equal(ranks[:, 0], expected)


@pytest.mark.parametrize(
    "priority, priority_discount, expected",
    [
        # Each routed bundle ties with its own school without one, which
        # ranks first.
        ("distance", 0.0, [[0, -1], [4, 3], [2, 1]]),
        # s2's route, 6m taken down to 3m, outranks s0 at 4m, and s1's, 10m
        # taken down to 5m, does not.
        ("distance", 0.5, [[1, -1], [2, 4], [0, 3]]),
        # Every routed bundle ties at 0m, the nearer rider ranking first.
        ("distance", 1.0, [[2, -1], [1, 4], [0, 3]]),
        # Local s0 and both routes share the top bracket, whatever the
        # discount.
        ("bracket", 0.0, [[0, -1], [2, 4], [1, 3]]),
        ("bracket", 1.0, [[0, -1], [2, 4], [1, 3]]),
    ],
)
def test_rank_bundles_discounts_routed_bundles_under_distance_priority_alone(
    priority, priority_discount, expected
):
    # s0 lives 4m from the school, in its district, and riders s1 and s2 10m
    # and 6m. A full route discount puts each rider's route first on its list.
    _, ranks = rank_bundles(
        np.array([[4.0, 0.0], [10.0, 0.0], [6.0, 0.0]]),
        np.array([0, 1, 1]),
        np.array([[0.0, 0.0]]),
        np.array([0]),
        np.array([1]),
        np.array([0]),
        route_discount=1.0,
        priority=priority,
        priority_discount=priority_discount,
    )
    np.testing.assert_array_equal(ranks, expected)


def test_rank_bundles_without_routes_ranks_schools_by_distance_alone():
    preferences, ranks = rank_bundles(
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
    # Every school ranks both students by distance, the tie at c1 going to the
    # lower index, and each rank sits beside its school on the list.
    np.testing.assert_array_equal(ranks, [[0, 0, 1], [1, 1, 0]])


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


def test_rank_bundles_ranks_each_student_on_its_own_weights():
    scores = np.array([0.0, 1.0])
    mixed, _ = rank_two_districts(
        school_scores=scores,
        performance_weight=np.array([0.0, 0.3]),
        route_discount=np.array([0.0, 1.0]),
    )
    first, _ = rank_two_districts(
        school_scores=scores, performance_weight=0.0, route_discount=0.0
    )
    second, _ = rank_two_districts(
        school_scores=scores, performance_weight=0.3, route_discount=1.0
    )

    # The two settings rank the second student differently, so its row shows
    # whose weights it was ranked on.
    assert not np.array_equal(first[1], second[1])
    np.testing.assert_array_equal(mixed[0], first[0])
    np.testing.assert_array_equal(mixed[1], second[1])


def by_bundle(preferences, ranks):
    """Each student's ranks ordered by bundle rather than by preference."""
    key = preferences[..., 0] * (preferences[..., 1].max() + 2) + preferences[..., 1]
    return np.take_along_axis(ranks, np.argsort(key, axis=1), axis=1)


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
    # Priorities are always ranked on the unperturbed distances, so every
    # bundle keeps its rank wherever the noise moves it on the list.
    np.testing.assert_array_equal(by_bundle(*quiet), by_bundle(*noisy))


def test_rank_bundles_noise_gives_the_same_tastes_with_routes_and_without():
    rng = np.random.default_rng(3)
    student_xy = rng.normal(size=(40, 2)) * 1000
    district = rng.integers(0, 3, size=40)
    school_xy = rng.normal(size=(6, 2)) * 1000
    school_district = np.arange(6) % 3

    def preferences(route_district, route_school):
        return rank_bundles(
            student_xy,
            district,
            school_xy,
            school_district,
            route_district,
            route_school,
            route_discount=0.5,
            noise_scale=0.5,
            rng=np.random.default_rng(11),
        )[0]

    routed = preferences(np.array([0, 1]), np.array([2, 4]))
    plain = preferences(np.array([], dtype=int), np.array([], dtype=int))

    # Routes only add bundles, so the schools a student ranks without a route
    # keep the order they take with no routes at all.
    assert (routed[..., 1] >= 0).any()
    for student in range(len(student_xy)):
        bundles = routed[student]
        schools = bundles[(bundles[:, 0] >= 0) & (bundles[:, 1] < 0), 0]
        np.testing.assert_array_equal(schools, plain[student, :, 0])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"performance_weight": 1.5}, "performance_weight must lie"),
        ({"performance_weight": -0.1}, "performance_weight must lie"),
        ({"route_discount": 2.0}, "route_discount must lie"),
        ({"route_discount": -1.0}, "route_discount must lie"),
        ({"performance_weight": np.array([0.2, 1.5])}, "performance_weight must lie"),
        ({"route_discount": np.array([-0.1, 0.5])}, "route_discount must lie"),
        ({"route_discount": np.array([0.1, 0.2, 0.3])}, "requested shape"),
        ({"performance_weight": 0.5}, "school_scores is required"),
        (
            {"performance_weight": 0.5, "school_scores": np.array([1.0, 2.0, 3.0])},
            "one score per school",
        ),
        (
            {"performance_weight": 0.5, "school_scores": np.array([1.0, np.nan])},
            "non-finite",
        ),
        ({"priority": "nothing"}, "priority must be one of"),
        ({"priority_discount": 1.5}, "priority_discount must lie"),
        ({"priority_discount": -0.1}, "priority_discount must lie"),
        ({"noise_scale": 0.5}, "noise_scale > 0 needs an rng"),
        ({"student_district": np.array([0, 1, 0])}, "one district per student"),
        ({"school_district": np.array([0])}, "one district per school"),
        ({"route_district": np.array([0, 1])}, "must align"),
        ({"route_school": np.array([5])}, "outside the 2 schools"),
    ],
)
def test_rank_bundles_rejects_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rank_two_districts(**kwargs)


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
    # student carries, so no student is local to it.
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


def test_cohort_capacity_takes_the_places_offered_as_they_stand():
    schools = make_schools([500.0, 210.0], [4.0, 11.0], [11.0, 16.0]).assign(
        PlacesOffered=[60, 180]
    )
    capacity = cohort_capacity(schools)

    np.testing.assert_array_equal(capacity, [60, 180])  # not 500 / 7, 210 / 5
    assert capacity.dtype == np.int32


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
# secondary_instance: against the real data folder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def students(areas):
    """A cohort drawn in the default region: where each student lives, its
    district and whether it is disadvantaged."""
    rng = np.random.default_rng(0)
    student_xy, student_lsoa = sample_students(
        areas["Borders"],
        areas["Centroids"],
        draw_cohort_sizes(areas, "secondary", rng),
        rng,
    )
    disadvantaged = disadvantaged_students(
        student_lsoa, areas, np.random.default_rng(1)
    )
    assert disadvantaged.any() and not disadvantaged.all()
    return student_xy, student_lsoa, disadvantaged


def test_secondary_instance_ranks_nearest_first_without_performance_weight(
    areas, secondary, students
):
    student_xy, student_lsoa, disadvantaged = students

    preferences, _, _, _ = secondary_instance(
        student_xy,
        student_lsoa,
        areas,
        secondary,
        None,
        disadvantaged,
        performance_weight=0.0,
        disadvantaged_performance_weight=0.0,
        noise_scale=0.0,
    )

    school_xy = secondary[["Easting", "Northing"]].to_numpy()
    nearest = np.argmin(
        np.linalg.norm(student_xy[:, None] - school_xy[None], axis=2), axis=1
    )
    np.testing.assert_array_equal(preferences[:, 0, 0], nearest)


def test_secondary_instance_ranks_each_group_on_its_own_weights(
    areas, secondary, routes, students
):
    student_xy, student_lsoa, disadvantaged = students

    def instance(other, own):
        return secondary_instance(
            student_xy,
            student_lsoa,
            areas,
            secondary,
            routes,
            disadvantaged,
            *other,
            *own,
            noise_seed=0,
        )

    mixed = instance((0.1, 0.2), (0.7, 0.9))
    other = instance((0.1, 0.2), (0.1, 0.2))
    own = instance((0.7, 0.9), (0.7, 0.9))

    # The two settings rank the disadvantaged differently, so their rows show
    # whose weights they were ranked on.
    assert not np.array_equal(other[0][disadvantaged], own[0][disadvantaged])
    np.testing.assert_array_equal(mixed[0][~disadvantaged], other[0][~disadvantaged])
    np.testing.assert_array_equal(mixed[0][disadvantaged], own[0][disadvantaged])
    # School priorities do not depend on how students weigh schools, so every
    # bundle keeps its rank wherever the weights move it on the list.
    np.testing.assert_array_equal(by_bundle(*mixed[:2]), by_bundle(*other[:2]))


def test_secondary_instance_ranks_priorities_on_the_rule_and_discount_given(
    areas, secondary, routes, students
):
    student_xy, student_lsoa, disadvantaged = students

    def ranks(priority, priority_discount):
        return secondary_instance(
            student_xy,
            student_lsoa,
            areas,
            secondary,
            routes,
            disadvantaged,
            priority=priority,
            priority_discount=priority_discount,
            noise_seed=0,
        )[1]

    # The discount moves routed bundles under "distance" and none under
    # "bracket", so a rule or a discount left unpassed fails one of these.
    assert not np.array_equal(ranks("distance", 0.0), ranks("distance", 1.0))
    np.testing.assert_array_equal(ranks("bracket", 0.0), ranks("bracket", 1.0))


def test_secondary_instance_offers_other_students_routes_only_when_open(
    areas, secondary, routes, students
):
    student_xy, student_lsoa, disadvantaged = students
    offered = {}
    for disadvantage in ("score", "score-open"):
        preferences, _, _, _ = secondary_instance(
            student_xy,
            student_lsoa,
            areas,
            secondary,
            routes,
            disadvantaged,
            disadvantage=disadvantage,
            noise_seed=0,
        )
        offered[disadvantage] = np.any(preferences[..., 1] >= 0, axis=1)

    # Under "score" the routes carry the disadvantaged alone; open, they carry
    # the other students of the same districts as well.
    assert offered["score"][disadvantaged].any()
    assert not offered["score"][~disadvantaged].any()
    assert offered["score-open"][~disadvantaged].any()
    np.testing.assert_array_equal(
        offered["score-open"][disadvantaged], offered["score"][disadvantaged]
    )
