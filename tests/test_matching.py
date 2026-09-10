import numpy as np
import pytest

from geo_model.matching import fast_DAT
from geo_model.utils import rank_bundles

NO_ROUTES = np.empty(0, dtype=np.int32)


def run(preferences, priorities, school_capacities, route_capacities=NO_ROUTES):
    """Call fast_DAT with the dtypes the compiled kernel is written against."""
    return fast_DAT(
        np.asarray(preferences, dtype=np.int32),
        np.asarray(priorities, dtype=np.int32),
        np.asarray(school_capacities, dtype=np.int32),
        np.asarray(route_capacities, dtype=np.int32),
    )


# --------------------------------------------------------------------------
# Explicit instances
# --------------------------------------------------------------------------

def test_a_routeless_instance_reduces_to_deferred_acceptance():
    # Both students want the first school, which holds one seat and prefers
    # the first student, so the second is deferred onto its second choice.
    matching = run(
        preferences=[[[0, -1], [1, -1]], [[0, -1], [1, -1]]],
        priorities=[[[0, 1]], [[0, 1]]],  # (school, routeless, student)
        school_capacities=[1, 1],
    )
    np.testing.assert_array_equal(matching, [[0, -1], [1, -1]])


def test_a_school_never_admits_beyond_its_capacity():
    matching = run(
        preferences=[[[0, -1]], [[0, -1]], [[0, -1]]],
        priorities=[[[0, 1, 2]]],
        school_capacities=[2],
    )
    # The lowest-priority student exhausts its list and stays unmatched.
    np.testing.assert_array_equal(matching, [[0, -1], [0, -1], [-1, -1]])


def test_a_student_rejected_everywhere_is_left_unmatched():
    matching = run(
        preferences=[[[0, -1], [-1, -1]], [[0, -1], [-1, -1]]],
        priorities=[[[0, 1]]],
        school_capacities=[1],
    )
    # The padding a short preference list carries is not a school, so it is
    # passed over rather than matched.
    np.testing.assert_array_equal(matching, [[0, -1], [-1, -1]])


def test_a_route_never_carries_beyond_its_capacity():
    # Both students want the seat on the one route to a school with room for
    # both. The school prefers the first student's routed bundle, so the
    # second falls back to the same school on foot.
    matching = run(
        preferences=[[[0, 0], [0, -1]], [[0, 0], [0, -1]]],
        priorities=[[[0, 1], [2, 3]]],  # (school, route 0 then routeless, student)
        school_capacities=[2],
        route_capacities=[1],
    )
    np.testing.assert_array_equal(matching, [[0, 0], [0, -1]])


def test_a_full_route_holds_when_the_applicant_ranks_below_its_riders():
    matching = run(
        preferences=[[[0, 0], [0, -1]], [[0, 0], [0, -1]]],
        priorities=[[[1, 0], [2, 3]]],  # the route prefers the second student
        school_capacities=[2],
        route_capacities=[1],
    )
    np.testing.assert_array_equal(matching, [[0, -1], [0, 0]])


def test_a_full_school_evicts_its_worst_bundle_across_the_whole_route_axis():
    # The school holds one seat, taken on a route. The applicant walks in
    # without a route but outranks the rider, so it takes the seat and the
    # rider's route seat is given up with it.
    matching = run(
        preferences=[[[0, -1], [-1, -1]], [[0, 0], [1, -1]]],
        priorities=[[[3, 1], [0, 2]], [[2, 2], [1, 0]]],
        school_capacities=[1, 1],
        route_capacities=[1],
    )
    np.testing.assert_array_equal(matching, [[0, -1], [1, -1]])


def test_a_full_school_holds_when_the_applicant_ranks_below_every_bundle():
    matching = run(
        preferences=[[[0, -1], [1, -1]], [[0, 0], [-1, -1]]],
        priorities=[[[0, 1], [2, 3]], [[2, 2], [0, 1]]],
        school_capacities=[1, 1],
        route_capacities=[1],
    )
    # The routed student holds the only seat, so the applicant walks on.
    np.testing.assert_array_equal(matching, [[1, -1], [0, 0]])


def test_an_instance_with_no_students_returns_an_empty_matching():
    matching = run(
        preferences=np.empty((0, 1, 2)),
        priorities=np.empty((1, 1, 0)),
        school_capacities=[5],
    )
    assert matching.shape == (0, 2)


# --------------------------------------------------------------------------
# Stability over instances built by rank_bundles
# --------------------------------------------------------------------------

def instance(seed, n_students=60, n_schools=5, n_districts=4, **kwargs):
    """A random geography, ranked into an SCT instance by rank_bundles."""
    rng = np.random.default_rng(seed)
    student_district = rng.integers(0, n_districts, size=n_students)
    school_district = rng.integers(0, n_districts, size=n_schools)

    # One route from each of the first two districts to each of two schools.
    route_district = np.repeat([0, 1], 2)
    route_school = np.tile([0, n_schools - 1], 2)

    preferences, priorities = rank_bundles(
        rng.normal(size=(n_students, 2)) * 5000, student_district,
        rng.normal(size=(n_schools, 2)) * 5000, school_district,
        route_district, route_school,
        route_discount=0.5,
        **kwargs,
    )
    # Seats are tight but sufficient, so rationing bites without stranding a
    # large tail of students on padding.
    school_capacities = np.full(n_schools, n_students // n_schools + 2, dtype=np.int32)
    route_capacities = np.full(len(route_district), 5, dtype=np.int32)
    return preferences, priorities, school_capacities, route_capacities


def blocking_pairs(preferences, priorities, school_capacities, route_capacities, matching):
    """Every bundle a student both prefers to its match and has a claim on.

    A bundle (c, r) blocks when the student prefers it to what it holds and
    either c has a seat going spare, or c is full but the student outranks a
    bundle it has admitted. A full route narrows the claim to that route's own
    riders, since the student cannot board without displacing one of them.
    """
    n_routes = len(route_capacities)
    rank_of = lambda c, r, s: priorities[c, r if r >= 0 else n_routes, s]

    seats_taken = np.bincount(
        matching[matching[:, 0] >= 0, 0], minlength=len(school_capacities)
    )
    riders = np.bincount(matching[matching[:, 1] >= 0, 1], minlength=max(n_routes, 1))

    blocking = []
    for student, (held_school, held_route) in enumerate(matching):
        for school, route in preferences[student]:
            if (school, route) == (held_school, held_route):
                break  # nothing further down the list is preferred
            if school < 0:
                continue  # padding, not a school

            if route >= 0 and riders[route] >= route_capacities[route]:
                seated = np.flatnonzero(
                    (matching[:, 0] == school) & (matching[:, 1] == route)
                )
            elif seats_taken[school] < school_capacities[school]:
                blocking.append((student, school, route))
                continue
            else:
                seated = np.flatnonzero(matching[:, 0] == school)

            claim = rank_of(school, route, student)
            if any(claim < rank_of(school, matching[o, 1], o) for o in seated):
                blocking.append((student, school, route))

    return blocking


def test_the_stability_check_catches_a_wasted_seat():
    # Nobody is seated, so every school every student ranked has room going
    # spare and no student should escape the check.
    preferences, priorities, school_capacities, route_capacities = instance(0)
    unmatched = np.full((len(preferences), 2), -1, dtype=np.int32)

    blocking = blocking_pairs(
        preferences, priorities, school_capacities, route_capacities, unmatched
    )
    assert {student for student, _, _ in blocking} == set(range(len(preferences)))


def test_the_stability_check_catches_justified_envy():
    # Two students swap places, so the one the school ranks higher now holds
    # the bundle it likes less and envies the other with cause.
    preferences, priorities, school_capacities, route_capacities = instance(0)
    matching = run(preferences, priorities, school_capacities, route_capacities)
    swapped = matching.copy()
    swapped[[0, 1]] = swapped[[1, 0]]

    assert blocking_pairs(
        preferences, priorities, school_capacities, route_capacities, swapped
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_matching_is_stable(seed):
    preferences, priorities, school_capacities, route_capacities = instance(seed)
    matching = run(preferences, priorities, school_capacities, route_capacities)

    assert blocking_pairs(
        preferences, priorities, school_capacities, route_capacities, matching
    ) == []


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_matching_respects_every_capacity(seed):
    preferences, priorities, school_capacities, route_capacities = instance(seed)
    matching = run(preferences, priorities, school_capacities, route_capacities)

    seats_taken = np.bincount(
        matching[matching[:, 0] >= 0, 0], minlength=len(school_capacities)
    )
    riders = np.bincount(
        matching[matching[:, 1] >= 0, 1], minlength=len(route_capacities)
    )
    assert (seats_taken <= school_capacities).all()
    assert (riders <= route_capacities).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_every_student_is_matched_to_a_bundle_it_ranked(seed):
    preferences, priorities, school_capacities, route_capacities = instance(seed)
    matching = run(preferences, priorities, school_capacities, route_capacities)

    for student, bundle in enumerate(matching):
        if bundle[0] < 0:
            assert (bundle == -1).all()  # unmatched carries no route either
            continue
        assert (preferences[student] == bundle).all(axis=1).any()


def test_a_tighter_instance_is_still_stable():
    # Half the students cannot be seated, so schools ration hard and the
    # eviction paths are walked far more often.
    preferences, priorities, school_capacities, route_capacities = instance(9)
    school_capacities = np.full_like(school_capacities, 6)
    matching = run(preferences, priorities, school_capacities, route_capacities)

    assert (matching[:, 0] < 0).any()
    assert blocking_pairs(
        preferences, priorities, school_capacities, route_capacities, matching
    ) == []


def test_a_performance_weighted_instance_is_still_stable():
    preferences, priorities, school_capacities, route_capacities = instance(
        5, school_scores=np.array([-0.7, 0.1, 0.0, 0.5, 1.2]), performance_weight=0.3
    )
    matching = run(preferences, priorities, school_capacities, route_capacities)

    assert blocking_pairs(
        preferences, priorities, school_capacities, route_capacities, matching
    ) == []
