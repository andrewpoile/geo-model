import numpy as np
import pandas as pd
import shapely
from numpy.typing import ArrayLike
from scipy.spatial import distance as spdist
from shapely.geometry.base import BaseGeometry

from geo_model.load_data import NTS_BANDS, NTS_MODES

MILE = 1609.344  # metres
# Miles, the lower edge of every NTS trip-length band but the first.
BAND_EDGES = np.array([1.0, 2.0, 5.0])
# Road distance per straight-line metre, applied before a trip is banded. 1 is
# no uplift, so a trip is banded on its straight-line length.
CIRCUITY = 1.0
# The NTS main modes and the route, which a routed student takes with certainty.
MODES = [*NTS_MODES.values(), "route"]


def sample_in_polygon(
    geom: BaseGeometry,
    centre: tuple[float, float],
    size: int,
    rng: np.random.Generator,
    oversample: int = 6,
) -> np.ndarray:
    """Sample points from a bivariate normal, truncated to a polygon.

    Points are drawn i.i.d. per axis from a normal centred on `centre` with
    standard deviation equal to half the larger side of the polygon's bounding
    box, and rejected unless they fall strictly inside `geom`.

    Candidates are drawn and tested in batches rather than one at a time. Only
    ~17% of candidates are accepted for a typical LSOA, so `oversample` batches
    enough candidates to satisfy the request in a single pass.

    Args:
        geom (BaseGeometry): Polygon the points must fall inside.

        centre (tuple[float, float]): Centre of the sampling distribution, in
        the same CRS as `geom`.

        size (int): Number of points to return.

        rng (np.random.Generator): Source of randomness.

        oversample (int, optional): Candidates drawn per point still needed.
        Defaults to 6.

    Returns:
        np.ndarray: Array of shape (size, 2) of accepted coordinates.
    """
    if geom.is_empty or size == 0:
        return np.empty((0, 2))

    shapely.prepare(geom)  # build the GEOS index once, not once per candidate
    xmin, ymin, xmax, ymax = shapely.bounds(geom)
    sd = max((xmax - xmin) / 2, (ymax - ymin) / 2)

    accepted, needed = [], size
    for _ in range(100):
        candidates = rng.normal(size=(max(needed * oversample, 64), 2)) * sd + centre
        inside = candidates[
            shapely.contains_xy(geom, candidates[:, 0], candidates[:, 1])
        ]
        accepted.append(inside)
        needed -= len(inside)
        if needed <= 0:
            return np.vstack(accepted)[:size]

    raise RuntimeError(
        f"Rejection sampling failed to place {size} points in polygon with bounds "
        f"{(xmin, ymin, xmax, ymax)} after 100 batches; {needed} still needed. "
        f"The polygon may be degenerate or `centre` may lie far outside it."
    )


def sample_students(
    borders: pd.Series,
    centroids: pd.Series,
    sizes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample student locations for every area.

    Args:
        borders (pd.Series): Area polygons.

        centroids (pd.Series): Population-weighted centre of each area,
        aligned with `borders`.

        sizes (np.ndarray): Number of students to place in each area, aligned
        with `borders`.

        rng (np.random.Generator): Source of randomness.

    Returns:
        tuple[np.ndarray, np.ndarray]: Coordinates of shape (n_students, 2),
        and the positional index of the area each student was drawn in.
    """
    centre_x = shapely.get_x(np.asarray(centroids))
    centre_y = shapely.get_y(np.asarray(centroids))
    sizes = np.asarray(sizes, dtype=np.int64)

    if not (len(borders) == len(centre_x) == len(sizes)):
        raise ValueError(
            f"borders, centroids and sizes must align: got {len(borders)}, "
            f"{len(centre_x)} and {len(sizes)}."
        )

    chunks = [
        sample_in_polygon(geom, (cx, cy), int(n), rng)
        for geom, cx, cy, n in zip(borders, centre_x, centre_y, sizes)
    ]
    student_xy = np.vstack(chunks) if chunks else np.empty((0, 2))
    area_index = np.repeat(np.arange(len(sizes)), [len(c) for c in chunks])
    return student_xy, area_index


def _spread(values: np.ndarray, name: str) -> float:
    """Standard deviation of `values`, rejecting the degenerate zero case.

    Dividing by a zero spread yields NaN, which `np.argsort` orders arbitrarily
    rather than failing, so the ranking would be silently meaningless.
    """
    sd = float(values.std())
    if sd == 0:
        raise ValueError(f"{name} have zero spread, so cannot be scaled for ranking.")
    return sd


def rank_bundles(
    student_xy: np.ndarray,
    student_district: np.ndarray,
    school_xy: np.ndarray,
    school_district: np.ndarray,
    route_district: np.ndarray,
    route_school: np.ndarray,
    school_scores: np.ndarray | None = None,
    performance_weight: float = 0.0,
    route_discount: float = 0.0,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank (school, route) bundles for students and (student, route) bundles for schools.

    A bundle pairs a school with a route to it, or with no route at all, written
    as route -1. A route serves one district, so a student ranks the bundles of
    the routes leaving their own district alongside the routeless bundle of
    every school; a student whose district has no routes ranks schools alone.

    Student preferences trade travel off against school performance, while
    school priorities are the transpose of the same pairwise distances, so one
    distance matrix serves both.

    Distances and scores are each divided by their own standard deviation before
    being combined, so `performance_weight` is a unit-free share rather than a
    metres-per-score-point rate. Bundle cost is

        (1 - w) * (1 - discount) * distance / sd(distance) - w * score / sd(score)

    ranked ascending, so nearer and higher-scoring schools come first. The
    discount scales the travel term alone rather than the whole cost, which runs
    negative for a school scoring above average and would turn a route into a
    penalty. A bundle therefore never ranks below the same school without one.

    School priorities carry the model's two brackets. The top bracket holds
    every routed bundle to the school and every student living in the school's
    own district, the bottom bracket holds the routeless bundles of everyone
    else, and distance orders both. Every bundle takes its own rank, since the
    mechanism compares ranks across the whole route axis when it evicts.

    Args:
        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        student_district (np.ndarray): Positional index of the district each
        student lives in, shape (n_students,).

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2). Must
        share a CRS with `student_xy`.

        school_district (np.ndarray): Positional index of the district each
        school sits in, aligned with `school_xy` and indexed the same way as
        `student_district`.

        route_district (np.ndarray): District each route leaves, shape
        (n_routes,). A route is identified by its position in this array, the
        route axis the matching mechanism indexes.

        route_school (np.ndarray): School each route arrives at, as a positional
        index into `school_xy`, aligned with `route_district`.

        school_scores (np.ndarray | None, optional): Performance score per
        school, aligned with `school_xy`, higher being better. Required when
        `performance_weight` > 0. Defaults to None.

        performance_weight (float, optional): Share of the preference ranking
        driven by performance rather than travel, in [0, 1]. 0 gives
        nearest-first ordering, 1 ranks on performance alone. School priorities
        are unaffected either way. Defaults to 0.0.

        route_discount (float, optional): Share of the travel a route takes out
        of the ranking of the school it serves, in [0, 1]. 0 leaves a bundle
        tied with the same school without a route, 1 ranks it as if the school
        were next door. School priorities are unaffected. Defaults to 0.0.

        noise_scale (float, optional): Standard deviation of Gaussian noise
        added to the combined preference cost, measured in units of that cost
        rather than in metres. Useful to break ties or add mild randomness
        without destroying the underlying signal. Set to 0 for a deterministic
        ordering. School priorities are always ranked on the unperturbed
        distances. Defaults to 0.0.

        rng (np.random.Generator | None, optional): Source of randomness, used
        only when `noise_scale` > 0. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Student preferences of shape
        (n_students, max_options, 2) holding (school, route) pairs best-first
        and right-padded with (-1, -1), and school priorities of shape
        (n_schools, n_routes + 1, n_students) holding the rank of every
        (student, route) bundle. Routeless bundles sit last on the route axis,
        where the mechanism's index of -1 lands.
    """
    if not 0.0 <= performance_weight <= 1.0:
        raise ValueError(
            f"performance_weight must lie in [0, 1], got {performance_weight}."
        )
    if not 0.0 <= route_discount <= 1.0:
        raise ValueError(f"route_discount must lie in [0, 1], got {route_discount}.")

    n_students = len(student_xy)
    n_schools = len(school_xy)
    student_district = np.asarray(student_district, dtype=np.int64)
    school_district = np.asarray(school_district, dtype=np.int64)
    route_district = np.asarray(route_district, dtype=np.int64)
    route_school = np.asarray(route_school, dtype=np.int64)
    n_routes = len(route_district)

    if student_district.shape != (n_students,):
        raise ValueError(
            f"student_district must hold one district per student: expected "
            f"shape {(n_students,)}, got {student_district.shape}."
        )
    if school_district.shape != (n_schools,):
        raise ValueError(
            f"school_district must hold one district per school: expected shape "
            f"{(n_schools,)}, got {school_district.shape}."
        )
    if route_school.shape != route_district.shape:
        raise ValueError(
            f"route_district and route_school must align: got "
            f"{route_district.shape} and {route_school.shape}."
        )
    if n_routes and not ((0 <= route_school) & (route_school < n_schools)).all():
        raise ValueError(
            f"route_school holds indices outside the {n_schools} schools given."
        )

    distances = spdist.cdist(student_xy, school_xy)
    travel = (1 - performance_weight) * distances / _spread(distances, "Distances")
    merit = np.zeros(n_schools)

    if performance_weight > 0:
        if school_scores is None:
            raise ValueError("school_scores is required when performance_weight > 0.")
        scores = np.asarray(school_scores, dtype=float)
        if scores.shape != (n_schools,):
            raise ValueError(
                f"school_scores must hold one score per school: expected shape "
                f"{(n_schools,)}, got {scores.shape}."
            )
        if not np.isfinite(scores).all():
            raise ValueError("school_scores holds non-finite values.")
        merit = -performance_weight * scores / _spread(scores, "School scores")

    noise_rng = (rng or np.random.default_rng()) if noise_scale > 0 else None

    # Every student in a district is offered that district's routes, so option
    # sets are built once per district rather than once per student.
    routes_per_district = np.bincount(route_district) if n_routes else np.zeros(1)
    max_options = n_schools + int(routes_per_district.max())
    preferences = np.full((n_students, max_options, 2), -1, dtype=np.int32)
    routeless = np.column_stack((np.arange(n_schools), np.full(n_schools, -1)))

    for district in np.unique(student_district):
        students = np.flatnonzero(student_district == district)
        routes = np.flatnonzero(route_district == district)
        schools = route_school[routes]

        options = np.vstack((routeless, np.column_stack((schools, routes))))
        cost = np.hstack(
            (
                travel[students] + merit,
                (1 - route_discount) * travel[np.ix_(students, schools)]
                + merit[schools],
            )
        )
        if noise_rng is not None:
            cost = cost + noise_rng.normal(0, noise_scale, size=cost.shape)

        # Stable, so a bundle left tied with its own school by a zero discount
        # ranks below it and a seat is only taken when the route earns it.
        order = np.argsort(cost, axis=1, kind="stable")
        preferences[students, : len(options)] = options[order]

    priorities = np.empty((n_schools, n_routes + 1, n_students), dtype=np.int32)
    for school in range(n_schools):
        bundle_student = [np.arange(n_students)]
        bundle_route = [np.full(n_students, n_routes)]
        bracket = [np.where(student_district == school_district[school], 0, 1)]

        for route in np.flatnonzero(route_school == school):
            riders = np.flatnonzero(student_district == route_district[route])
            bundle_student.append(riders)
            bundle_route.append(np.full(len(riders), route))
            bracket.append(np.zeros(len(riders), dtype=np.int64))

        bundle_student = np.concatenate(bundle_student)
        bundle_route = np.concatenate(bundle_route)
        order = np.lexsort((distances[bundle_student, school], np.concatenate(bracket)))
        rank = np.empty(len(order), dtype=np.int32)
        rank[order] = np.arange(len(order), dtype=np.int32)

        # Bundles on another school's routes are never proposed here, so their
        # cells take a rank worse than any real one rather than being left to
        # masquerade as a strong claim on a seat.
        priorities[school] = len(order)
        priorities[school, bundle_route, bundle_student] = rank

    return preferences, priorities


def route_eligibility(
    student_xy: np.ndarray,
    school_xy: np.ndarray,
    routeless_matching: np.ndarray,
    walk_distance: float,
    rule: str,
) -> np.ndarray:
    """Whether each student may be offered a route, by a walking threshold.

    A route carries a student to a school they could not walk to, so a student
    who can walk is offered none. Two rules say what "can walk" means:

    - "routeless": the student holds a seat within `walk_distance` in the
      matching without routes. A student that matching leaves unseated can
      walk nowhere, so is always eligible.
    - "nearest": a school lies within `walk_distance` of the student's home,
      whatever seat they would hold. Exogenous, unlike the first rule.

    Args:
        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2), in
        the same CRS as `student_xy`.

        routeless_matching (np.ndarray): The matching of the same students
        without routes, as returned by `fast_DAT`. Read by "routeless" alone.

        walk_distance (float): Straight-line metres a student is taken to
        walk. A student exactly this far is not eligible.

        rule (str): "routeless" or "nearest".

    Returns:
        np.ndarray: Boolean, one per student.
    """
    if rule == "routeless":
        school = routeless_matching[:, 0]
        seated = school >= 0
        distance = np.full(len(school), np.inf)
        distance[seated] = np.linalg.norm(
            student_xy[seated] - school_xy[school[seated]], axis=1
        )
    elif rule == "nearest":
        distance = spdist.cdist(student_xy, school_xy).min(axis=1)
    else:
        raise ValueError(f"rule must be 'routeless' or 'nearest', got {rule!r}.")
    return distance > walk_distance


def withhold_routes(preferences: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Take every routed bundle out of the lists of ineligible students.

    The bundles are overwritten with the (-1, -1) padding, which `fast_DAT`
    passes over wherever it sits in a list, so the mechanism is unchanged.
    Priorities need no change either: a bundle never proposed is never held.

    Args:
        preferences (np.ndarray): Student preferences, shape (n_students,
        max_options, 2), as returned by `rank_bundles`.

        eligible (np.ndarray): Whether each student keeps their routed
        bundles, shape (n_students,).

    Returns:
        np.ndarray: A copy of `preferences` with the bundles withheld.
    """
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (len(preferences),):
        raise ValueError(
            f"eligible must hold one flag per student: expected shape "
            f"{(len(preferences),)}, got {eligible.shape}."
        )
    withheld = preferences.copy()
    withheld[~eligible[:, None] & (preferences[:, :, 1] >= 0)] = -1
    return withheld


def reserve_routes(
    preferences: np.ndarray,
    priorities: np.ndarray,
    school_capacities: np.ndarray,
    route_school: np.ndarray,
    route_capacities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recast an SCT instance so every route holds seats of its own.

    In the instance `rank_bundles` builds a routed bundle competes for the
    school's seats and can evict a routeless student. Here each route becomes
    a slot pool at its school, numbered `n_schools + route`, holding the
    route's seats on top of the school's: a routed applicant competes only
    with the route's other riders and a routeless applicant only with other
    routeless applicants. The result is a routeless instance for `fast_DAT`,
    whose matching `unreserve` maps back to (school, route).

    Args:
        preferences (np.ndarray): Student preferences, shape (n_students,
        max_options, 2), as returned by `rank_bundles`.

        priorities (np.ndarray): School priorities, shape (n_schools,
        n_routes + 1, n_students), as returned by `rank_bundles`.

        school_capacities (np.ndarray): Seats at each school, shape
        (n_schools,).

        route_school (np.ndarray): School each route arrives at, shape
        (n_routes,).

        route_capacities (np.ndarray): Seats on each route, aligned with
        `route_school`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Preferences over the
        pools with every route index -1, pool priorities of shape
        (n_schools + n_routes, 1, n_students), and pool capacities.
    """
    n_schools, n_routes = priorities.shape[0], priorities.shape[1] - 1
    route_school = np.asarray(route_school, dtype=np.int64)
    if route_school.shape != (n_routes,) or len(route_capacities) != n_routes:
        raise ValueError(
            f"priorities hold {n_routes} routes, route_school {route_school.shape} "
            f"and route_capacities {len(route_capacities)}."
        )

    pooled = preferences.copy()
    routed = pooled[:, :, 1] >= 0
    pooled[routed, 0] = n_schools + pooled[routed, 1]
    pooled[:, :, 1] = -1

    pool_priorities = np.concatenate(
        (
            priorities[:, n_routes, :],
            priorities[route_school, np.arange(n_routes), :],
        )
    )[:, None, :]
    pool_capacities = np.concatenate((school_capacities, route_capacities))
    return pooled, pool_priorities, pool_capacities.astype(np.int32)


def unreserve(
    matching: np.ndarray, n_schools: int, route_school: np.ndarray
) -> np.ndarray:
    """Map a matching over the pools of `reserve_routes` back to (school, route).

    Args:
        matching (np.ndarray): As returned by `fast_DAT` on the reserved
        instance, every route index -1.

        n_schools (int): Number of schools ahead of the route pools.

        route_school (np.ndarray): School each route arrives at.

    Returns:
        np.ndarray: (school, route) per student, -1 when absent, in the form
        `fast_DAT` returns on the original instance.
    """
    route_school = np.asarray(route_school, dtype=np.int64)
    pool = matching[:, 0]
    if (pool >= n_schools + len(route_school)).any():
        raise ValueError(
            f"matching holds pools beyond the {n_schools} schools and "
            f"{len(route_school)} routes."
        )
    on_route = pool >= n_schools
    unreserved = matching.copy()
    unreserved[on_route, 1] = pool[on_route] - n_schools
    unreserved[on_route, 0] = route_school[unreserved[on_route, 1]]
    return unreserved


def district_index(schools: pd.DataFrame, areas: pd.DataFrame) -> np.ndarray:
    """Positional index into `areas` of the district each school sits in.

    Students carry the positional index of the area they were sampled in, so a
    school's district has to be expressed the same way before the two can be
    compared for the local priority bracket.

    A school run by the authority can stand just over the boundary in a
    neighbouring one, in a district the model does not hold. Such a school takes
    -1, an index no student carries, so nobody holds local priority there.

    Args:
        schools (pd.DataFrame): Schools carrying "LSOA21CD" and
        "EstablishmentName" columns.

        areas (pd.DataFrame): Districts carrying an "LSOA21CD" column, in the
        order students were sampled from.

    Returns:
        np.ndarray: Positional index into `areas`, one per school, or -1 for a
        school sitting outside every area.
    """
    lookup = pd.Series(np.arange(len(areas)), index=areas["LSOA21CD"])
    index = lookup.reindex(schools["LSOA21CD"])

    outside = index.isna().to_numpy()
    if outside.any():
        print(
            "Sits in a district the model does not hold, so no student holds "
            "local priority there: "
            + ", ".join(schools.loc[outside, "EstablishmentName"])
        )
    return index.fillna(-1).to_numpy(dtype=np.int64)


def cohort_capacity(schools: pd.DataFrame) -> np.ndarray:
    """Seats one year group holds at each school.

    The register publishes capacity across every year group a school teaches,
    while the matching admits a single cohort, so capacity is spread evenly over
    the years the school spans. A sixth form is smaller than the year groups
    below it, so this understates the intake of an 11-18 school; the register
    publishes no admission number to use in its place.

    Args:
        schools (pd.DataFrame): Schools carrying "EstablishmentName",
        "SchoolCapacity", "StatutoryLowAge" and "StatutoryHighAge" columns.

    Returns:
        np.ndarray: Seats for one cohort at each school.
    """
    unsized = (
        schools[["SchoolCapacity", "StatutoryLowAge", "StatutoryHighAge"]]
        .isna()
        .any(axis=1)
    )
    if unsized.any():
        raise ValueError(
            "No capacity or no age range published, so a cohort cannot be "
            "sized: " + ", ".join(schools.loc[unsized.to_numpy(), "EstablishmentName"])
        )

    year_groups = schools["StatutoryHighAge"] - schools["StatutoryLowAge"]
    if (year_groups <= 0).any():
        raise ValueError(
            "These schools publish an age range covering no year group, so "
            "their capacity cannot be split into cohorts: "
            + ", ".join(schools.loc[(year_groups <= 0).to_numpy(), "EstablishmentName"])
        )
    capacity = np.rint((schools["SchoolCapacity"] / year_groups).to_numpy())

    if (capacity < 1).any():
        raise ValueError(
            "These schools hold fewer than one seat per cohort, so the age "
            "range or the capacity is wrong: "
            + ", ".join(schools.loc[capacity < 1, "EstablishmentName"])
        )
    return capacity.astype(np.int32)


def school_intake(
    matched_school: ArrayLike,
    disadvantaged: ArrayLike,
    n_schools: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count the disadvantaged and the other students seated at each school.

    A student without a seat belongs to no school's intake, so unmatched
    students are left out of both counts.

    Args:
        matched_school (ArrayLike): School each student is matched to, shape
        (n_students,), -1 for an unmatched student. The first column of the
        matching `fast_DAT` returns.

        disadvantaged (ArrayLike): Whether each student lives in a
        disadvantaged district, shape (n_students,).

        n_schools (int): Number of schools the matching indexes.

    Returns:
        tuple[np.ndarray, np.ndarray]: Disadvantaged and other students
        seated at each school, each of shape (n_schools,).
    """
    matched_school = np.asarray(matched_school, dtype=np.int64)
    disadvantaged = np.asarray(disadvantaged, dtype=bool)
    if matched_school.shape != disadvantaged.shape:
        raise ValueError(
            f"matched_school and disadvantaged must align: got "
            f"{matched_school.shape} and {disadvantaged.shape}."
        )
    if (matched_school >= n_schools).any():
        raise ValueError(
            f"matched_school holds indices beyond the {n_schools} schools."
        )

    seated = matched_school >= 0
    return (
        np.bincount(matched_school[seated & disadvantaged], minlength=n_schools),
        np.bincount(matched_school[seated & ~disadvantaged], minlength=n_schools),
    )


def dissimilarity_index(
    matched_school: ArrayLike,
    disadvantaged: ArrayLike,
    n_schools: int,
) -> float:
    """Dissimilarity index of a matching, disadvantaged students against the rest.

    With a_c disadvantaged and b_c other students seated at school c, out of
    A and B seated in all, the index is

        D = 1/2 * sum_c | a_c / A - b_c / B |

    the share of either group that would have to change school for every
    school to hold the city-wide mix. 0 is that mix everywhere, 1 is complete
    segregation. The counts are those of `school_intake`, so unmatched
    students are left out of both groups.

    Args:
        matched_school (ArrayLike): School each student is matched to, shape
        (n_students,), -1 for an unmatched student. The first column of the
        matching `fast_DAT` returns.

        disadvantaged (ArrayLike): Whether each student lives in a
        disadvantaged district, shape (n_students,).

        n_schools (int): Number of schools the matching indexes.

    Returns:
        float: The index, in [0, 1].
    """
    group_a, group_b = school_intake(matched_school, disadvantaged, n_schools)
    if group_a.sum() == 0 or group_b.sum() == 0:
        raise ValueError(
            f"{group_a.sum()} disadvantaged and {group_b.sum()} other students hold "
            "a seat, so the index is undefined."
        )
    return 0.5 * float(np.abs(group_a / group_a.sum() - group_b / group_b.sum()).sum())


def trip_band(distance_m: ArrayLike, circuity: float = CIRCUITY) -> np.ndarray:
    """The NTS trip-length band of each trip.

    Args:
        distance_m (ArrayLike): Straight-line length of each trip, in metres.

        circuity (float, optional): Road distance per straight-line metre.
        Defaults to CIRCUITY.

    Returns:
        np.ndarray: The band of each trip, one of NTS_BANDS.
    """
    if circuity <= 0:
        raise ValueError(f"circuity must be positive, got {circuity}.")
    miles = np.asarray(distance_m, dtype=float) * circuity / MILE
    return np.array(NTS_BANDS)[np.digitize(miles, BAND_EDGES)]


def expected_modes(
    matching: np.ndarray,
    student_xy: np.ndarray,
    school_xy: np.ndarray,
    shares: pd.DataFrame,
    circuity: float = CIRCUITY,
) -> pd.Series:
    """Expected students per travel mode under a matching.

    A student seated without a route travels by the NTS mode split for school
    trips of their length, held as a probability vector rather than sampled,
    so the count is an expected value. A student seated with a route takes it
    with certainty. An unseated student makes no trip and is left out.

    Args:
        matching (np.ndarray): (school, route) per student, -1 when absent, as
        returned by `fast_DAT`.

        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2), in
        the same CRS as `student_xy`.

        shares (pd.DataFrame): Mode share within each band, as returned by
        `load_nts_mode_shares`.

        circuity (float, optional): Passed to `trip_band`. Defaults to
        CIRCUITY.

    Returns:
        pd.Series: Expected students per mode, indexed by MODES, summing to
        the number seated.
    """
    seated = matching[:, 0] >= 0
    school, route = matching[seated, 0], matching[seated, 1]
    distance = np.linalg.norm(student_xy[seated] - school_xy[school], axis=1)
    band = trip_band(distance, circuity)

    modes = shares.loc[band[route < 0]].sum()
    modes["route"] = int((route >= 0).sum())
    return modes.reindex(MODES)


def mode_change(results: pd.DataFrame) -> pd.DataFrame:
    """The change in expected students per mode the routes induce, per seed.

    Args:
        results (pd.DataFrame): As returned by `run`, or any frame carrying
        "seed", "scenario", "mode" and "students" plus other key columns.

    Returns:
        pd.DataFrame: One row per (seed, mode) and any other key columns,
        with "change" as the students with routes minus those without.
    """
    keys = [c for c in results.columns if c not in ("scenario", "students")]
    change = results.pivot(index=keys, columns="scenario", values="students")
    change = change["with routes"] - change["without routes"]
    return change.rename("change").reset_index()
