import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import distance as spdist

from geo_model.model.settings import (
    DISADVANTAGE,
    DISADVANTAGED_PERFORMANCE_WEIGHT,
    DISADVANTAGED_ROUTE_DISCOUNT,
    PERFORMANCE_WEIGHT,
    PREFERENCE_NOISE,
    PRIORITY,
    PRIORITY_CHOICES,
    PRIORITY_DISCOUNT,
    ROUTE_DISCOUNT,
)


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
    performance_weight: ArrayLike = 0.0,
    route_discount: ArrayLike = 0.0,
    route_eligible: ArrayLike = True,
    priority: str = PRIORITY,
    priority_discount: float = PRIORITY_DISCOUNT,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank (school, route) bundles for students and (student, route) bundles for schools.

    A bundle pairs a school with a route to it, or with no route at all, written
    as route -1. A route serves one district, so a student eligible for routes
    ranks the bundles of the routes leaving their own district alongside the
    routeless bundle of every school; a student who is not eligible, or whose
    district has no routes, ranks schools alone.

    Student preferences trade travel off against school performance, while
    school priorities are the transpose of the same pairwise distances, so one
    distance matrix serves both.

    Distances and scores are each divided by their own standard deviation before
    being combined, so `performance_weight` is a unit-free share rather than a
    metres-per-score-point rate. The cost of a bundle to student i is

        (1 - w_i) * (1 - d_i) * distance / sd(distance) - w_i * score / sd(score)

    with w_i and d_i the student's own performance weight and route discount,
    ranked ascending, so nearer and higher-scoring schools come first. The
    discount scales the travel term alone rather than the whole cost, which runs
    negative for a school scoring above average and would turn a route into a
    penalty. A bundle therefore never ranks below the same school without one.

    School priorities follow `priority`. Under "distance" a school orders every
    bundle by the student's distance, a routed bundle's taken down by
    `priority_discount`, a tie going to the nearer on the true distance. Under
    "bracket" they carry the model's two brackets: the top holds every routed
    bundle to the school, those of eligible students alone, and every student
    living in the school's own district, the bottom the routeless bundles of
    everyone else, and distance orders both. A tie either rule leaves goes to
    the routeless bundle, then to the lower route and student index.
    Every bundle takes its own rank, since the mechanism compares ranks across
    the whole route axis when it evicts. The mechanism only ever reads the
    rank of the bundle a student proposes, so each rank is returned beside
    the bundle on the student's list rather than in an array over every
    school, route and student, which would grow with all three at once.

    Args:
        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        student_district (np.ndarray): Positional index of the district each
        student lives in, shape (n_students,).

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2). Must
        share a CRS with `student_xy`.

        school_district (np.ndarray): Positional index of the district each
        school sits in, aligned with `school_xy` and indexed the same way as
        `student_district`. Read by the "bracket" priority alone.

        route_district (np.ndarray): District each route leaves, shape
        (n_routes,). A route is identified by its position in this array, the
        route axis the matching mechanism indexes.

        route_school (np.ndarray): School each route arrives at, as a positional
        index into `school_xy`, aligned with `route_district`.

        school_scores (np.ndarray | None, optional): Performance score per
        school, aligned with `school_xy`, higher being better. Required when
        `performance_weight` > 0. Defaults to None.

        performance_weight (ArrayLike, optional): Share of the preference
        ranking driven by performance rather than travel, in [0, 1], one for
        every student or one per student. 0 gives nearest-first ordering, 1
        ranks on performance alone. School priorities are unaffected either
        way. Defaults to 0.0.

        route_discount (ArrayLike, optional): Share of the travel a route
        takes out of the ranking of the school it serves, in [0, 1], one for
        every student or one per student. 0 leaves a bundle tied with the
        same school without a route, 1 ranks it as if the school were next
        door. School priorities are unaffected. Defaults to 0.0.

        route_eligible (ArrayLike, optional): Whether a student may take the
        routes leaving their district, one for every student or one per
        student. Defaults to True.

        priority (str, optional): One of PRIORITY_CHOICES, the rule school
        priorities follow. Defaults to PRIORITY.

        priority_discount (float, optional): Share of a routed student's
        distance a school's priority takes off under "distance", in [0, 1].
        0 ranks a routed bundle on its true distance, 1 as if the student
        lived at the school. Student preferences are unaffected. Defaults to
        PRIORITY_DISCOUNT.

        noise_scale (float, optional): Standard deviation of Gaussian noise
        added to the combined preference cost, measured in units of that cost
        rather than in metres. One draw is taken per student and school, a
        taste the student has for the school, and added to every bundle of
        that school, so a route still differs from its school's routeless
        bundle by travel alone. The draw does not depend on the routes, so
        the same `rng` state gives the same tastes with routes and without.
        Set to 0 for a deterministic ordering. School priorities are always
        ranked on the unperturbed distances. Defaults to 0.0.

        rng (np.random.Generator | None, optional): Source of randomness,
        required when `noise_scale` > 0, so the noise is reproducible.
        Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Student preferences of shape
        (n_students, max_options, 2) holding (school, route) pairs best-first
        and right-padded with (-1, -1), and preference ranks of shape
        (n_students, max_options) holding the rank the school of each of
        those bundles gives it, 0 being its strongest claim on a seat, padded
        with -1.
    """
    n_students = len(student_xy)
    n_schools = len(school_xy)
    # A scalar is every student's value; an array must hold one per student.
    performance_weight = np.broadcast_to(
        np.asarray(performance_weight, dtype=float), (n_students,)
    )
    route_discount = np.broadcast_to(
        np.asarray(route_discount, dtype=float), (n_students,)
    )
    route_eligible = np.broadcast_to(
        np.asarray(route_eligible, dtype=bool), (n_students,)
    )
    if not ((0.0 <= performance_weight) & (performance_weight <= 1.0)).all():
        raise ValueError(
            f"performance_weight must lie in [0, 1], got values from "
            f"{performance_weight.min()} to {performance_weight.max()}."
        )
    if not ((0.0 <= route_discount) & (route_discount <= 1.0)).all():
        raise ValueError(
            f"route_discount must lie in [0, 1], got values from "
            f"{route_discount.min()} to {route_discount.max()}."
        )
    if priority not in PRIORITY_CHOICES:
        raise ValueError(
            f"priority must be one of {PRIORITY_CHOICES}, got {priority!r}."
        )
    if not 0.0 <= priority_discount <= 1.0:
        raise ValueError(
            f"priority_discount must lie in [0, 1], got {priority_discount}."
        )

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
    travel = (
        (1 - performance_weight)[:, None] * distances / _spread(distances, "Distances")
    )
    merit = np.zeros((n_students, n_schools))

    if (performance_weight > 0).any():
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
        merit = -performance_weight[:, None] * scores / _spread(scores, "School scores")

    if noise_scale > 0:
        if rng is None:
            raise ValueError(
                "noise_scale > 0 needs an rng, so the noise can be reproduced."
            )
        merit = merit + rng.normal(0, noise_scale, size=(n_students, n_schools))

    # Every school ranks its own bundles, the routeless one of every student
    # and then those of each route arriving at it, so the routeless ranks are
    # kept per school and student and the routed ones per route and rider.
    routeless_rank = np.empty((n_schools, n_students), dtype=np.int32)
    routed_rank = [np.empty(0, dtype=np.int32)] * n_routes
    for school in range(n_schools):
        routes = np.flatnonzero(route_school == school)
        riders = [
            np.flatnonzero((student_district == route_district[route]) & route_eligible)
            for route in routes
        ]
        bundle_student = np.concatenate([np.arange(n_students), *riders])
        distance = distances[bundle_student, school]
        routed = np.arange(len(bundle_student)) >= n_students
        # Each rule sorts on a key of its own, the true distance breaking its
        # ties.
        if priority == "bracket":
            key = np.where(
                routed | (student_district[bundle_student] == school_district[school]),
                0,
                1,
            )
        else:
            key = np.where(routed, 1 - priority_discount, 1.0) * distance
        order = np.lexsort((distance, key))
        rank = np.empty(len(order), dtype=np.int32)
        rank[order] = np.arange(len(order), dtype=np.int32)

        routeless_rank[school] = rank[:n_students]
        ends = np.cumsum([n_students, *map(len, riders)])
        for route, start, stop in zip(routes, ends[:-1], ends[1:]):
            routed_rank[route] = rank[start:stop]

    # Every eligible student in a district is offered that district's routes,
    # and every other student none, so option sets are built once per
    # (district, eligibility) pair rather than once per student.
    routes_per_district = np.bincount(route_district) if n_routes else np.zeros(1)
    max_options = n_schools + int(routes_per_district.max())
    preferences = np.full((n_students, max_options, 2), -1, dtype=np.int32)
    ranks = np.full((n_students, max_options), -1, dtype=np.int32)
    routeless = np.column_stack((np.arange(n_schools), np.full(n_schools, -1)))

    for district, eligible in np.unique(
        np.column_stack((student_district, route_eligible)), axis=0
    ):
        students = np.flatnonzero(
            (student_district == district) & (route_eligible == eligible)
        )
        routes = np.flatnonzero((route_district == district) & bool(eligible))
        schools = route_school[routes]

        options = np.vstack((routeless, np.column_stack((schools, routes))))
        cost = np.hstack(
            (
                travel[students] + merit[students],
                (1 - route_discount[students, None]) * travel[np.ix_(students, schools)]
                + merit[np.ix_(students, schools)],
            )
        )

        # Stable, so a bundle left tied with its own school by a zero discount
        # ranks below it and a seat is only taken when the route earns it.
        order = np.argsort(cost, axis=1, kind="stable")
        preferences[students, : len(options)] = options[order]
        # A route's riders are the eligible students of its district, these
        # students in the same order, so its ranks line up with them as they
        # stand.
        option_rank = np.column_stack(
            (routeless_rank[:, students].T, *(routed_rank[route] for route in routes))
        )
        ranks[students, : len(options)] = np.take_along_axis(option_rank, order, axis=1)

    return preferences, ranks


def district_index(schools: pd.DataFrame, areas: pd.DataFrame) -> np.ndarray:
    """Positional index into `areas` of the district each school sits in.

    Students carry the positional index of the area they were sampled in, so a
    school's district has to be expressed the same way before the two can be
    compared, to find the students local to a school.

    A school run by the authority can stand just over the boundary in a
    neighbouring one, in a district the model does not hold. Such a school takes
    -1, an index no student carries, so no student is local to it.

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
            "Sits in a district the model does not hold, so no student is local "
            "to it: " + ", ".join(schools.loc[outside, "EstablishmentName"])
        )
    return index.fillna(-1).to_numpy(dtype=np.int64)


def cohort_capacity(schools: pd.DataFrame) -> np.ndarray:
    """Seats one year group holds at each school.

    A secondary school is sized on the Year 7 places it offers, as
    `load_places_offered` reads them, Year 7 being the cohort the matching
    fills, so a frame carrying them is sized on them as they stand. A primary
    school is sized on the register's capacity across every year group it
    teaches, spread evenly over the years it spans.

    Args:
        schools (pd.DataFrame): Schools carrying a "PlacesOffered" column, or
        else "EstablishmentName", "SchoolCapacity", "StatutoryLowAge" and
        "StatutoryHighAge" columns.

    Returns:
        np.ndarray: Seats for one cohort at each school.
    """
    if "PlacesOffered" in schools.columns:
        return schools["PlacesOffered"].to_numpy(dtype=np.int32)

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


def secondary_instance(
    student_xy: np.ndarray,
    student_lsoa: np.ndarray,
    areas: pd.DataFrame,
    secondary_schools: gpd.GeoDataFrame,
    routes: pd.DataFrame | None,
    disadvantaged: np.ndarray,
    performance_weight: float = PERFORMANCE_WEIGHT,
    route_discount: float = ROUTE_DISCOUNT,
    disadvantaged_performance_weight: float = DISADVANTAGED_PERFORMANCE_WEIGHT,
    disadvantaged_route_discount: float = DISADVANTAGED_ROUTE_DISCOUNT,
    disadvantage: str = DISADVANTAGE,
    priority: str = PRIORITY,
    priority_discount: float = PRIORITY_DISCOUNT,
    noise_scale: float = PREFERENCE_NOISE,
    noise_seed: int | np.random.SeedSequence | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rank one secondary student sample into an SCT instance.

    Progress 8 weights the preferences, and routes are offered when a route
    set is given. Without one the instance is a plain school choice problem
    on the same students, the baseline the transport scheme is measured
    against. Disadvantaged students rank with their own pair of weights and
    the rest with theirs. The weights default to the model's constants, and
    are taken as arguments so a sweep can vary them without rebinding the
    module.

    Args:
        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        areas (pd.DataFrame): Districts carrying an "LSOA21CD" column, in the
        order students were sampled from.

        secondary_schools (gpd.GeoDataFrame): Schools carrying "LSOA21CD",
        "EstablishmentName", "P8MEA", "Easting", "Northing" and the capacity
        columns `cohort_capacity` reads.

        routes (pd.DataFrame | None): The route set, as returned by
        `route_network`, or None for an instance without routes.

        disadvantaged (np.ndarray): Whether each student is disadvantaged,
        as `disadvantaged_group` gives it.

        performance_weight (float, optional): Share of the preference ranking
        driven by Progress 8 rather than travel, in [0, 1], for students who
        are not disadvantaged. Defaults to PERFORMANCE_WEIGHT.

        route_discount (float, optional): Share of the travel a route takes
        out of the ranking of the school it serves, in [0, 1], for students
        who are not disadvantaged. Defaults to ROUTE_DISCOUNT.

        disadvantaged_performance_weight (float, optional): As
        `performance_weight`, for disadvantaged students. Defaults to
        DISADVANTAGED_PERFORMANCE_WEIGHT.

        disadvantaged_route_discount (float, optional): As `route_discount`,
        for disadvantaged students. Defaults to DISADVANTAGED_ROUTE_DISCOUNT.

        disadvantage (str, optional): One of DISADVANTAGE_CHOICES. Under
        "score" only disadvantaged students may take a route, otherwise every
        student of its district may. Defaults to DISADVANTAGE.

        priority (str, optional): Passed to `rank_bundles`. Defaults to
        PRIORITY.

        priority_discount (float, optional): Passed to `rank_bundles`.
        Defaults to PRIORITY_DISCOUNT.

        noise_scale (float, optional): Passed to `rank_bundles`. Defaults to
        PREFERENCE_NOISE.

        noise_seed (int | np.random.SeedSequence | None, optional): Seed the
        tastes are drawn from, afresh on every call, so two instances of a
        sample given the same seed rank with the same tastes. Required when
        `noise_scale` > 0. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Student
        preferences, the school's rank of every bundle on them, school
        capacities and route capacities, in the form `fast_DAT` takes. Route capacities are empty
        when there are no routes.
    """
    if routes is None:
        route_district = route_school = route_capacities = np.empty(0, dtype=np.int32)
    else:
        route_district = routes["district_idx"].to_numpy()
        route_school = routes["school_idx"].to_numpy()
        route_capacities = routes["capacity"].to_numpy(dtype=np.int32)

    preferences, ranks = rank_bundles(
        student_xy,
        student_lsoa,
        secondary_schools[["Easting", "Northing"]].to_numpy(),
        district_index(secondary_schools, areas),
        route_district,
        route_school,
        school_scores=secondary_schools["P8MEA"].to_numpy(),
        performance_weight=np.where(
            disadvantaged, disadvantaged_performance_weight, performance_weight
        ),
        route_discount=np.where(
            disadvantaged, disadvantaged_route_discount, route_discount
        ),
        route_eligible=disadvantaged if disadvantage == "score" else True,
        priority=priority,
        priority_discount=priority_discount,
        noise_scale=noise_scale,
        rng=None if noise_seed is None else np.random.default_rng(noise_seed),
    )
    return preferences, ranks, cohort_capacity(secondary_schools), route_capacities
