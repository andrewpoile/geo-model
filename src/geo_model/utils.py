import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import distance as spdist


def sample_in_polygon(
    geom: shapely.Geometry,
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
        geom (shapely.Geometry): Polygon the points must fall inside.

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
    borders: gpd.GeoSeries,
    centroids: gpd.GeoSeries,
    sizes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample student locations for every area.

    Args:
        borders (gpd.GeoSeries): Area polygons.

        centroids (gpd.GeoSeries): Population-weighted centre of each area,
        aligned with `borders`.

        sizes (np.ndarray): Number of students to place in each area, aligned
        with `borders`.

        rng (np.random.Generator): Source of randomness.

    Returns:
        tuple[np.ndarray, np.ndarray]: Coordinates of shape (n_students, 2),
        and the positional index of the area each student was drawn in.
    """
    borders = np.asarray(borders)
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

    if noise_scale > 0:
        rng = rng or np.random.default_rng()

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
        cost = np.hstack((
            travel[students] + merit,
            (1 - route_discount) * travel[np.ix_(students, schools)] + merit[schools],
        ))
        if noise_scale > 0:
            cost = cost + rng.normal(0, noise_scale, size=cost.shape)

        # Stable, so a bundle left tied with its own school by a zero discount
        # ranks below it and a seat is only taken when the route earns it.
        order = np.argsort(cost, axis=1, kind="stable")
        preferences[students, :len(options)] = options[order]

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
        order = np.lexsort(
            (distances[bundle_student, school], np.concatenate(bracket))
        )
        rank = np.empty(len(order), dtype=np.int32)
        rank[order] = np.arange(len(order), dtype=np.int32)

        # Bundles on another school's routes are never proposed here, so their
        # cells take a rank worse than any real one rather than being left to
        # masquerade as a strong claim on a seat.
        priorities[school] = len(order)
        priorities[school, bundle_route, bundle_student] = rank

    return preferences, priorities


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
    unsized = schools[
        ["SchoolCapacity", "StatutoryLowAge", "StatutoryHighAge"]
    ].isna().any(axis=1)
    if unsized.any():
        raise ValueError(
            "No capacity or no age range published, so a cohort cannot be "
            "sized: "
            + ", ".join(schools.loc[unsized.to_numpy(), "EstablishmentName"])
        )

    year_groups = schools["StatutoryHighAge"] - schools["StatutoryLowAge"]
    if (year_groups <= 0).any():
        raise ValueError(
            "These schools publish an age range covering no year group, so "
            "their capacity cannot be split into cohorts: "
            + ", ".join(schools.loc[(year_groups <= 0).to_numpy(), "EstablishmentName"])
        )
    capacity = np.rint(schools["SchoolCapacity"] / year_groups).to_numpy()

    if (capacity < 1).any():
        raise ValueError(
            "These schools hold fewer than one seat per cohort, so the age "
            "range or the capacity is wrong: "
            + ", ".join(schools.loc[capacity < 1, "EstablishmentName"])
        )
    return capacity.astype(np.int32)
