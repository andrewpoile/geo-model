from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import distance as spdist

# Routes are built only from LSOAs at or below this IDACI decile, 1 being the
# most deprived 10% of LSOAs. Under DISADVANTAGE "decile" it also sets which
# students are disadvantaged.
DISADVANTAGED_DECILE = 3

# Who is disadvantaged, and who may ride a route leaving their district:
#   "score"       Each LSOA's students are drawn disadvantaged in proportion to
#                 its IDACI score, and only they ride.
#   "score-open"  Drawn the same way, but every student of a route's district
#                 rides, as the SCT model's district-level routes.
#   "decile"      Every student of an LSOA at or below the decile is
#                 disadvantaged and rides: the model as it was before the
#                 scores were drawn from.
# Under every choice route seats are sized on the disadvantaged students.
DISADVANTAGE_CHOICES = ("score", "score-open", "decile")
DISADVANTAGE = "score"

# Metres. A route carries a student to a school they cannot reach unaided, so
# nearer schools need none. 3218m is two miles, the lower bound of the extended
# rights to free home-to-school travel held by low-income 11-16 year olds.
MIN_ROUTE_DISTANCE = 3218

# Seats on the routes into a school, as a multiple of the disadvantaged
# students' fair share of the Year 7 places it offers: k * places * D / N, with
# D the region's disadvantaged students and N its Year 7 cohort. At 1 a
# school's routes hold the seats its disadvantaged students would take if
# every intake matched the region's mix. The seats are split between the
# school's routes in proportion to the disadvantaged students of each route's
# district. Route capacities are exogenous to the SCT instance.
ROUTE_CAPACITY_SCALE = 5.0

# How far a school's route seats lean towards the more deprived districts,
# p >= 0. A district at IDACI decile D at or below the threshold T is weighted
# f(D)^p, f running from 1 at decile 1 to 1 / T at decile T, so 0 splits a
# school's seats on its routes' disadvantaged students alone and p weights
# decile 1 T^p times decile T. The weights only split each school's seats
# between its routes, so they move seats between districts without changing
# how many the school holds.
ROUTE_PROGRESSIVITY = 1.0

# The profile f of the progressive weight: 1 / D, or with True the linear
# (T + 1 - D) / T. The two agree at deciles 1 and T and differ between them.
LINEAR_PROGRESSIVITY = False

# A school's route seats are rounded to the nearest whole seat, then split
# between its routes by largest remainder, so they sum to it exactly, and a
# route apportioned none is not built. True rounds each school's seats up
# instead, so a school with a route keeps at least one seat on it.
ROUND_UP_SEATS = False

# True keeps, for each school, only its route to the nearest district holding
# a seat on one, so every school offers a single route.
SHORTEST_ROUTE_ONLY = False

# A route carries a student past their local schools, so a district already
# within reach of a school performing above this needs none. P8MEA is centred
# on the England average, so 0 is an average school.
MAX_LOCAL_P8 = 0.0

# Metres. 1609m is one mile, read the other way round from MIN_ROUTE_DISTANCE:
# the distance a district is taken to be served over rather than the distance
# it must be carried. The two are independent, so they are swept apart.
LOCAL_RADIUS = 1609

# Written to each region's folder.
ROUTES_CSV = "secondary_routes.csv"
ROUTES_NPZ = "secondary_routes.npz"


class EmptyRouteSet(ValueError):
    """No route survives the conditions: an outcome of the settings, not bad input."""


def centroid_xy(districts: gpd.GeoDataFrame) -> np.ndarray:
    """Read the population centroid of every district as coordinates.

    Args:
        districts (gpd.GeoDataFrame): Districts carrying a "Centroids" geometry.

    Returns:
        np.ndarray: Coordinates of shape (n_districts, 2), in the CRS of the
        centroids.
    """
    centroids = np.asarray(districts["Centroids"])
    return np.column_stack((shapely.get_x(centroids), shapely.get_y(centroids)))


def disadvantaged_cohort(areas: pd.DataFrame, sizes: np.ndarray) -> np.ndarray:
    """Disadvantaged students in each area's cohort, to the nearest student.

    An area's IDACI score is the share of its children, aged 0 to 15, living
    in income-deprived families, so it is taken as the share of the cohort
    sampled there who are disadvantaged.

    Args:
        areas (pd.DataFrame): Areas carrying an "IDACI Score" column.

        sizes (np.ndarray): Students in each area's cohort, aligned with
        `areas`.

    Returns:
        np.ndarray: Disadvantaged students in each area, aligned with `areas`.
    """
    return np.rint(areas["IDACI Score"].to_numpy() * sizes).astype(np.int64)


def apportion(quotas: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Round each column of `quotas` to whole seats summing to its total.

    By largest remainder: every entry takes the floor of its quota, and the
    seats a column still lacks go one each to its largest remainders, the
    first row winning a tie.

    Args:
        quotas (np.ndarray): Non-negative quotas, shape (n_rows, n_columns),
        each column summing to its total.

        totals (np.ndarray): Whole seats in each column, shape (n_columns,).

    Returns:
        np.ndarray: Whole seats, shape (n_rows, n_columns), each column
        summing to its total.
    """
    seats = np.floor(quotas)
    order = np.argsort(seats - quotas, axis=0, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(len(quotas))[:, None], axis=0)
    return (seats + (rank < totals - seats.sum(axis=0))).astype(np.int32)


def build_routes(
    districts: gpd.GeoDataFrame,
    school_xy: np.ndarray,
    min_distance: float,
    weight: np.ndarray,
    school_seats: np.ndarray,
    shortest_only: bool = False,
) -> pd.DataFrame:
    """Build the routes connecting route-eligible districts to schools.

    A route joins one district to one school, so the route set is the subset of
    (district, school) pairs lying more than `min_distance` apart, the district
    carrying some weight, and holding at least one seat. Distance is measured
    from the population centroid of the district, the point its students are
    sampled around, to the school. With `shortest_only` each school keeps only
    the shortest of its routes, the first district winning a tie. A school's
    seats are split between its routes in proportion to their districts'
    weight, by `apportion`, so they sum to its seats exactly; a route
    apportioned none is not built.

    Args:
        districts (gpd.GeoDataFrame): Route-eligible districts, carrying an
        "LSOA21CD" column and a "Centroids" geometry. The index of the frame is
        kept as `district_idx`, so it must be positional into the frame students
        were sampled from.

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2), in the
        same CRS as the centroids. `school_idx` is positional into this array.

        min_distance (float): Districts are routed only to schools further away
        than this, in the units of the CRS.

        weight (np.ndarray): Each district's claim on the seats of the schools
        it is routed to, shape (n_districts,). A district of no weight is not
        routed.

        school_seats (np.ndarray): Whole seats on the routes into each school,
        shape (n_schools,).

        shortest_only (bool, optional): Keep only each school's shortest
        route. Defaults to False.

    Returns:
        pd.DataFrame: One row per route, with columns "route_id" (contiguous
        from 0, the route axis `fast_DAT` indexes), "district_idx", "LSOA21CD",
        "school_idx" and "capacity".
    """
    distances = spdist.cdist(centroid_xy(districts), school_xy)
    routed = (distances > min_distance) & (np.asarray(weight) > 0)[:, None]
    if shortest_only:
        # A school with no route has every distance masked, and stays unrouted.
        nearest = np.where(routed, distances, np.inf).argmin(axis=0)
        routed &= np.arange(len(districts))[:, None] == nearest

    # A school no district is routed to has no route to hold its seats.
    share = np.where(routed, np.asarray(weight, dtype=float)[:, None], 0.0)
    total = share.sum(axis=0)
    held = total > 0
    quotas = np.divide(
        share * school_seats, total, out=np.zeros_like(share), where=held
    )
    capacity = apportion(quotas, np.where(held, school_seats, 0))
    if (routed & (capacity == 0)).any():
        print(
            f"{(routed & (capacity == 0)).sum()} of the {routed.sum()} routes "
            f"beyond {min_distance}m would hold no seat, so are not built."
        )
    routed &= capacity > 0
    district_pos, school_idx = np.nonzero(routed)

    return pd.DataFrame(
        {
            "route_id": np.arange(len(district_pos), dtype=np.int32),
            "district_idx": districts.index.to_numpy()[district_pos].astype(np.int32),
            "LSOA21CD": districts["LSOA21CD"].to_numpy()[district_pos],
            "school_idx": school_idx.astype(np.int32),
            "capacity": capacity[district_pos, school_idx].astype(np.int32),
        }
    )


def route_network(
    areas: gpd.GeoDataFrame,
    school_xy: np.ndarray,
    school_scores: np.ndarray,
    school_places: np.ndarray,
    district_cohort: np.ndarray,
    decile: int = DISADVANTAGED_DECILE,
    min_distance: float = MIN_ROUTE_DISTANCE,
    capacity_scale: float = ROUTE_CAPACITY_SCALE,
    max_local_p8: float = MAX_LOCAL_P8,
    local_radius: float = LOCAL_RADIUS,
    round_up: bool = ROUND_UP_SEATS,
    progressivity: float = ROUTE_PROGRESSIVITY,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
    shortest_only: bool = SHORTEST_ROUTE_ONLY,
) -> pd.DataFrame:
    """Select the route-eligible districts of `areas` and route them to schools.

    A district is route-eligible on two conditions: it sits at or below `decile`,
    and no school scoring above `max_local_p8` lies within `local_radius` of it,
    since a district a high-performing school already serves needs no transport
    past it. The local schools are read from the same population centroid the
    distance condition measures from.

    The disadvantaged group is untouched by the performance condition, so the
    dissimilarity index measures the same students whether their district is
    route-eligible or not. Under every `disadvantage` but "decile" it is drawn
    from IDACI scores, so the decile does not move it either.

    The routes into school s hold k * P_s * D / N seats between them, with P_s
    the Year 7 places the school offers, D the region's disadvantaged students
    and N its cohort, so at k = 1 they hold the seats the region's
    disadvantaged students would take at the school if every intake matched
    the region's mix. Those seats are split between the school's routes in
    proportion to n_d * w_d, with n_d the disadvantaged students of the route's
    district: its `disadvantaged_cohort` under "score" and "score-open", and
    its whole cohort under "decile", where every student of a district at or
    below the decile is disadvantaged. The weight w_d is (1 / D_d)^p for a
    district at IDACI decile D_d, or ((decile + 1 - D_d) / decile)^p when
    `linear`: p moves a school's seats towards the more deprived districts
    without changing how many it holds.

    Args:
        areas (gpd.GeoDataFrame): Every district, carrying "LSOA21CD", "IDACI
        Decile", "IDACI Score" and "Centroids" columns, positionally indexed
        in the order students were sampled from.

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2), in the
        same CRS as the centroids.

        school_scores (np.ndarray): Progress 8 per school, shape (n_schools,),
        aligned with `school_xy`. `load_schools` drops a secondary school
        without a published score, so such a school is absent from both arrays
        and takes no part in this condition.

        school_places (np.ndarray): Year 7 places each school offers, as
        `load_places_offered` reads them, shape (n_schools,), aligned with
        `school_xy`.

        district_cohort (np.ndarray): Students in the cohort of every district,
        positionally aligned with `areas`, as `cohort_sizes` gives it.

        decile (int, optional): Routes are built only from districts at or
        below this IDACI decile. Defaults to DISADVANTAGED_DECILE.

        min_distance (float, optional): Districts are routed only to schools
        further away than this, in metres. Defaults to MIN_ROUTE_DISTANCE.

        capacity_scale (float, optional): k, the seats on the routes into a
        school as a multiple of the disadvantaged students' fair share of its
        places. Defaults to ROUTE_CAPACITY_SCALE.

        max_local_p8 (float, optional): A district with a school scoring above
        this within `local_radius` is not route-eligible. Defaults to
        MAX_LOCAL_P8.

        local_radius (float, optional): Radius around the centroid of a district
        that its local schools are read from, in metres. Defaults to
        LOCAL_RADIUS.

        round_up (bool, optional): Round each school's route seats up rather
        than to the nearest seat, before they are split between its routes.
        Defaults to ROUND_UP_SEATS.

        progressivity (float, optional): p >= 0, how far a school's seats lean
        towards the more deprived districts: 0 is flat, p weights decile 1
        `decile`^p times decile `decile`. Defaults to ROUTE_PROGRESSIVITY.

        linear (bool, optional): Weight by the linear profile
        (decile + 1 - D) / decile rather than by 1 / D. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): One of DISADVANTAGE_CHOICES, which sets
        who is disadvantaged. Defaults to DISADVANTAGE.

        shortest_only (bool, optional): Keep only each school's route to the
        nearest route-eligible district holding a disadvantaged student, which
        then holds all of the school's route seats. Defaults to
        SHORTEST_ROUTE_ONLY.

    Returns:
        pd.DataFrame: The route set, as returned by `build_routes`.
    """
    if not 1 <= decile <= 10:
        raise ValueError(f"decile must be an IDACI decile in [1, 10], got {decile}.")
    if disadvantage not in DISADVANTAGE_CHOICES:
        raise ValueError(
            f"disadvantage must be one of {DISADVANTAGE_CHOICES}, got {disadvantage!r}."
        )

    # A student is tied to their district by position, since sample_students
    # returns a positional area index, so district_idx only means anything if
    # the area frame is positionally indexed too.
    if not areas.index.equals(pd.RangeIndex(len(areas))):
        raise ValueError(
            "The area frame is not positionally indexed, so route district_idx "
            "would not line up with the area index sample_students returns."
        )

    # Misaligned scores would gate on the wrong schools, plausibly and silently.
    if len(school_scores) != len(school_xy):
        raise ValueError(
            f"school_scores must align with school_xy: got {len(school_scores)} "
            f"scores for {len(school_xy)} schools."
        )
    if len(school_places) != len(school_xy):
        raise ValueError(
            f"school_places must align with school_xy: got {len(school_places)} "
            f"places for {len(school_xy)} schools."
        )
    if len(district_cohort) != len(areas):
        raise ValueError(
            f"district_cohort must align with areas: got {len(district_cohort)} "
            f"cohorts for {len(areas)} districts."
        )
    if capacity_scale <= 0:
        raise ValueError(f"capacity_scale must be positive, got {capacity_scale}.")
    if progressivity < 0:
        raise ValueError(f"progressivity must be non-negative, got {progressivity}.")

    deprived = areas[areas["IDACI Decile"] <= decile]
    if deprived.empty:
        raise ValueError(
            f"No LSOA sits at or below IDACI decile {decile}, so there are no "
            "districts to build routes from."
        )

    # A school above the threshold inside the radius serves the district
    # already, so no route is built from it. Carried on the district index, so
    # the eligible subset keeps the positions district_idx is read from.
    local = spdist.cdist(centroid_xy(deprived), school_xy) <= local_radius
    served = pd.DataFrame(
        local & (np.asarray(school_scores) > max_local_p8), index=deprived.index
    ).any(axis=1)
    eligible = deprived[~served]
    if eligible.empty:
        raise EmptyRouteSet(
            f"Every one of the {len(deprived)} districts at or below IDACI "
            f"decile {decile} has a school scoring above Progress 8 "
            f"{max_local_p8} within {local_radius}m, so no district is "
            "route-eligible."
        )

    cohort = np.asarray(district_cohort, dtype=float)
    disadvantaged = (
        np.where(areas["IDACI Decile"].to_numpy() <= decile, cohort, 0.0)
        if disadvantage == "decile"
        else disadvantaged_cohort(areas, cohort)
    )
    eligible_disadvantaged = disadvantaged[eligible.index]
    if eligible_disadvantaged.sum() == 0:
        raise EmptyRouteSet(
            f"The {len(eligible)} route-eligible districts hold no disadvantaged "
            "student, so no route can hold a seat."
        )
    district_decile = eligible["IDACI Decile"].to_numpy()
    rank = (decile + 1 - district_decile) / decile if linear else 1 / district_decile
    seats = (
        capacity_scale
        * np.asarray(school_places, dtype=float)
        * disadvantaged.sum()
        / cohort.sum()
    )
    routes = build_routes(
        eligible,
        school_xy,
        min_distance,
        eligible_disadvantaged * rank**progressivity,
        (np.ceil(seats) if round_up else np.rint(seats)).astype(np.int64),
        shortest_only,
    )
    if routes.empty:
        raise EmptyRouteSet(
            f"No school lies more than {min_distance}m from any of the "
            f"{len(eligible)} route-eligible districts on a route holding a seat "
            f"at capacity scale {capacity_scale}, so the route set is empty."
        )

    # Every school can sit within the threshold of a district, or hold no seat
    # for it beyond, leaving it out of the transport scheme entirely. That is a
    # real result at a large min_distance or a small capacity_scale rather than
    # an error, but it is silent, so it is named here. Districts the
    # performance condition excludes are counted in the summary instead, since
    # it excludes them by the dozen. Under `shortest_only` most districts are
    # nearest no school, which leaves them unrouted for that reason instead.
    unrouted = eligible.loc[~eligible.index.isin(routes["district_idx"]), "LSOA21CD"]
    if len(unrouted) and not shortest_only:
        print(
            f"No secondary school beyond {min_distance}m holds a seat on a "
            "route, so no routes: " + ", ".join(unrouted)
        )

    print(
        f"{len(deprived)} districts at IDACI decile {decile} or "
        f"below, {len(eligible)} of them with no school above Progress 8 "
        f"{max_local_p8} within {local_radius}m, {len(routes)} routes to "
        f"{len(school_xy)} secondary schools, "
        f"{len(routes) / len(eligible):.1f} per district"
        + (
            ", each school keeping only its route to the nearest district."
            if shortest_only
            else "."
        )
    )
    return routes


def save_routes(routes: pd.DataFrame, out_dir: Path) -> None:
    """Write the route set out, whole as a CSV and by axis as arrays, to
    ROUTES_CSV and ROUTES_NPZ in `out_dir`.

    Args:
        routes (pd.DataFrame): The route set, as returned by `route_network`.

        out_dir (Path): Folder the route set is written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    routes.to_csv(out_dir / ROUTES_CSV, index=False)
    np.savez(
        out_dir / ROUTES_NPZ,
        route_capacities=routes["capacity"].to_numpy(dtype=np.int32),
        route_school_idx=routes["school_idx"].to_numpy(dtype=np.int32),
        route_district_idx=routes["district_idx"].to_numpy(dtype=np.int32),
    )
