from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import distance as spdist

# LSOAs at or below this decile are the disadvantaged districts, the only
# districts routes are built from. 1 is the most deprived 10% of LSOAs.
DISADVANTAGED_DECILE = 3

# Metres. A route carries a student to a school they cannot reach unaided, so
# nearer schools need none. 3218m is two miles, the lower bound of the extended
# rights to free home-to-school travel held by low-income 11-16 year olds.
MIN_ROUTE_DISTANCE = 3218

# Seats on every route. Route capacities are exogenous to the SCT instance.
ROUTE_CAPACITY = 30

# A route carries a student past their local schools, so a district already
# within reach of a school performing above this needs none. P8MEA is centred
# on the England average, so 0 is an average school.
MAX_LOCAL_P8 = 0.0

# Metres. Two miles again, but read the other way round: the distance a district
# is taken to be served over. It coincides with MIN_ROUTE_DISTANCE by
# construction rather than by definition, so the two are swept apart.
LOCAL_RADIUS = 3218

ROUTES_CSV = Path("temp/secondary_routes.csv")
ROUTES_NPZ = Path("temp/secondary_routes.npz")


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


def build_routes(
    districts: gpd.GeoDataFrame,
    school_xy: np.ndarray,
    min_distance: float,
    capacity: int,
) -> pd.DataFrame:
    """Build the routes connecting route-eligible districts to schools.

    A route joins one district to one school, so the route set is the subset of
    (district, school) pairs lying more than `min_distance` apart. Distance is
    measured from the population centroid of the district, the point its
    students are sampled around, to the school.

    Args:
        districts (gpd.GeoDataFrame): Route-eligible districts, carrying an
        "LSOA21CD" column and a "Centroids" geometry. The index of the frame is
        kept as `district_idx`, so it must be positional into the frame students
        were sampled from.

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2), in the
        same CRS as the centroids. `school_idx` is positional into this array.

        min_distance (float): Districts are routed only to schools further away
        than this, in the units of the CRS.

        capacity (int): Seats on every route.

    Returns:
        pd.DataFrame: One row per route, with columns "route_id" (contiguous
        from 0, the route axis `fast_DAT` indexes), "district_idx", "LSOA21CD",
        "school_idx" and "capacity".
    """
    distances = spdist.cdist(centroid_xy(districts), school_xy)
    district_pos, school_idx = np.nonzero(distances > min_distance)

    return pd.DataFrame(
        {
            "route_id": np.arange(len(district_pos), dtype=np.int32),
            "district_idx": districts.index.to_numpy()[district_pos].astype(np.int32),
            "LSOA21CD": districts["LSOA21CD"].to_numpy()[district_pos],
            "school_idx": school_idx.astype(np.int32),
            "capacity": np.full(len(district_pos), capacity, dtype=np.int32),
        }
    )


def route_network(
    areas: gpd.GeoDataFrame,
    school_xy: np.ndarray,
    school_scores: np.ndarray,
    decile: int = DISADVANTAGED_DECILE,
    min_distance: float = MIN_ROUTE_DISTANCE,
    capacity: int = ROUTE_CAPACITY,
    max_local_p8: float = MAX_LOCAL_P8,
    local_radius: float = LOCAL_RADIUS,
) -> pd.DataFrame:
    """Select the route-eligible districts of `areas` and route them to schools.

    A district is route-eligible on two conditions: it sits at or below `decile`,
    and no school scoring above `max_local_p8` lies within `local_radius` of it,
    since a district a high-performing school already serves needs no transport
    past it. The local schools are read from the same population centroid the
    distance condition measures from.

    The disadvantaged group itself is the decile group, untouched by the
    performance condition, so the dissimilarity index still measures every
    district at or below `decile` whether it is route-eligible or not.

    Args:
        areas (gpd.GeoDataFrame): Every district, carrying "LSOA21CD", "IMD
        Decile" and "Centroids" columns, positionally indexed in the order
        students were sampled from.

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2), in the
        same CRS as the centroids.

        school_scores (np.ndarray): Progress 8 per school, shape (n_schools,),
        aligned with `school_xy`. `load_schools` drops a secondary school
        without a published score, so such a school is absent from both arrays
        and takes no part in this condition.

        decile (int, optional): Districts at or below this IMD decile are the
        disadvantaged ones routes are built from. Defaults to
        DISADVANTAGED_DECILE.

        min_distance (float, optional): Districts are routed only to schools
        further away than this, in metres. Defaults to MIN_ROUTE_DISTANCE.

        capacity (int, optional): Seats on every route. Defaults to
        ROUTE_CAPACITY.

        max_local_p8 (float, optional): A district with a school scoring above
        this within `local_radius` is not route-eligible. Defaults to
        MAX_LOCAL_P8.

        local_radius (float, optional): Radius around the centroid of a district
        that its local schools are read from, in metres. Defaults to
        LOCAL_RADIUS.

    Returns:
        pd.DataFrame: The route set, as returned by `build_routes`.
    """
    if not 1 <= decile <= 10:
        raise ValueError(f"decile must be an IMD decile in [1, 10], got {decile}.")

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

    disadvantaged = areas[areas["IMD Decile"] <= decile]
    if disadvantaged.empty:
        raise ValueError(
            f"No LSOA sits at or below IMD decile {decile}, so there are no "
            "disadvantaged districts to build routes from."
        )

    # A school above the threshold inside the radius serves the district
    # already, so no route is built from it. Carried on the district index, so
    # the eligible subset keeps the positions district_idx is read from.
    local = spdist.cdist(centroid_xy(disadvantaged), school_xy) <= local_radius
    served = pd.DataFrame(
        local & (np.asarray(school_scores) > max_local_p8), index=disadvantaged.index
    ).any(axis=1)
    eligible = disadvantaged[~served]
    if eligible.empty:
        raise ValueError(
            f"Every one of the {len(disadvantaged)} disadvantaged districts has "
            f"a school scoring above Progress 8 {max_local_p8} within "
            f"{local_radius}m, so no district is route-eligible."
        )

    routes = build_routes(eligible, school_xy, min_distance, capacity)
    if routes.empty:
        raise ValueError(
            f"No school lies more than {min_distance}m from any of the "
            f"{len(eligible)} route-eligible districts, so the route set is "
            "empty."
        )

    # Every school can sit within the threshold of a district, leaving it out of
    # the transport scheme entirely. That is a real result at a large
    # min_distance rather than an error, but it is silent, so it is named here.
    # Districts the performance condition excludes are counted in the summary
    # instead, since it excludes them by the dozen.
    unrouted = eligible.loc[~eligible.index.isin(routes["district_idx"]), "LSOA21CD"]
    if len(unrouted):
        print(
            f"Every secondary school is within {min_distance}m, so no routes: "
            + ", ".join(unrouted)
        )

    print(
        f"{len(disadvantaged)} disadvantaged districts at IMD decile {decile} or "
        f"below, {len(eligible)} of them with no school above Progress 8 "
        f"{max_local_p8} within {local_radius}m, {len(routes)} routes to "
        f"{len(school_xy)} secondary schools, "
        f"{len(routes) / len(eligible):.1f} per district."
    )
    return routes


def save_routes(routes: pd.DataFrame) -> None:
    """Write the route set out, whole as a CSV and by axis as arrays.

    Args:
        routes (pd.DataFrame): The route set, as returned by `route_network`.
    """
    ROUTES_CSV.parent.mkdir(parents=True, exist_ok=True)
    routes.to_csv(ROUTES_CSV, index=False)
    np.savez(
        ROUTES_NPZ,
        route_capacities=routes["capacity"].to_numpy(dtype=np.int32),
        route_school_idx=routes["school_idx"].to_numpy(dtype=np.int32),
        route_district_idx=routes["district_idx"].to_numpy(dtype=np.int32),
    )
