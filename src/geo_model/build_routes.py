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

ROUTES_CSV = Path("temp/secondary_routes.csv")
ROUTES_NPZ = Path("temp/secondary_routes.npz")


def build_routes(
    districts: gpd.GeoDataFrame,
    school_xy: np.ndarray,
    min_distance: float,
    capacity: int,
) -> pd.DataFrame:
    """Build the routes connecting disadvantaged districts to schools.

    A route joins one district to one school, so the route set is the subset of
    (district, school) pairs lying more than `min_distance` apart. Distance is
    measured from the district's population centroid, the point its students are
    sampled around, to the school.

    Args:
        districts (gpd.GeoDataFrame): Disadvantaged districts, carrying an
        "LSOA21CD" column and a "Centroids" geometry. The frame's index is kept
        as `district_idx`, so it must be positional into the frame students were
        sampled from.

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
    centroids = np.asarray(districts["Centroids"])
    district_xy = np.column_stack(
        (shapely.get_x(centroids), shapely.get_y(centroids))
    )
    distances = spdist.cdist(district_xy, school_xy)
    district_pos, school_idx = np.nonzero(distances > min_distance)

    return pd.DataFrame({
        "route_id": np.arange(len(district_pos), dtype=np.int32),
        "district_idx": districts.index.to_numpy()[district_pos].astype(np.int32),
        "LSOA21CD": districts["LSOA21CD"].to_numpy()[district_pos],
        "school_idx": school_idx.astype(np.int32),
        "capacity": np.full(len(district_pos), capacity, dtype=np.int32),
    })


def route_network(
    areas: gpd.GeoDataFrame,
    school_xy: np.ndarray,
    decile: int = DISADVANTAGED_DECILE,
    min_distance: float = MIN_ROUTE_DISTANCE,
    capacity: int = ROUTE_CAPACITY,
) -> pd.DataFrame:
    """Select the disadvantaged districts of `areas` and route them to schools.

    Args:
        areas (gpd.GeoDataFrame): Every district, carrying "LSOA21CD", "IMD
        Decile" and "Centroids" columns, positionally indexed in the order
        students were sampled from.

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2), in the
        same CRS as the centroids.

        decile (int, optional): Districts at or below this IMD decile are the
        disadvantaged ones routes are built from. Defaults to
        DISADVANTAGED_DECILE.

        min_distance (float, optional): Districts are routed only to schools
        further away than this, in metres. Defaults to MIN_ROUTE_DISTANCE.

        capacity (int, optional): Seats on every route. Defaults to
        ROUTE_CAPACITY.

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

    disadvantaged = areas[areas["IMD Decile"] <= decile]
    if disadvantaged.empty:
        raise ValueError(
            f"No LSOA sits at or below IMD decile {decile}, so there are no "
            "disadvantaged districts to build routes from."
        )

    routes = build_routes(disadvantaged, school_xy, min_distance, capacity)
    if routes.empty:
        raise ValueError(
            f"No school lies more than {min_distance}m from any of the "
            f"{len(disadvantaged)} disadvantaged districts, so the route set is "
            "empty."
        )

    # Every school can sit within the threshold of a district, leaving it out of
    # the transport scheme entirely. That is a real result at a large
    # min_distance rather than an error, but it is silent, so it is named here.
    unrouted = disadvantaged.loc[
        ~disadvantaged.index.isin(routes["district_idx"]), "LSOA21CD"
    ]
    if len(unrouted):
        print(
            f"Every secondary school is within {min_distance}m, so no routes: "
            + ", ".join(unrouted)
        )

    print(
        f"{len(disadvantaged)} disadvantaged districts at IMD decile {decile} or "
        f"below, {len(routes)} routes to {len(school_xy)} secondary schools, "
        f"{len(routes) / len(disadvantaged):.1f} per district."
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
        route_capacities = routes["capacity"].to_numpy(dtype=np.int32),
        route_school_idx = routes["school_idx"].to_numpy(dtype=np.int32),
        route_district_idx = routes["district_idx"].to_numpy(dtype=np.int32),
    )
