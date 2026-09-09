from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy.spatial import distance as spdist

from build_prefs import geo_soton, secondary_school_xy

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


if not 1 <= DISADVANTAGED_DECILE <= 10:
    raise ValueError(
        f"DISADVANTAGED_DECILE must be an IMD decile in [1, 10], got "
        f"{DISADVANTAGED_DECILE}."
    )

# A student is tied to their district by position, since sample_students returns
# a positional area index, so district_idx only means anything if geo_soton is
# positionally indexed too.
if not geo_soton.index.equals(pd.RangeIndex(len(geo_soton))):
    raise ValueError(
        "geo_soton is not positionally indexed, so route district_idx would not "
        "line up with the area index sample_students returns."
    )

disadvantaged = geo_soton[geo_soton["IMD Decile"] <= DISADVANTAGED_DECILE]
if disadvantaged.empty:
    raise ValueError(
        f"No LSOA sits at or below IMD decile {DISADVANTAGED_DECILE}, so there "
        "are no disadvantaged districts to build routes from."
    )

secondary_routes = build_routes(
    disadvantaged, secondary_school_xy, MIN_ROUTE_DISTANCE, ROUTE_CAPACITY,
)
if secondary_routes.empty:
    raise ValueError(
        f"No school lies more than {MIN_ROUTE_DISTANCE}m from any of the "
        f"{len(disadvantaged)} disadvantaged districts, so the route set is empty."
    )

# Every school can sit within the threshold of a district, leaving it out of the
# transport scheme entirely. That is a real result at a large MIN_ROUTE_DISTANCE
# rather than an error, but it is silent, so it is named here.
unrouted = disadvantaged.loc[
    ~disadvantaged.index.isin(secondary_routes["district_idx"]), "LSOA21CD"
]
if len(unrouted):
    print(
        f"Every secondary school is within {MIN_ROUTE_DISTANCE}m, so no routes: "
        + ", ".join(unrouted)
    )

print(
    f"{len(disadvantaged)} disadvantaged districts at IMD decile "
    f"{DISADVANTAGED_DECILE} or below, {len(secondary_routes)} routes to "
    f"{len(secondary_school_xy)} secondary schools, "
    f"{len(secondary_routes) / len(disadvantaged):.1f} per district."
)

ROUTES_CSV.parent.mkdir(parents=True, exist_ok=True)
secondary_routes.to_csv(ROUTES_CSV, index=False)
np.savez(
    ROUTES_NPZ,
    route_capacities = secondary_routes["capacity"].to_numpy(dtype=np.int32),
    route_school_idx = secondary_routes["school_idx"].to_numpy(dtype=np.int32),
    route_district_idx = secondary_routes["district_idx"].to_numpy(dtype=np.int32),
)
