from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geo_model.build_routes import route_network, save_routes
from geo_model.load_data import load_areas, load_schools
from geo_model.utils import (
    cohort_capacity,
    disadvantaged_students,
    district_index,
    rank_bundles,
    sample_students,
)

SEED = 20260907
OUTPUT_NPZ = Path("temp/prefprio.npz")

# Share of the secondary preference ranking driven by Progress 8 rather than
# distance, for students outside the disadvantaged districts. At 0.3 one
# Progress 8 point is worth roughly 1.4km of extra travel.
PERFORMANCE_WEIGHT = 0.3

# Share of the travel a route takes out of the ranking of the school it serves,
# for students outside the disadvantaged districts. At 0.5 a routed school ranks
# as if it stood half as far away, so a route is worth taking but does not put
# every distant school ahead of the local one.
ROUTE_DISCOUNT = 0.5

# The same two shares for students in the disadvantaged districts. They default
# to the other students' own, so the population is homogeneous until a sweep
# moves one group away from the other.
DISADVANTAGED_PERFORMANCE_WEIGHT = PERFORMANCE_WEIGHT
DISADVANTAGED_ROUTE_DISCOUNT = ROUTE_DISCOUNT


def cohort_sizes(areas: pd.DataFrame, phase: str) -> np.ndarray:
    """Students to sample in each area for a phase.

    Args:
        areas (pd.DataFrame): Districts carrying the "F4", "M4", "F11" and
        "M11" single-year-of-age counts.

        phase (str): "primary" takes the four-year-olds, "secondary" the
        eleven-year-olds.

    Returns:
        np.ndarray: One count per area, aligned with `areas`.
    """
    age = {"primary": "4", "secondary": "11"}[phase]
    return (areas[f"F{age}"].astype(int) + areas[f"M{age}"].astype(int)).to_numpy()


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

        disadvantaged (np.ndarray): Whether each student lives in a
        disadvantaged district, as `disadvantaged_students` gives it.

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

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Student
        preferences, school priorities, school capacities and route
        capacities, in the form `fast_DAT` takes. Route capacities are empty
        when there are no routes.
    """
    if routes is None:
        route_district = route_school = route_capacities = np.empty(0, dtype=np.int32)
    else:
        route_district = routes["district_idx"].to_numpy()
        route_school = routes["school_idx"].to_numpy()
        route_capacities = routes["capacity"].to_numpy(dtype=np.int32)

    preferences, priorities = rank_bundles(
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
    )
    return preferences, priorities, cohort_capacity(secondary_schools), route_capacities


def main() -> None:
    """Build the matching inputs for both phases and write them out."""
    rng = np.random.default_rng(SEED)

    geo_soton = load_areas()
    primary_schools, secondary_schools = load_schools()

    # Simulate student locations within each LSOA, clustered on its population centroid.
    primary_student_xy, primary_student_lsoa = sample_students(
        geo_soton["Borders"],
        geo_soton["Centroids"],
        cohort_sizes(geo_soton, "primary"),
        rng,
    )
    secondary_student_xy, secondary_student_lsoa = sample_students(
        geo_soton["Borders"],
        geo_soton["Centroids"],
        cohort_sizes(geo_soton, "secondary"),
        rng,
    )

    # Routes connect the disadvantaged districts to the secondary schools they
    # cannot reach unaided, so they exist for the secondary phase alone.
    secondary_routes = route_network(
        geo_soton,
        secondary_schools[["Easting", "Northing"]].to_numpy(),
        secondary_schools["P8MEA"].to_numpy(),
    )
    save_routes(secondary_routes)

    # School priorities are the transpose of the same pairwise distances that rank
    # student preferences, so one distance matrix per phase serves both. Progress 8
    # is a KS4 measure with no primary analogue, so it shapes secondary preferences
    # only, and priorities stay on distance and district alone in both phases.
    primary_student_preferences, primary_school_priorities = rank_bundles(
        primary_student_xy,
        primary_student_lsoa,
        primary_schools[["Easting", "Northing"]].to_numpy(),
        district_index(primary_schools, geo_soton),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )
    (
        secondary_student_preferences,
        secondary_school_priorities,
        secondary_school_capacities,
        _,
    ) = secondary_instance(
        secondary_student_xy,
        secondary_student_lsoa,
        geo_soton,
        secondary_schools,
        secondary_routes,
        disadvantaged_students(secondary_student_lsoa, geo_soton),
    )

    # The register publishes capacity across every year group a school teaches,
    # while the matching admits one cohort, so the two are not interchangeable.
    primary_school_capacities = cohort_capacity(primary_schools)

    print(
        f"Primary: {primary_school_capacities.sum()} seats in a cohort for "
        f"{len(primary_student_xy)} students."
    )
    print(
        f"Secondary: {secondary_school_capacities.sum()} seats in a cohort for "
        f"{len(secondary_student_xy)} students."
    )

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    # Compressed, since a priority array is mostly the filler that stands in for
    # the bundles a school never hears from. The district each student was
    # drawn in and each district's decile are saved alongside, so a matching of
    # this instance can be scored for segregation on its own.
    np.savez_compressed(
        OUTPUT_NPZ,
        primary_student_preferences=primary_student_preferences,
        secondary_student_preferences=secondary_student_preferences,
        primary_school_priorities=primary_school_priorities,
        primary_school_capacities=primary_school_capacities,
        secondary_school_priorities=secondary_school_priorities,
        secondary_school_capacities=secondary_school_capacities,
        primary_student_district=primary_student_lsoa,
        secondary_student_district=secondary_student_lsoa,
        district_decile=geo_soton["IMD Decile"].to_numpy(dtype=np.int64),
    )


if __name__ == "__main__":
    main()
