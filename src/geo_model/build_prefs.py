import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from geo_model.build_routes import (
    DISADVANTAGE,
    DISADVANTAGED_DECILE,
    route_network,
    save_routes,
)
from geo_model.load_data import load_schools
from geo_model.utils import (
    add_region_arguments,
    cohort_capacity,
    disadvantaged_group,
    disadvantaged_students,
    district_index,
    rank_bundles,
    region_dir,
    regions,
    sample_students,
)

SEED = 20260907
# Written to each region's folder.
OUTPUT_NPZ = "prefprio.npz"

# Share of the secondary preference ranking driven by Progress 8 rather than
# distance, for students who are not disadvantaged. Distances and scores are
# each scaled by their own spread over the region, so the trade-off is the
# region's: in Southampton, at 0.3, one Progress 8 point is worth roughly
# 1.4km of extra travel.
PERFORMANCE_WEIGHT = 0.3

# Share of the travel a route takes out of the ranking of the school it serves,
# for students who are not disadvantaged. At 0.5 a routed school ranks
# as if it stood half as far away, so a route is worth taking but does not put
# every distant school ahead of the local one.
ROUTE_DISCOUNT = 0.5

# The same two shares for disadvantaged students. They default
# to the other students' own, so the population is homogeneous until a sweep
# moves one group away from the other.
DISADVANTAGED_PERFORMANCE_WEIGHT = PERFORMANCE_WEIGHT
DISADVANTAGED_ROUTE_DISCOUNT = ROUTE_DISCOUNT


# The age of each phase's cohort: the four-year-olds start primary school, the
# eleven-year-olds secondary.
PHASE_AGE = {"primary": 4, "secondary": 11}


def cohort_sizes(areas: pd.DataFrame, phase: str) -> np.ndarray:
    """Expected students in each area's cohort for a phase, the mean over the
    years. Route seats are sized on it, so one route set serves every sample.

    Args:
        areas (pd.DataFrame): Districts carrying the "Age 4 Mean" and "Age 11
        Mean" columns `load_areas` gives them.

        phase (str): "primary" takes the four-year-olds, "secondary" the
        eleven-year-olds.

    Returns:
        np.ndarray: One mean cohort per area, aligned with `areas`.
    """
    return areas[f"Age {PHASE_AGE[phase]} Mean"].to_numpy(dtype=float)


def draw_cohort_sizes(
    areas: pd.DataFrame, phase: str, rng: np.random.Generator
) -> np.ndarray:
    """Students to sample in each area for a phase, drawn from a normal of
    its cohort's mean and standard deviation over the years.

    Each draw is rounded to the nearest student, and a negative draw places
    none.

    Args:
        areas (pd.DataFrame): Districts carrying the "Age 4 Mean", "Age 4 SD",
        "Age 11 Mean" and "Age 11 SD" columns `load_areas` gives them.

        phase (str): "primary" takes the four-year-olds, "secondary" the
        eleven-year-olds.

        rng (np.random.Generator): Source of randomness.

    Returns:
        np.ndarray: One count per area, aligned with `areas`.
    """
    age = PHASE_AGE[phase]
    draw = rng.normal(
        areas[f"Age {age} Mean"].to_numpy(), areas[f"Age {age} SD"].to_numpy()
    )
    return np.maximum(np.rint(draw), 0).astype(np.int64)


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
    )
    return preferences, ranks, cohort_capacity(secondary_schools), route_capacities


def build(
    areas: gpd.GeoDataFrame,
    primary_schools: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    out_dir: Path,
) -> None:
    """Build one region's matching inputs for both phases and write them out,
    to OUTPUT_NPZ in `out_dir` and the route set beside it.

    Args:
        areas (gpd.GeoDataFrame): The region's districts, as `load_areas`
        returns them.

        primary_schools (gpd.GeoDataFrame): The region's primary schools, as
        `load_schools` returns them.

        secondary_schools (gpd.GeoDataFrame): The region's secondary schools,
        as `load_schools` returns them.

        out_dir (Path): Folder the inputs are written to.
    """
    rng = np.random.default_rng(SEED)

    # Draw each LSOA's cohort, then simulate its students' locations, clustered
    # on its population centroid.
    primary_student_xy, primary_student_lsoa = sample_students(
        areas["Borders"],
        areas["Centroids"],
        draw_cohort_sizes(areas, "primary", rng),
        rng,
    )
    secondary_student_xy, secondary_student_lsoa = sample_students(
        areas["Borders"],
        areas["Centroids"],
        draw_cohort_sizes(areas, "secondary", rng),
        rng,
    )
    # Who is disadvantaged is drawn after both phases' locations, so the
    # locations do not depend on it.
    primary_student_disadvantaged, secondary_student_disadvantaged = (
        disadvantaged_group(
            student_lsoa,
            disadvantaged_students(student_lsoa, areas, rng),
            areas,
            DISADVANTAGED_DECILE,
            DISADVANTAGE,
        )
        for student_lsoa in (primary_student_lsoa, secondary_student_lsoa)
    )

    # Routes connect the most deprived districts to the secondary schools they
    # cannot reach unaided, so they exist for the secondary phase alone.
    secondary_routes = route_network(
        areas,
        secondary_schools[["Easting", "Northing"]].to_numpy(),
        secondary_schools["P8MEA"].to_numpy(),
        secondary_schools["PlacesOffered"].to_numpy(),
        cohort_sizes(areas, "secondary"),
    )
    save_routes(secondary_routes, out_dir)

    # School priorities are the transpose of the same pairwise distances that rank
    # student preferences, so one distance matrix per phase serves both. Progress 8
    # is a KS4 measure with no primary analogue, so it shapes secondary preferences
    # only, and priorities stay on distance and district alone in both phases.
    primary_student_preferences, primary_preference_ranks = rank_bundles(
        primary_student_xy,
        primary_student_lsoa,
        primary_schools[["Easting", "Northing"]].to_numpy(),
        district_index(primary_schools, areas),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )
    (
        secondary_student_preferences,
        secondary_preference_ranks,
        secondary_school_capacities,
        _,
    ) = secondary_instance(
        secondary_student_xy,
        secondary_student_lsoa,
        areas,
        secondary_schools,
        secondary_routes,
        secondary_student_disadvantaged,
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

    out_dir.mkdir(parents=True, exist_ok=True)
    # The district each student was drawn in and whether they are
    # disadvantaged are saved alongside, so a matching of this instance can be
    # scored for segregation on its own.
    np.savez_compressed(
        out_dir / OUTPUT_NPZ,
        primary_student_preferences=primary_student_preferences,
        secondary_student_preferences=secondary_student_preferences,
        primary_preference_ranks=primary_preference_ranks,
        primary_school_capacities=primary_school_capacities,
        secondary_preference_ranks=secondary_preference_ranks,
        secondary_school_capacities=secondary_school_capacities,
        primary_student_district=primary_student_lsoa,
        secondary_student_district=secondary_student_lsoa,
        primary_student_disadvantaged=primary_student_disadvantaged,
        secondary_student_disadvantaged=secondary_student_disadvantaged,
    )


def main() -> None:
    """Build the matching inputs of every region named and write them out."""
    parser = argparse.ArgumentParser(
        description="Build the matching inputs for both phases and write them out."
    )
    add_region_arguments(parser)
    args = parser.parse_args()

    for las, areas in regions(args.la, args.merge):
        primary_schools, secondary_schools = load_schools(las)
        build(areas, primary_schools, secondary_schools, region_dir(las))


if __name__ == "__main__":
    main()
