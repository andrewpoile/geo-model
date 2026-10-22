import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np

from geo_model.cli import add_priority_argument, add_region_arguments
from geo_model.data.regions import regions
from geo_model.data.schools import load_schools
from geo_model.model.preferences import (
    cohort_capacity,
    district_index,
    rank_bundles,
    secondary_instance,
)
from geo_model.model.routes import route_network
from geo_model.model.settings import (
    DISADVANTAGE,
    DISADVANTAGED_DECILE,
    PREFERENCE_NOISE,
    PRIORITY,
    SEED,
)
from geo_model.model.students import (
    cohort_sizes,
    disadvantaged_group,
    disadvantaged_students,
    draw_cohort_sizes,
    sample_students,
)
from geo_model.output import region_dir, save_routes

# Written to each region's folder.
OUTPUT_NPZ = "prefprio.npz"


def build(
    areas: gpd.GeoDataFrame,
    primary_schools: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    out_dir: Path,
    priority: str = PRIORITY,
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

        priority (str, optional): One of PRIORITY_CHOICES, the rule both
        phases' school priorities follow. Defaults to PRIORITY.
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
    # only, and priorities follow the same rule, `priority`, in both phases.
    primary_student_preferences, primary_preference_ranks = rank_bundles(
        primary_student_xy,
        primary_student_lsoa,
        primary_schools[["Easting", "Northing"]].to_numpy(),
        district_index(primary_schools, areas),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        priority=priority,
        noise_scale=PREFERENCE_NOISE,
        rng=rng,
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
        priority=priority,
        noise_seed=int(rng.integers(2**63)),
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
    # scored for segregation on its own, and the rule the priorities follow,
    # so the instance says how it was ranked.
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
        priority=priority,
    )


def main() -> None:
    """Build the matching inputs of every region named and write them out."""
    parser = argparse.ArgumentParser(
        description="Build the matching inputs for both phases and write them out."
    )
    add_priority_argument(parser)
    add_region_arguments(parser)
    args = parser.parse_args()

    for las, areas in regions(args.la, args.merge):
        primary_schools, secondary_schools = load_schools(las)
        build(
            areas,
            primary_schools,
            secondary_schools,
            region_dir(las, priority=args.priority),
            priority=args.priority,
        )


if __name__ == "__main__":
    main()
