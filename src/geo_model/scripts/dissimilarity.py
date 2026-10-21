import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geo_model.analysis.scoring import score_sample, settings_routes
from geo_model.cli import add_region_arguments, add_seeds_argument
from geo_model.data.regions import regions
from geo_model.data.schools import load_schools
from geo_model.data.travel import load_nts_mode_shares
from geo_model.measures.statistics import routes_t_test, t_test_label
from geo_model.measures.travel import CIRCUITY
from geo_model.model.routes import route_network
from geo_model.model.settings import (
    DEFAULTS,
    DISADVANTAGE,
    LINEAR_PROGRESSIVITY,
    N_SEEDS,
    ROUND_UP_SEATS,
    Settings,
)
from geo_model.model.students import (
    Sample,
    cohort_sizes,
    disadvantaged_group,
    student_samples,
)
from geo_model.output import region_dir
from geo_model.plots.comparison import plot_dissimilarity

# Written to each region's folder.
RESULTS_CSV = "dissimilarity.csv"
PLOT_PNG = "dissimilarity.png"


def score_settings(
    settings: Settings,
    samples: list[Sample],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    circuity: float = CIRCUITY,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
    utilisation: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Score every sample at one setting of the parameters.

    Args:
        settings (Settings): The parameter values to build routes and rank
        preferences with.

        samples (list[Sample]): Student samples, as yielded by
        `student_samples`, in seed order.

        areas (gpd.GeoDataFrame): Districts, as `route_network` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools, as `score_sample`
        takes them.

        shares (pd.DataFrame): Passed to `score_sample`.

        circuity (float, optional): Passed to `score_sample`. Defaults to
        CIRCUITY.

        round_up (bool, optional): Passed to `settings_routes`. Defaults to
        ROUND_UP_SEATS.

        linear (bool, optional): Passed to `settings_routes`. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): Passed to `settings_routes` and
        `score_sample`, and sets each sample's disadvantaged group with
        `disadvantaged_group` at `settings.decile`. Defaults to DISADVANTAGE.

        utilisation (bool, optional): Keep the route rows. A row per seed
        and route would run to tens of millions over a sweep of a merged
        region's whole network, so they are kept only where read. Defaults
        to False.

    Returns:
        tuple[list[dict], list[dict], list[dict], list[dict]]: The scenario,
        school, mode and route rows `score_sample` returns for every sample,
        each tagged with "seed", the scenario rows with "n_routes" as well.
        The route rows are empty unless `utilisation`.
    """
    print(f"Scoring {settings}")
    routes = settings_routes(
        settings, areas, secondary_schools, round_up, linear, disadvantage
    )
    rows, school_rows, mode_rows, route_rows = [], [], [], []
    for seed, (student_xy, student_lsoa, drawn, noise_seed) in enumerate(samples):
        scenario_rows, intake_rows, travel_rows, rider_rows = score_sample(
            student_xy,
            student_lsoa,
            areas,
            secondary_schools,
            routes,
            disadvantaged_group(
                student_lsoa, drawn, areas, settings.decile, disadvantage
            ),
            shares,
            noise_seed,
            performance_weight=settings.performance_weight,
            route_discount=settings.route_discount,
            disadvantaged_performance_weight=settings.disadvantaged_performance_weight,
            disadvantaged_route_discount=settings.disadvantaged_route_discount,
            circuity=circuity,
            disadvantage=disadvantage,
        )
        rows += [{"seed": seed, "n_routes": len(routes), **r} for r in scenario_rows]
        school_rows += [{"seed": seed, **r} for r in intake_rows]
        mode_rows += [{"seed": seed, **r} for r in travel_rows]
        if utilisation:
            route_rows += [{"seed": seed, **r} for r in rider_rows]
    return rows, school_rows, mode_rows, route_rows


def run(
    n_seeds: int,
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Score the secondary matching with and without routes over fresh samples.

    Each seed draws a new student sample and scores it with `score_sample`.

    Args:
        n_seeds (int): Number of student samples to draw.

        areas (gpd.GeoDataFrame): The region's districts, as `route_network`
        takes them.

        secondary_schools (gpd.GeoDataFrame): The region's schools, as
        `score_sample` takes them.

        out_dir (Path): Folder the rows are written to.

    Returns:
        pd.DataFrame: One row per (seed, scenario), with column "seed" ahead
        of the scenario-row keys `score_sample` returns. Also written to
        RESULTS_CSV in `out_dir`.
    """
    routes = route_network(
        areas,
        secondary_schools[["Easting", "Northing"]].to_numpy(),
        secondary_schools["P8MEA"].to_numpy(),
        secondary_schools["PlacesOffered"].to_numpy(),
        cohort_sizes(areas, "secondary"),
    )
    shares = load_nts_mode_shares()

    rows = []
    for seed, (student_xy, student_lsoa, drawn, noise_seed) in enumerate(
        student_samples(areas, n_seeds)
    ):
        scenario_rows, _, _, _ = score_sample(
            student_xy,
            student_lsoa,
            areas,
            secondary_schools,
            routes,
            disadvantaged_group(
                student_lsoa, drawn, areas, DEFAULTS.decile, DISADVANTAGE
            ),
            shares,
            noise_seed,
        )
        rows += [{"seed": seed, **r} for r in scenario_rows]
        print(f"Seed {seed + 1} of {n_seeds}: {len(student_xy)} students.")

    results = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / RESULTS_CSV, index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the secondary matching with and without routes over fresh samples."
    )
    add_seeds_argument(parser, N_SEEDS)
    add_region_arguments(parser)
    args = parser.parse_args()

    for las, areas in regions(args.la, args.merge):
        out_dir = region_dir(las)
        _, secondary_schools = load_schools(las)
        results = run(args.seeds, areas, secondary_schools, out_dir)
        print(" + ".join(las) + ":")
        print(results.groupby("scenario")["dissimilarity"].describe())
        print(t_test_label(routes_t_test(results)))
        plot_dissimilarity(results, out_dir / PLOT_PNG)
        print(f"Wrote {RESULTS_CSV} and {PLOT_PNG} in {out_dir}.")


if __name__ == "__main__":
    main()
