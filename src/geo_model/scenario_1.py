"""Scenario 1: every school offers a single route, its shortest, to the
route-eligible districts beyond the default minimum distance, with no
condition on nearby schools, holding all of the school's route seats.

The scenario's students are matched with and without routes over fresh
samples, and every plot `optimise` draws of a setting is drawn of it.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from geo_model.__main__ import INTAKE_MEASURES
from geo_model.build_routes import DISADVANTAGE, DISADVANTAGE_CHOICES, save_routes
from geo_model.dissimilarity import (
    DEFAULTS,
    Sample,
    Settings,
    settings_routes,
    student_samples,
    t_test_label,
)
from geo_model.load_data import NTS_YEARS, load_nts_mode_shares, load_schools
from geo_model.optimise import compare, route_utilisation, utilisation_summary
from geo_model.utils import (
    CIRCUITY,
    MODES,
    add_region_arguments,
    disadvantaged_group,
    district_groups,
    mode_change,
    region_dir,
    regions,
    routes_t_test,
)

# The scenario's folder within each region's, and the score rows written to it.
OUT_DIR = Path("scenario_1")
RESULTS_CSV = "results.csv"
SCHOOLS_CSV = "schools.csv"
MODES_CSV = "modes.csv"
ROUTE_UTILISATION_CSV = "route_utilisation.csv"
DISTRICTS_CSV = "districts.csv"

N_SEEDS = 100

# A route only to a school beyond the default minimum distance, two miles, and
# no school scoring above Progress 8 infinity, so no district is served by its
# local schools and the local radius is inert. Each school keeps only its route
# to the nearest district beyond the minimum distance, which holds all of the
# school's route seats, so progressivity is inert too.
SCENARIO = replace(
    DEFAULTS,
    progressivity=0.0,
    max_local_p8=np.inf,
    shortest_only=True,
)
# Seats are rounded up, so a school's shortest route keeps a seat where the
# school's route seats fall below half of one.
ROUND_UP = True


def district_rows(
    samples: list[Sample], areas: pd.DataFrame, decile: int, disadvantage: str
) -> pd.DataFrame:
    """Each district's disadvantaged and advantaged students in every
    sample, as `district_groups` gives them, a seed's cohorts being drawn
    afresh.

    Args:
        samples (list[Sample]): Student samples, as yielded by
        `student_samples`, in seed order.

        areas (pd.DataFrame): Districts, as `district_groups` takes them.

        decile (int): Passed to `disadvantaged_group`.

        disadvantage (str): Passed to `disadvantaged_group`.

    Returns:
        pd.DataFrame: One row per (seed, district), "seed" ahead of the
        columns `district_groups` returns.
    """
    return pd.concat(
        [
            district_groups(
                areas,
                student_lsoa,
                disadvantaged_group(student_lsoa, drawn, areas, decile, disadvantage),
            ).assign(seed=seed)
            for seed, (_, student_lsoa, drawn, _) in enumerate(samples)
        ],
        ignore_index=True,
    ).pipe(lambda rows: rows[["seed", *rows.columns.drop("seed")]])


def run(settings: Settings, scenario_dir: Path, description: str) -> None:
    """Match fresh samples with and without the routes of a scenario, as the
    command line sets out, and write and report the two against each other.

    Args:
        settings (Settings): The scenario, its capacity scale the command
        line's default.

        scenario_dir (Path): The scenario's folder within each region's.

        description (str): The command line's description.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument(
        "--capacity-scale",
        type=float,
        default=settings.capacity_scale,
        help="seats on a school's route as a multiple of the disadvantaged "
        "students' fair share of the places it offers",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=NTS_YEARS,
        help="NTS years to take mode shares from, pooled when more than one",
    )
    parser.add_argument(
        "--circuity",
        type=float,
        default=CIRCUITY,
        help="road distance per straight-line metre, applied before banding",
    )
    parser.add_argument(
        "--round-up-seats",
        action=argparse.BooleanOptionalAction,
        default=ROUND_UP,
        help="round each school's route seats up rather than to the nearest "
        "seat, before they are split between its routes",
    )
    parser.add_argument(
        "--disadvantage",
        choices=DISADVANTAGE_CHOICES,
        default=DISADVANTAGE,
        help="score: each LSOA's students are drawn disadvantaged in proportion "
        "to its IDACI score, and routes carry only them; score-open: drawn the "
        "same way, but routes carry every student of their district; decile: "
        "every student of an LSOA at or below the decile is disadvantaged and "
        "rides, the model as it was before",
    )
    parser.add_argument(
        "--intake",
        choices=list(INTAKE_MEASURES),
        default="dissimilarity",
        help="what the intake heatmap shows of each school: its term of the "
        "dissimilarity index, or the disadvantaged share of its intake",
    )
    add_region_arguments(parser)
    args = parser.parse_args()

    settings = replace(settings, capacity_scale=args.capacity_scale)
    shares = load_nts_mode_shares(args.years)
    for las, areas in regions(args.la, args.merge):
        print(" + ".join(las) + ":")
        out_dir = region_dir(las) / scenario_dir
        _, secondary_schools = load_schools(las)
        samples = list(student_samples(areas, args.seeds))

        save_routes(
            settings_routes(
                settings,
                areas,
                secondary_schools,
                args.round_up_seats,
                disadvantage=args.disadvantage,
            ),
            out_dir,
        )
        results, schools, modes, routes = compare(
            settings,
            samples,
            areas,
            secondary_schools,
            shares,
            out_dir,
            args.circuity,
            args.round_up_seats,
            disadvantage=args.disadvantage,
            measure=args.intake,
        )
        districts = district_rows(samples, areas, settings.decile, args.disadvantage)
        for frame, name in (
            (results, RESULTS_CSV),
            (schools, SCHOOLS_CSV),
            (modes, MODES_CSV),
            (routes, ROUTE_UTILISATION_CSV),
            (districts, DISTRICTS_CSV),
        ):
            frame.to_csv(out_dir / name, index=False)

        print(f"With routes and without, mean over {len(samples)} samples:")
        print(
            results.groupby("scenario")[
                ["dissimilarity", "n_unmatched", "unassigned_disadvantaged"]
            ]
            .mean()
            .round(3)
            .to_string()
        )
        print(t_test_label(routes_t_test(results)))
        print("Change in students per mode with routes:")
        print(
            mode_change(modes)
            .groupby("mode")["change"]
            .mean()[MODES]
            .round(1)
            .to_string()
        )
        print("Seats on each route and the riders taking them, mean over samples:")
        print(
            route_utilisation(routes)
            .drop(columns="used")
            .round(2)
            .to_string(index=False)
        )
        print(utilisation_summary(routes))
        print(
            f"Wrote the routes, {RESULTS_CSV}, {SCHOOLS_CSV}, {MODES_CSV}, "
            f"{ROUTE_UTILISATION_CSV}, {DISTRICTS_CSV} and the plots in {out_dir}."
        )


def main() -> None:
    run(
        SCENARIO,
        OUT_DIR,
        "Match fresh samples with and without scenario 1's routes, each "
        "school's shortest, and plot and map the two against each other.",
    )


if __name__ == "__main__":
    main()
