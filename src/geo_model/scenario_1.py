"""Scenario 1: every school offers a single route, its shortest, to the
route-eligible districts beyond the default minimum distance, with no
condition on nearby schools and seats on each district's fair share alone.

The scenario's students are matched with and without routes over fresh
samples, and every plot `optimise` draws of a setting is drawn of it.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from geo_model.__main__ import INTAKE_MEASURES
from geo_model.build_prefs import cohort_sizes
from geo_model.build_routes import DISADVANTAGE, DISADVANTAGE_CHOICES, save_routes
from geo_model.dissimilarity import (
    DEFAULTS,
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

# A route only to a school beyond the default minimum distance, two miles,
# seats on every district's fair share alone, and no school scoring above
# Progress 8 infinity, so no district is served by its local schools and the
# local radius is inert. Each school keeps only its route to the nearest
# district beyond the minimum distance.
SCENARIO = replace(
    DEFAULTS,
    progressivity=0.0,
    max_local_p8=np.inf,
    shortest_only=True,
)
# Seats are rounded up, so a school's shortest route keeps a seat where its
# fair share falls below half of one.
ROUND_UP = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match fresh samples with and without scenario 1's routes, "
        "each school's shortest, and plot and map the two against each other."
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument(
        "--capacity-scale",
        type=float,
        default=SCENARIO.capacity_scale,
        help="seats on a route as a multiple of its district's fair share of "
        "the places its school offers",
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
        help="round route seats up, so every route keeps one, rather than to "
        "the nearest seat, which leaves a route rounding to none unbuilt",
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

    settings = replace(SCENARIO, capacity_scale=args.capacity_scale)
    shares = load_nts_mode_shares(args.years)
    for las, areas in regions(args.la, args.merge):
        print(" + ".join(las) + ":")
        out_dir = region_dir(las) / OUT_DIR
        _, secondary_schools = load_schools(las)
        samples = list(
            student_samples(areas, cohort_sizes(areas, "secondary"), args.seeds)
        )

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
        # Every seed draws each district's cohort and its disadvantaged
        # students in the same numbers, so the first sample stands for all.
        _, student_lsoa, drawn = samples[0]
        districts = district_groups(
            areas,
            student_lsoa,
            disadvantaged_group(
                student_lsoa, drawn, areas, settings.decile, args.disadvantage
            ),
        )
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


if __name__ == "__main__":
    main()
