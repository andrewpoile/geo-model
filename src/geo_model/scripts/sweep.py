"""Sweep each model parameter one at a time around DEFAULTS, scoring every
setting with and without routes over the same student samples, and plot the
index, the students left unassigned, each school's intake and the change in
travel mode against every parameter.
"""

import argparse
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geo_model.analysis.scoring import settings_matchings
from geo_model.analysis.sweep import GRID, default_cell, sweep_tests, swept
from geo_model.cli import (
    add_disadvantage_argument,
    add_intake_argument,
    add_progressivity_argument,
    add_region_arguments,
    add_seat_arguments,
    add_seeds_argument,
    add_travel_arguments,
    add_workers_argument,
)
from geo_model.data.regions import regions
from geo_model.data.schools import load_schools
from geo_model.data.travel import load_nts_mode_shares
from geo_model.measures.segregation import (
    fsm_share,
    fsm_term,
)
from geo_model.measures.summaries import unassigned_rows
from geo_model.measures.travel import CIRCUITY, MODES, mode_change
from geo_model.model.settings import (
    DEFAULTS,
    DISADVANTAGE,
    LINEAR_PROGRESSIVITY,
    N_SEEDS,
    ROUND_UP_SEATS,
)
from geo_model.model.students import Sample, student_samples
from geo_model.output import region_dir
from geo_model.plots.comparison import plot_lorenz
from geo_model.plots.maps import plot_map
from geo_model.plots.style import INTAKE_MEASURES
from geo_model.plots.sweep import plot_sweep
from geo_model.scripts.dissimilarity import score_settings

# The sweep's folder within each region's, and the files written to it.
SWEEP_DIR = Path("sweep")
RESULTS_CSV = "sweep.csv"
SCHOOLS_CSV = "schools.csv"
MODES_CSV = "modes.csv"
TESTS_CSV = "t_tests.csv"
PLOT_PNG = "sweep.png"
MAP_PNG = "map.png"
LORENZ_PNG = "lorenz.png"


def sweep(
    samples: list[Sample],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    out_dir: Path,
    circuity: float = CIRCUITY,
    workers: int = 1,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every cell of the parameters `swept` names on the same samples.

    Every cell matches the same student samples, so the scores of two cells
    differ by their parameter values alone, and the default cell of every
    parameter reproduces `dissimilarity.run` and `car_displacement.run` at the
    same `disadvantage`.

    Args:
        samples (list[Sample]): Student samples, as yielded by
        `student_samples`, in seed order.

        areas (gpd.GeoDataFrame): Districts, as `route_network` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools, as `score_sample`
        takes them.

        shares (pd.DataFrame): Passed to `score_settings`.

        out_dir (Path): Folder the rows are written to.

        circuity (float, optional): Passed to `score_settings`. Defaults to
        CIRCUITY.

        workers (int, optional): Processes to score the cells in, one cell
        per task. A routed instance's preferences and their ranks take under
        1MB in Southampton and 37MB for Southampton and Hampshire merged.
        The speed-ups last measured, on a 24-core, 32GB machine, predate
        them: 4 workers ran the sweep 2.2x faster than one and 8 ran it 3x
        faster, while 16 ran out of memory holding priority arrays over
        every school, route and student, of up to 150MB each. Threads were
        slower than one process. Defaults to 1, which scores in this
        process.

        round_up (bool, optional): Passed to `score_settings`. Defaults to
        ROUND_UP_SEATS.

        linear (bool, optional): Passed to `score_settings`. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): Passed to `swept` and `score_settings`.
        Defaults to DISADVANTAGE.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The scenario rows,
        one per (parameter, value, seed, scenario), the school rows, one per
        (parameter, value, seed, scenario, school), and the mode rows, one
        per (parameter, value, seed, scenario, mode), each with "parameter"
        and "value" ahead of the columns `score_settings` returns. Also
        written to RESULTS_CSV, SCHOOLS_CSV and MODES_CSV in `out_dir`.
    """
    cells = [(p, settings) for p in swept(GRID, disadvantage) for settings in GRID[p]]
    score = functools.partial(
        score_settings,
        samples=samples,
        areas=areas,
        secondary_schools=secondary_schools,
        shares=shares,
        circuity=circuity,
        round_up=round_up,
        linear=linear,
        disadvantage=disadvantage,
    )
    if workers == 1:
        scored = [score(settings) for _, settings in cells]
    else:
        with ProcessPoolExecutor(workers) as pool:
            scored = list(pool.map(score, [settings for _, settings in cells]))

    rows, school_rows, mode_rows = [], [], []
    # The route rows are left empty, since `score_settings` keeps them only
    # when asked.
    for (parameter, settings), (*cell_rows, _) in zip(cells, scored):
        cell = {"parameter": parameter, "value": getattr(settings, parameter)}
        for tagged, untagged in zip(
            (rows, school_rows, mode_rows), cell_rows, strict=True
        ):
            tagged += [{**cell, **r} for r in untagged]

    results, schools, modes = (
        pd.DataFrame(rows),
        pd.DataFrame(school_rows),
        pd.DataFrame(mode_rows),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / RESULTS_CSV, index=False)
    schools.to_csv(out_dir / SCHOOLS_CSV, index=False)
    modes.to_csv(out_dir / MODES_CSV, index=False)
    return results, schools, modes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep each parameter of the secondary matching and score it "
        "with and without routes over the same fresh samples."
    )
    add_seeds_argument(parser, N_SEEDS)
    add_workers_argument(parser)
    add_travel_arguments(parser)
    add_seat_arguments(parser, ROUND_UP_SEATS)
    add_progressivity_argument(parser)
    add_disadvantage_argument(parser)
    add_intake_argument(parser, list(INTAKE_MEASURES))
    parser.add_argument(
        "--map",
        action="store_true",
        help="also map the first sample's matchings at the default settings, "
        "with routes and without",
    )
    parser.add_argument(
        "--lorenz",
        action="store_true",
        help="also plot the Lorenz curve of disadvantaged against seated "
        "students over schools at the default settings, with routes and "
        "without, and its Gini coefficient",
    )
    add_region_arguments(parser)
    args = parser.parse_args()

    shares = load_nts_mode_shares(args.years)
    parameters = swept(GRID, args.disadvantage)
    fsm = {"dissimilarity": fsm_term, "share": fsm_share}[args.intake]
    for las, areas in regions(args.la, args.merge):
        out_dir = region_dir(las) / SWEEP_DIR
        _, secondary_schools = load_schools(las)
        samples = list(student_samples(areas, args.seeds))

        results, schools, modes = sweep(
            samples,
            areas,
            secondary_schools,
            shares,
            out_dir,
            args.circuity,
            args.workers,
            args.round_up_seats,
            args.linear_progressivity,
            args.disadvantage,
        )
        print(" + ".join(las) + ":")
        tests = sweep_tests(results)
        tests.to_csv(out_dir / TESTS_CSV)
        summary = results.pivot_table(
            index=["parameter", "value"],
            columns="scenario",
            values="dissimilarity",
            aggfunc="mean",
        ).join(tests[["t", "p", "p_holm"]])
        print(
            "Mean index over seeds and its paired t-test, with routes against "
            "without, p Holm-corrected over every cell as p_holm:"
        )
        # Whole, so no test is folded away, and the p-values left unrounded,
        # so a small one does not print as 0.
        print(
            summary.loc[parameters]
            .round({"with routes": 3, "without routes": 3, "t": 2})
            .to_string()
        )
        change = mode_change(modes).pivot_table(
            index=["parameter", "value"],
            columns="mode",
            values="change",
            aggfunc="mean",
        )
        print("Change in students per mode with routes:")
        print(change.loc[parameters, MODES].round(1))
        unassigned = unassigned_rows(results).pivot_table(
            index=["parameter", "value"],
            columns=["group", "scenario"],
            values="unassigned",
            aggfunc="mean",
        )
        print("Students of each group left unassigned:")
        # Four columns under a two-level header fold at the default width.
        with pd.option_context("display.width", 100):
            print(unassigned.loc[parameters].round(1))
        plot_sweep(
            results,
            schools,
            modes,
            fsm(secondary_schools),
            GRID,
            parameters,
            out_dir / PLOT_PNG,
            args.intake,
        )
        if args.map:
            plot_map(
                areas,
                secondary_schools,
                samples[0][0],
                settings_matchings(
                    DEFAULTS,
                    samples[0],
                    areas,
                    secondary_schools,
                    args.round_up_seats,
                    args.linear_progressivity,
                    args.disadvantage,
                ),
                out_dir / MAP_PNG,
            )
        if args.lorenz:
            plot_lorenz(
                default_cell(schools, GRID), out_dir / LORENZ_PNG, args.disadvantage
            )
        print(
            f"Wrote {RESULTS_CSV}, {SCHOOLS_CSV}, {MODES_CSV}, {TESTS_CSV} and "
            f"the plots in {out_dir}."
        )


if __name__ == "__main__":
    main()
