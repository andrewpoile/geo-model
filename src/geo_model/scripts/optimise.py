"""Search the route parameters for the settings that trade segregation off
against modal shift, by NSGA-II over the pipeline run end to end.

Every candidate setting builds its route network, matches every student
sample with and without routes and scores both matchings, as `sweep.sweep`
scores a cell, so a candidate's objectives are the sweep's own
measures at that setting. Every candidate matches the same samples, so two
candidates differ by their parameter values alone, and the front found is
fitted to those samples. The compromise setting of the front is therefore
compared with the same students matched without routes on samples the search
never saw.
"""

import argparse
import contextlib
import functools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from geo_model.analysis.search import (
    DEFAULT_OBJECTIVES,
    LEVERS,
    OBJECTIVES,
    OPTIMISABLE,
    Pipeline,
    Scored,
    cast,
    compromise,
    minimised,
    objective_values,
    searchable,
)
from geo_model.cli import (
    add_anon_argument,
    add_disadvantage_argument,
    add_intake_argument,
    add_progressivity_argument,
    add_region_arguments,
    add_seat_arguments,
    add_seeds_argument,
    add_travel_arguments,
    add_workers_argument,
)
from geo_model.data.anonymise import anonymise
from geo_model.data.regions import regions
from geo_model.data.schools import load_schools
from geo_model.data.travel import load_nts_mode_shares
from geo_model.measures.travel import CIRCUITY
from geo_model.model.routes import EmptyRouteSet
from geo_model.model.settings import (
    DEFAULTS,
    DISADVANTAGE,
    LINEAR_PROGRESSIVITY,
    ROUND_UP_SEATS,
    SEED,
    Settings,
)
from geo_model.model.settings import N_SEEDS as HELD_OUT_SEEDS
from geo_model.model.students import Sample, student_samples
from geo_model.output import region_dir
from geo_model.plots.front import plot_front
from geo_model.plots.style import INTAKE_MEASURES
from geo_model.scripts.run import run, score_settings

# The search's folder within each region's, and the files written to it.
OUT_DIR = Path("optimise")
EVALUATIONS_CSV = "evaluations.csv"
FRONT_CSV = "front.csv"
FRONT_PNG = "front.png"


# Several samples rather than the 30 `run` draws, since every candidate
# scores them all. The compromise is run on HELD_OUT_SEEDS more, as many as
# `run` draws, taken from the seeds after these.
N_SEEDS = 5
POP_SIZE = 20
GENERATIONS = 10


def score_feasible(settings: Settings, **kwargs: Any) -> Scored | None:
    """Score one setting with `score_settings`, or None where it leaves no route.

    A setting leaving no route has no routed matching to score, so it is
    infeasible rather than wrong, and is named here. Every other error is
    raised. Lives at module level so worker processes can import it.

    Args:
        settings (Settings): Passed to `score_settings`.

        **kwargs: Passed to `score_settings`.

    Returns:
        Scored | None: The rows `score_settings` returns, or None.
    """
    try:
        return score_settings(settings, **kwargs)
    except EmptyRouteSet as error:
        print(f"Infeasible, no route at {settings}: {error}")
        return None


def optimise(
    samples: list[Sample],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    out_dir: Path,
    parameters: list[str] = LEVERS,
    objectives: list[str] = DEFAULT_OBJECTIVES,
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    workers: int = 1,
    circuity: float = CIRCUITY,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search `parameters` by NSGA-II for the Pareto front of `objectives`.

    Every other parameter holds DEFAULTS, and every candidate matches the
    same samples. The front is the non-dominated set of every feasible
    setting scored, not only of the last generation, so no setting NSGA-II's
    truncation dropped is lost from it. The search is seeded from SEED, so it
    is reproducible.

    Args:
        samples (list[Sample]): Student samples, as yielded by
        `student_samples`, in seed order.

        areas (gpd.GeoDataFrame): Passed to `score_settings`.

        secondary_schools (gpd.GeoDataFrame): Passed to `score_settings`.

        shares (pd.DataFrame): Passed to `score_settings`.

        out_dir (Path): Folder the settings scored and the front are written
        to.

        parameters (list[str], optional): Keys of OPTIMISABLE to search, each
        over its `bounds`, every one `searchable` under `disadvantage`.
        Defaults to LEVERS.

        objectives (list[str], optional): Keys of OBJECTIVES to optimise.
        Defaults to DEFAULT_OBJECTIVES, every one.

        pop_size (int, optional): Candidate settings per generation. Defaults
        to POP_SIZE.

        generations (int, optional): Generations to run, the first being the
        random initial population. Defaults to GENERATIONS.

        workers (int, optional): Processes to score a generation's candidates
        in, one candidate per task, as `sweep` scores its cells. Defaults to
        1, which scores in this process.

        circuity (float, optional): Passed to `score_settings`. Defaults to
        CIRCUITY.

        round_up (bool, optional): Passed to `score_settings`. Defaults to
        ROUND_UP_SEATS.

        linear (bool, optional): Passed to `score_settings`. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): Passed to `score_settings`. Defaults to
        DISADVANTAGE.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Every setting scored, one row each
        with its "generation", parameter values, objectives and "feasible";
        and the feasible settings on the front, sorted by the first
        objective. Also written to EVALUATIONS_CSV and FRONT_CSV in
        `out_dir`.
    """
    if not samples:
        raise ValueError("No student sample to score, so every objective is undefined.")
    for named in (parameters, objectives):
        if len(set(named)) != len(named):
            raise ValueError(f"Each may be named once, got {named}.")
    allowed = searchable(disadvantage)
    barred = [parameter for parameter in parameters if parameter not in allowed]
    if barred:
        raise ValueError(
            f"Under disadvantage {disadvantage!r} {barred} cannot be searched: "
            "the other students' route discount is inert unless they ride, and "
            "under 'decile' the decile moves the group the index measures. "
            f"Choose from {allowed}."
        )

    score = functools.partial(
        score_feasible,
        samples=samples,
        areas=areas,
        secondary_schools=secondary_schools,
        shares=shares,
        circuity=circuity,
        round_up=round_up,
        linear=linear,
        disadvantage=disadvantage,
    )
    with (
        ProcessPoolExecutor(workers) if workers > 1 else contextlib.nullcontext()
    ) as pool:
        problem = Pipeline(
            parameters, objectives, score, map if pool is None else pool.map
        )
        minimize(problem, NSGA2(pop_size=pop_size), ("n_gen", generations), seed=SEED)

    evaluations = pd.DataFrame(problem.evaluations)
    feasible = evaluations[evaluations["feasible"]]
    if feasible.empty:
        raise ValueError(
            f"None of the {len(evaluations)} settings scored leaves a route, so "
            "there is no front."
        )
    on_front = NonDominatedSorting().do(
        minimised(feasible, objectives), only_non_dominated_front=True
    )
    front = (
        feasible.iloc[on_front]
        .drop(columns="feasible")
        .sort_values(objectives[0])
        .reset_index(drop=True)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluations.to_csv(out_dir / EVALUATIONS_CSV, index=False)
    front.to_csv(out_dir / FRONT_CSV, index=False)
    return evaluations, front


def search(
    args: argparse.Namespace,
    las: list[str],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Search one region's settings, then run its compromise with and
    without routes on held-out samples, as `main`'s arguments set out.

    Args:
        args (argparse.Namespace): `main`'s parsed arguments.

        las (list[str]): The region's local authorities.

        areas (gpd.GeoDataFrame): The region's districts, as `regions`
        yields them.

        secondary_schools (gpd.GeoDataFrame): The region's schools, as
        `load_schools` returns them.

        shares (pd.DataFrame): NTS mode shares, as `load_nts_mode_shares`
        returns them.

        out_dir (Path): Folder everything is written to.
    """
    print(" + ".join(las) + ":")
    # A seed's sample depends on its place in the stream alone, so the first
    # samples are the ones a search of as many seeds alone would draw.
    samples = list(student_samples(areas, args.seeds + args.held_out_seeds))
    searched, held_out = samples[: args.seeds], samples[args.seeds :]

    reference = objective_values(
        score_settings(
            DEFAULTS,
            searched,
            areas,
            secondary_schools,
            shares,
            args.circuity,
            args.round_up_seats,
            args.linear_progressivity,
            args.disadvantage,
        ),
        args.objectives,
    )
    evaluations, front = optimise(
        searched,
        areas,
        secondary_schools,
        shares,
        out_dir,
        args.parameters,
        args.objectives,
        args.pop_size,
        args.generations,
        args.workers,
        args.circuity,
        args.round_up_seats,
        args.linear_progressivity,
        args.disadvantage,
    )
    print(
        f"{len(front)} settings on the Pareto front, of the "
        f"{evaluations['feasible'].sum()} feasible among {len(evaluations)} scored:"
    )
    print(front.round(3).to_string(index=False))
    print(
        "At the defaults: "
        + ", ".join(f"{name} {value:.3f}" for name, value in reference.items())
    )
    chosen = compromise(front, args.objectives)
    print("Compromise setting:")
    print(chosen.round(3).to_string())

    print(f"At the compromise, on {len(held_out)} held-out samples:")
    run(
        replace(DEFAULTS, **cast({p: chosen[p] for p in args.parameters})),
        held_out,
        areas,
        secondary_schools,
        shares,
        out_dir,
        args.circuity,
        args.round_up_seats,
        args.linear_progressivity,
        args.disadvantage,
        args.intake,
        args.anon,
    )

    written = [EVALUATIONS_CSV, FRONT_CSV]
    if len(args.objectives) == 2:
        plot_front(
            evaluations,
            front,
            chosen,
            reference,
            args.objectives,
            out_dir / FRONT_PNG,
        )
        written.append(FRONT_PNG)
    else:
        print("The front is plotted in two objectives only, so no front plot.")
    print("Wrote " + ", ".join(written) + f" in {out_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the route parameters by NSGA-II for the settings "
        "that trade the dissimilarity index off against car displacement, "
        "scoring every candidate with and without routes over the same fresh "
        "samples, then plot and map the front's compromise setting against "
        "the same students without routes on held-out samples."
    )
    add_seeds_argument(parser, N_SEEDS)
    parser.add_argument(
        "--held-out-seeds",
        type=int,
        default=HELD_OUT_SEEDS,
        help="fresh samples, after the searched ones, to compare the "
        "compromise setting with and without routes on",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=OPTIMISABLE,
        default=LEVERS,
        help="parameters to search, each over the span GRID sweeps; the rest "
        "hold DEFAULTS. route_discount is searchable only under "
        "--disadvantage score-open, the decile under every choice but decile",
    )
    parser.add_argument(
        "--objectives", nargs="+", choices=list(OBJECTIVES), default=DEFAULT_OBJECTIVES
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=POP_SIZE,
        help="candidate settings per generation",
    )
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    add_workers_argument(parser)
    add_travel_arguments(parser)
    add_seat_arguments(parser, ROUND_UP_SEATS)
    add_progressivity_argument(parser)
    add_disadvantage_argument(parser)
    add_intake_argument(parser, list(INTAKE_MEASURES))
    add_region_arguments(parser)
    add_anon_argument(parser)
    args = parser.parse_args()

    shares = load_nts_mode_shares(args.years)
    for las, areas in regions(args.la, args.merge):
        out_dir = region_dir(las, args.anon) / OUT_DIR
        _, secondary_schools = load_schools(las)
        if args.anon:
            areas, secondary_schools = anonymise(areas, secondary_schools)
        search(args, las, areas, secondary_schools, shares, out_dir)


if __name__ == "__main__":
    main()
