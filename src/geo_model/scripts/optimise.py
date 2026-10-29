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
import functools
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
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
from geo_model.analysis.workers import task_map
from geo_model.cli import (
    add_anon_argument,
    add_disadvantage_argument,
    add_distance_argument,
    add_intake_argument,
    add_priority_argument,
    add_progressivity_argument,
    add_region_arguments,
    add_seat_arguments,
    add_seeds_argument,
    add_travel_arguments,
    add_workers_argument,
    parsed_settings,
)
from geo_model.data.network import Distance
from geo_model.data.regions import regions
from geo_model.data.travel import load_nts_mode_shares
from geo_model.measures.travel import CIRCUITY
from geo_model.model.routes import EmptyRouteSet
from geo_model.model.settings import (
    DEFAULTS,
    ROUND_UP_SEATS,
    SEED,
    Settings,
)
from geo_model.model.settings import N_SEEDS as HELD_OUT_SEEDS
from geo_model.model.students import Sample, student_samples
from geo_model.plots.front import plot_front
from geo_model.plots.style import INTAKE_MEASURES
from geo_model.scripts.run import load_region, run, score_settings

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
    district_distance: np.ndarray,
    district_minutes: np.ndarray | None,
    shares: pd.DataFrame,
    out_dir: Path,
    parameters: list[str] = LEVERS,
    objectives: list[str] = DEFAULT_OBJECTIVES,
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    workers: int = 1,
    circuity: float = CIRCUITY,
    rules: Settings = DEFAULTS,
    *,
    distance: str,
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

        district_distance (np.ndarray): Passed to `score_settings`.

        district_minutes (np.ndarray): Passed to `score_settings`.

        shares (pd.DataFrame): Passed to `score_settings`.

        out_dir (Path): Folder the settings scored and the front are written
        to.

        parameters (list[str], optional): Keys of OPTIMISABLE to search, each
        over its `bounds`, every one `searchable` under the disadvantage and
        priority of `rules`. Defaults to LEVERS.

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

        rules (Settings, optional): Passed to `Pipeline`, whose RULES every
        candidate takes. Defaults to DEFAULTS.

        distance (str): Passed to `score_settings`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Every setting scored, one row each
        with its "generation", parameter values, objectives, "feasible",
        "priority" and "distance", the rules it was scored under;
        and the feasible settings on the front, sorted by the first
        objective. Also written to EVALUATIONS_CSV and FRONT_CSV in
        `out_dir`.
    """
    if not samples:
        raise ValueError("No student sample to score, so every objective is undefined.")
    for named in (parameters, objectives):
        if len(set(named)) != len(named):
            raise ValueError(f"Each may be named once, got {named}.")
    allowed = searchable(rules.disadvantage, rules.priority)
    barred = [parameter for parameter in parameters if parameter not in allowed]
    if barred:
        raise ValueError(
            f"Under disadvantage {rules.disadvantage!r} and priority "
            f"{rules.priority!r} "
            f"{barred} cannot be searched: the other students' route discount "
            "is inert unless they ride, the priority discount under 'bracket', "
            "and under 'decile' the decile moves the group the index measures. "
            f"Choose from {allowed}."
        )

    score = functools.partial(
        score_feasible,
        samples=samples,
        areas=areas,
        secondary_schools=secondary_schools,
        district_distance=district_distance,
        district_minutes=district_minutes,
        shares=shares,
        circuity=circuity,
        distance=distance,
    )
    with task_map(score, workers) as scores:
        problem = Pipeline(parameters, objectives, scores, rules)
        minimize(problem, NSGA2(pop_size=pop_size), ("n_gen", generations), seed=SEED)

    evaluations = pd.DataFrame(problem.evaluations).assign(
        priority=rules.priority, distance=distance
    )
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
    to_school: Distance,
    district_distance: np.ndarray,
    district_minutes: np.ndarray | None,
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

        to_school (Distance): Distance from any point to every school, as
        `load_region` gives it.

        district_distance (np.ndarray): Distance from every district to every
        school, as `load_region` gives it.

        district_minutes (np.ndarray): Minutes by bus from every district to
        every school, as `load_region` gives it.

        shares (pd.DataFrame): NTS mode shares, as `load_nts_mode_shares`
        returns them.

        out_dir (Path): Folder everything is written to.
    """
    print(" + ".join(las) + ":")
    # A seed's sample depends on its place in the stream alone, so the first
    # samples are the ones a search of as many seeds alone would draw.
    samples = list(student_samples(areas, args.seeds + args.held_out_seeds, to_school))
    searched, held_out = samples[: args.seeds], samples[args.seeds :]
    # The defaults under the rules given, which the search varies.
    base = parsed_settings(args)

    reference = objective_values(
        score_settings(
            base,
            searched,
            areas,
            secondary_schools,
            district_distance,
            district_minutes,
            shares,
            args.circuity,
            distance=to_school.rule,
        ),
        args.objectives,
    )
    evaluations, front = optimise(
        searched,
        areas,
        secondary_schools,
        district_distance,
        district_minutes,
        shares,
        out_dir,
        args.parameters,
        args.objectives,
        args.pop_size,
        args.generations,
        args.workers,
        args.circuity,
        base,
        distance=to_school.rule,
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
    print(
        f"Compromise setting, under priority {chosen['priority']!r} and "
        f"distance {chosen['distance']!r}:"
    )
    # The rules are text, which a row cannot be rounded with.
    print(chosen.drop(["priority", "distance"]).round(3).to_string())

    print(f"At the compromise, on {len(held_out)} held-out samples:")
    run(
        replace(base, **cast({p: chosen[p] for p in args.parameters})),
        held_out,
        areas,
        secondary_schools,
        district_distance,
        district_minutes,
        shares,
        out_dir,
        args.circuity,
        distance=to_school.rule,
        measure=args.intake,
        anon=args.anon,
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
        "--disadvantage score-open, priority_discount only under --priority "
        "distance, the decile under every choice but decile",
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
    add_priority_argument(parser)
    add_distance_argument(parser)
    add_intake_argument(parser, list(INTAKE_MEASURES))
    add_region_arguments(parser)
    add_anon_argument(parser)
    args = parser.parse_args()

    shares = load_nts_mode_shares(args.years)
    for las, areas in regions(args.la, args.merge):
        (
            areas,
            secondary_schools,
            to_school,
            district_distance,
            district_minutes,
            folder,
        ) = load_region(
            las,
            areas,
            args.anon,
            args.priority,
            args.distance,
            DEFAULTS.max_route_minutes,
        )
        search(
            args,
            las,
            areas,
            secondary_schools,
            to_school,
            district_distance,
            district_minutes,
            shares,
            folder / OUT_DIR,
        )


if __name__ == "__main__":
    main()
