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
from matplotlib.figure import Figure
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from geo_model.analysis.scoring import settings_matchings
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
from geo_model.measures.segregation import fsm_share, fsm_term
from geo_model.measures.statistics import routes_t_test, t_test_label
from geo_model.measures.summaries import (
    local_summary,
    utilisation_summary,
)
from geo_model.measures.travel import CIRCUITY, MODES, mode_change
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
from geo_model.output import region_dir, save
from geo_model.plots.comparison import (
    draw_local_acceptance,
    draw_route_utilisation,
    draw_school_intake,
    draw_unassigned,
    plot_dissimilarity,
    plot_lorenz,
    plot_modes,
)
from geo_model.plots.front import plot_front
from geo_model.plots.maps import plot_map
from geo_model.plots.style import INTAKE_MEASURES
from geo_model.scripts.dissimilarity import score_settings

# The search's folder within each region's, and the files written to it.
OUT_DIR = Path("optimise")
EVALUATIONS_CSV = "evaluations.csv"
FRONT_CSV = "front.csv"
FRONT_PNG = "front.png"
# The compromise setting against the same students without routes: every
# metric the sweep draws, the Lorenz curves and the map.
DISSIMILARITY_PNG = "dissimilarity.png"
UNASSIGNED_PNG = "unassigned.png"
INTAKE_PNG = "intake.png"
MODES_PNG = "car_displacement.png"
LORENZ_PNG = "lorenz.png"
MAP_PNG = "map.png"
ROUTES_PNG = "route_utilisation.png"
LOCALS_PNG = "local_acceptance.png"


# Several samples rather than the 30 `dissimilarity` draws, since every
# candidate scores them all. The comparison is drawn on HELD_OUT_SEEDS more,
# as many as `dissimilarity` draws, taken from the seeds after these.
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


def compare(
    settings: Settings,
    samples: list[Sample],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    out_dir: Path,
    circuity: float = CIRCUITY,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
    measure: str = "dissimilarity",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score `settings` over `samples` with routes and without, and plot the
    two against each other in `out_dir`.

    Draws every metric the sweep draws: the index per scenario and its
    paired t-test to DISSIMILARITY_PNG, who is left unassigned to
    UNASSIGNED_PNG, each school's intake to INTAKE_PNG and the expected modes
    and their change to MODES_PNG. Draws the Lorenz curves to LORENZ_PNG as well, a map of the
    first sample's two matchings to MAP_PNG, the seats each route provides
    and its riders take to ROUTES_PNG, and the local students each school
    accepts of those who applied, with the change the routes make to them
    per seed, to LOCALS_PNG. Each plot is drawn
    with its function in the plots subpackage.

    Args:
        settings (Settings): The parameter values to compare.

        samples (list[Sample]): Student samples, as yielded by
        `student_samples`, in seed order.

        areas (gpd.GeoDataFrame): Passed to `score_settings`.

        secondary_schools (gpd.GeoDataFrame): Passed to `score_settings`.

        shares (pd.DataFrame): Passed to `score_settings`.

        out_dir (Path): Folder the plots are written to.

        circuity (float, optional): Passed to `score_settings`. Defaults to
        CIRCUITY.

        round_up (bool, optional): Passed to `score_settings`. Defaults to
        ROUND_UP_SEATS.

        linear (bool, optional): Passed to `score_settings`. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): Passed to `score_settings`,
        `plot_lorenz` and `settings_matchings`. Defaults to DISADVANTAGE.

        measure (str, optional): A key of INTAKE_MEASURES, what the intake
        heatmap shows. Defaults to "dissimilarity".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The
        scenario, school, mode and route rows `score_settings` returns with
        `utilisation`.
    """
    if not samples:
        raise ValueError("No student sample to compare on.")
    results, schools, modes, routes = (
        pd.DataFrame(rows)
        for rows in score_settings(
            settings,
            samples,
            areas,
            secondary_schools,
            shares,
            circuity,
            round_up,
            linear,
            disadvantage,
            utilisation=True,
        )
    )
    plot_dissimilarity(results, out_dir / DISSIMILARITY_PNG)
    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(5, 4.5), layout="constrained")
    draw_unassigned(fig.subplots(), results)
    save(fig, out_dir / UNASSIGNED_PNG)
    fig = Figure(
        figsize=(8, 1.5 + 0.4 * schools["school"].nunique()), layout="constrained"
    )
    fsm = {"dissimilarity": fsm_term, "share": fsm_share}[measure]
    draw_school_intake(fig, schools, fsm(secondary_schools), measure)
    save(fig, out_dir / INTAKE_PNG)
    plot_modes(modes, out_dir / MODES_PNG)
    plot_lorenz(schools, out_dir / LORENZ_PNG, disadvantage)
    plot_map(
        areas,
        secondary_schools,
        samples[0][0],
        settings_matchings(
            settings,
            samples[0],
            areas,
            secondary_schools,
            round_up,
            linear,
            disadvantage,
        ),
        out_dir / MAP_PNG,
    )
    n_routes = routes["route"].nunique()
    fig = Figure(
        figsize=(min(max(8.0, 2 + 0.3 * n_routes), 30.0), 9), layout="constrained"
    )
    draw_route_utilisation(fig, routes)
    save(fig, out_dir / ROUTES_PNG)
    n_schools = schools["school"].nunique()
    fig = Figure(
        figsize=(min(max(8.0, 2 + 0.4 * n_schools), 30.0), 11), layout="constrained"
    )
    draw_local_acceptance(fig, schools)
    save(fig, out_dir / LOCALS_PNG)
    return results, schools, modes, routes


def search(
    args: argparse.Namespace,
    las: list[str],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Search one region's settings, then compare its compromise with and
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

    results, schools, modes, routes = compare(
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
    )
    print(
        f"At the compromise with routes and without, mean over "
        f"{len(held_out)} held-out samples:"
    )
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
        mode_change(modes).groupby("mode")["change"].mean()[MODES].round(1).to_string()
    )
    print(utilisation_summary(routes))
    print(local_summary(schools))

    plots = [
        DISSIMILARITY_PNG,
        UNASSIGNED_PNG,
        INTAKE_PNG,
        MODES_PNG,
        LORENZ_PNG,
        MAP_PNG,
        ROUTES_PNG,
        LOCALS_PNG,
    ]
    if len(args.objectives) == 2:
        plot_front(
            evaluations,
            front,
            chosen,
            reference,
            args.objectives,
            out_dir / FRONT_PNG,
        )
        plots.append(FRONT_PNG)
    else:
        print("The front is plotted in two objectives only, so no front plot.")
    print(
        f"Wrote {EVALUATIONS_CSV}, {FRONT_CSV}, " + ", ".join(plots) + f" in {out_dir}."
    )


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
    args = parser.parse_args()

    shares = load_nts_mode_shares(args.years)
    for las, areas in regions(args.la, args.merge):
        out_dir = region_dir(las) / OUT_DIR
        _, secondary_schools = load_schools(las)
        search(args, las, areas, secondary_schools, shares, out_dir)


if __name__ == "__main__":
    main()
