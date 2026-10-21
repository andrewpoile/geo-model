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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
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
from geo_model.car_displacement import plot as plot_modes
from geo_model.cli import add_region_arguments
from geo_model.data.regions import regions
from geo_model.data.schools import load_schools
from geo_model.data.travel import NTS_YEARS, load_nts_mode_shares
from geo_model.dissimilarity import plot as plot_dissimilarity
from geo_model.dissimilarity import score_settings
from geo_model.measures.segregation import fsm_share, fsm_term
from geo_model.measures.statistics import routes_t_test, t_test_label
from geo_model.measures.summaries import (
    local_acceptance,
    local_difference,
    local_summary,
    route_utilisation,
    unassigned_rows,
    utilisation_summary,
)
from geo_model.measures.travel import CIRCUITY, MODES, mode_change
from geo_model.model.routes import EmptyRouteSet
from geo_model.model.settings import (
    DEFAULTS,
    DISADVANTAGE,
    DISADVANTAGE_CHOICES,
    LINEAR_PROGRESSIVITY,
    ROUND_UP_SEATS,
    SEED,
    Settings,
)
from geo_model.model.settings import N_SEEDS as HELD_OUT_SEEDS
from geo_model.model.students import Sample, student_samples
from geo_model.output import region_dir
from geo_model.sweep import (
    COLOURS,
    GROUP_COLOURS,
    INK,
    INTAKE_MEASURES,
    plot_lorenz,
    plot_map,
)

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

# Past this many routes the utilisation bars go unlabelled, too narrow to name.
MAX_LABELLED_ROUTES = 40

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


def save(fig: Figure, path: Path) -> None:
    """Write `fig` to `path`, at the resolution every plot is written at."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def draw_unassigned(ax: Axes, results: pd.DataFrame) -> None:
    """Draw a bar per scenario of the students left without a place, the mean
    over seeds, stacked by their group in the colours the sweep's unassigned
    panel uses.

    Args:
        ax (Axes): Axis to draw on.

        results (pd.DataFrame): The scenario rows of one setting, as
        `score_settings` returns them.
    """
    stack = (
        unassigned_rows(results)
        .pivot_table(
            index="scenario", columns="group", values="unassigned", aggfunc="mean"
        )
        .reindex(list(COLOURS))
    )
    bottom = np.zeros(len(stack))
    for group in ("disadvantaged", "other"):
        ax.bar(
            stack.index,
            stack[group],
            0.5,
            bottom=bottom,
            color=[GROUP_COLOURS[f"{group}, {scenario}"] for scenario in stack.index],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += stack[group].to_numpy()
    # Above the axis, where no bar reaches.
    ax.legend(
        handles=[
            Patch(color=colour, label=label) for label, colour in GROUP_COLOURS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        frameon=False,
    )
    ax.set_ylabel("Students left unassigned (mean over seeds)")
    ax.yaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def draw_school_intake(
    fig: Figure, schools: pd.DataFrame, fsm: pd.Series, measure: str
) -> None:
    """Fill `fig` with a heatmap of `measure` at every school, the mean over
    seeds with routes and without, beside the register's own figure, as the
    sweep's intake panels and FSM strip draw it.

    Schools are ordered by their value without routes, highest first. The
    scale is the sweep's: [0, 1] for the share, and for the term centred on 0
    and reaching the largest term drawn, the register's included.

    Args:
        fig (Figure): Figure to draw on, using constrained layout.

        schools (pd.DataFrame): The school rows of one setting, as
        `score_settings` returns them.

        fsm (pd.Series): As `fsm_term` or `fsm_share` returns it, whichever
        `measure` is.

        measure (str): A key of INTAKE_MEASURES.
    """
    intake = schools.assign(
        share=schools["disadvantaged"] / (schools["disadvantaged"] + schools["other"])
    ).pivot_table(
        index="school",
        columns="scenario",
        values=INTAKE_MEASURES[measure]["column"],
        aggfunc="mean",
    )
    table = (
        intake[list(COLOURS)]
        .assign(FSM=fsm.reindex(intake.index))
        .sort_values("without routes", ascending=False)
    )
    if measure == "share":
        vmin, vmax = 0.0, 1.0
    else:
        vmax = float(np.nanmax(np.abs(table.to_numpy())))
        vmin = -vmax
    ax = fig.subplots()
    sns.heatmap(
        table,
        vmin=vmin,
        vmax=vmax,
        cmap=INTAKE_MEASURES[measure]["cmap"],
        annot=True,
        fmt=".2f",
        cbar_kws={"label": INTAKE_MEASURES[measure]["label"]},
        ax=ax,
    )
    # The register counts pupils by their own eligibility rather than as the
    # model draws them, so its column stands apart from the model's two.
    ax.axvline(len(COLOURS), color="white", linewidth=4)
    # A word a line, so the scenario names fit their narrow columns.
    ax.set_xticklabels([column.replace(" ", "\n") for column in table.columns])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)


def draw_route_utilisation(fig: Figure, routes: pd.DataFrame) -> None:
    """Fill `fig` with a bar per route: the seats its disadvantaged and its
    other riders take, the mean over seeds, stacked below its unused seats,
    so every bar stands at the seats the route provides.

    The used seats carry the seeds' range. Routes are ordered as
    `route_utilisation` orders them, and named by district and school up to
    MAX_LABELLED_ROUTES of them.

    Args:
        fig (Figure): Figure to draw on, using constrained layout.

        routes (pd.DataFrame): The route rows of one setting, as
        `score_settings` returns them with `utilisation`.
    """
    per_route = route_utilisation(routes)
    used = routes.assign(used=routes["disadvantaged"] + routes["other"]).groupby(
        "route"
    )["used"]
    low = used.min().reindex(per_route["route"]).to_numpy()
    high = used.max().reindex(per_route["route"]).to_numpy()
    mean = per_route["used"].to_numpy()
    labelled = len(per_route) <= MAX_LABELLED_ROUTES
    x = np.arange(len(per_route))

    ax = fig.subplots()
    bottom = np.zeros(len(per_route))
    for column, colour, label in (
        (
            "disadvantaged",
            GROUP_COLOURS["disadvantaged, with routes"],
            "disadvantaged riders",
        ),
        ("other", GROUP_COLOURS["other, with routes"], "other riders"),
        ("unused", "#e1e0d9", "unused seats"),
    ):
        ax.bar(
            x,
            per_route[column],
            0.8,
            bottom=bottom,
            color=colour,
            # White edges would hide bars too narrow to label.
            edgecolor="white",
            linewidth=0.5 if labelled else 0,
            label=label,
        )
        bottom += per_route[column].to_numpy()
    ax.errorbar(
        x,
        mean,
        yerr=[mean - low, high - mean],
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=2 if labelled else 0,
        label="seeds' range of seats used",
    )
    if labelled:
        ax.set_xticks(
            x,
            per_route["LSOA21CD"] + " → " + per_route["school"],
            rotation=90,
            fontsize="small",
        )
    else:
        ax.set_xticks([])
        ax.set_xlabel("Routes, by seats provided")
    ax.set_xlim(-0.5, len(per_route) - 0.5)
    # Above the axis, where no bar reaches.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
    ax.set_ylabel("Seats on route (mean over seeds)")
    ax.yaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def draw_local_acceptance(fig: Figure, schools: pd.DataFrame) -> None:
    """Fill `fig` with a pair of bars per school, with routes and without:
    the local students it accepts of each group, the mean over seeds,
    stacked below the local applicants it rejects, so every bar stands at
    the school's local applicants.

    Below them, the local students each school accepts with routes less
    those without, a box over the seeds and a point per seed, as
    `car_displacement.plot` draws the change per mode. Schools are ordered
    as `local_acceptance` orders them.

    Args:
        fig (Figure): Figure to draw on, using constrained layout.

        schools (pd.DataFrame): The school rows of one setting, as
        `score_settings` returns them.
    """
    per_school = local_acceptance(schools)
    names = per_school["school"].unique()
    # Each school's pair of bars sits either side of its tick, with routes on
    # the left.
    width = 0.4
    x = np.arange(len(names)).repeat(2) + np.tile([-width / 2, width / 2], len(names))

    ax, change = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
    bottom = np.zeros(len(per_school))
    for column, colour in (
        (
            "local_disadvantaged",
            [GROUP_COLOURS[f"disadvantaged, {s}"] for s in per_school["scenario"]],
        ),
        (
            "local_other",
            [GROUP_COLOURS[f"other, {s}"] for s in per_school["scenario"]],
        ),
        ("rejected", "#e1e0d9"),
    ):
        ax.bar(
            x,
            per_school[column],
            width,
            bottom=bottom,
            color=colour,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += per_school[column].to_numpy()
    # Above the axis, where no bar reaches.
    ax.legend(
        handles=[
            *(
                Patch(color=colour, label=label.replace(", ", " accepted, "))
                for label, colour in GROUP_COLOURS.items()
            ),
            Patch(color="#e1e0d9", label="rejected"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        frameon=False,
    )
    ax.set_ylabel("Local applicants (mean over seeds)")
    # The rejected bars rest on the accepted, so where a school rejects
    # nobody their foot would pin the axis to its tallest bar. Set to the
    # default margin above it instead.
    ax.set_ylim(0, 1.05 * per_school["local_applicants"].max())

    difference = local_difference(schools)
    # Every seed is drawn as a point, so the box draws no outliers of its own.
    sns.boxplot(
        difference,
        x="school",
        y="difference",
        order=names,
        color="#d9dde3",
        showfliers=False,
        ax=change,
    )
    sns.stripplot(
        difference, x="school", y="difference", order=names, color=INK, ax=change
    )
    change.axhline(0, color="#898781", linewidth=1)
    change.set_ylabel("Local students accepted,\nwith routes minus without")
    change.set_xlabel("")
    change.tick_params(axis="x", rotation=90, labelsize="small")
    change.set_xlim(-0.5, len(names) - 0.5)
    for panel in (ax, change):
        panel.yaxis.grid(True, color="#e1e0d9")
        panel.set_axisbelow(True)
        sns.despine(ax=panel)


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
    per seed, to LOCALS_PNG. Each plot the scripts share
    is drawn with the function its own script uses.

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


def plot_front(
    evaluations: pd.DataFrame,
    front: pd.DataFrame,
    chosen: pd.Series,
    reference: dict[str, float],
    objectives: list[str],
    path: Path,
) -> None:
    """Plot every feasible setting scored against two objectives, the front
    drawn over them, the compromise ringed and the defaults marked, to
    `path`.

    Args:
        evaluations (pd.DataFrame): As returned by `optimise`.

        front (pd.DataFrame): As returned by `optimise`, sorted by the first
        objective, so its line runs along the front.

        chosen (pd.Series): The row of `front` `compromise` chooses.

        reference (dict[str, float]): The objectives at DEFAULTS, as
        `objective_values` returns them.

        objectives (list[str]): The two keys of OBJECTIVES to plot, along x
        then y.

        path (Path): PNG to write.
    """
    if len(objectives) != 2:
        raise ValueError(f"The front is plotted in two objectives, got {objectives}.")
    x, y = objectives
    feasible = evaluations[evaluations["feasible"]]

    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(7, 5.5), layout="constrained")
    ax = fig.subplots()
    ax.scatter(
        feasible[x], feasible[y], s=16, color="#a5a49f", linewidths=0, label="scored"
    )
    ax.plot(
        front[x],
        front[y],
        color=COLOURS["with routes"],
        marker="o",
        markersize=7,
        markeredgecolor="white",
        linewidth=1.5,
        label="Pareto front",
    )
    ax.scatter(
        chosen[x],
        chosen[y],
        s=260,
        facecolors="none",
        edgecolors=INK,
        linewidths=1.5,
        zorder=3,
        label="compromise",
    )
    ax.scatter(
        reference[x],
        reference[y],
        s=110,
        marker="D",
        color=INK,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
        label="defaults",
    )
    ax.set_xlabel(OBJECTIVES[x].label)
    ax.set_ylabel(OBJECTIVES[y].label)
    ax.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    sns.despine(ax=ax)

    save(fig, path)


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
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes to score candidates in, see `sweep`",
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
        default=ROUND_UP_SEATS,
        help="round each school's route seats up rather than to the nearest "
        "seat, before they are split between its routes",
    )
    parser.add_argument(
        "--linear-progressivity",
        action=argparse.BooleanOptionalAction,
        default=LINEAR_PROGRESSIVITY,
        help="weight route seats by the linear profile (T + 1 - D) / T of "
        "each district's IDACI decile D rather than by 1 / D",
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

    shares = load_nts_mode_shares(args.years)
    for las, areas in regions(args.la, args.merge):
        out_dir = region_dir(las) / OUT_DIR
        _, secondary_schools = load_schools(las)
        search(args, las, areas, secondary_schools, shares, out_dir)


if __name__ == "__main__":
    main()
