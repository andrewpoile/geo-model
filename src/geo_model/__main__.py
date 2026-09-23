import argparse
import functools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from geo_model.build_prefs import cohort_sizes
from geo_model.build_routes import ROUND_UP_SEATS
from geo_model.dissimilarity import (
    DEFAULTS,
    N_SEEDS,
    score_settings,
    student_samples,
)
from geo_model.load_data import (
    NTS_YEARS,
    load_areas,
    load_nts_mode_shares,
    load_schools,
)
from geo_model.utils import CIRCUITY, MODES, dissimilarity_terms, mode_change

SWEEP_DIR = Path("temp/sweep")
RESULTS_CSV = SWEEP_DIR / "sweep.csv"
SCHOOLS_CSV = SWEEP_DIR / "schools.csv"
MODES_CSV = SWEEP_DIR / "modes.csv"
PLOT_PNG = SWEEP_DIR / "sweep.png"

# One-at-a-time grids: a parameter runs over its values while the others hold
# DEFAULTS, so every grid carries its own default.
GRID = {
    # Decile 10 would make every student disadvantaged and the index undefined.
    "decile": [replace(DEFAULTS, decile=d) for d in range(1, 10)],
    # Whole miles in metres. Every district lies within 9.9km of every school,
    # so the route set stays non-empty throughout.
    "min_distance": [
        replace(DEFAULTS, min_distance=m) for m in (0, 1609, 3218, 4828, 6437, 8047)
    ],
    # 1 is the fair share of the PAN. The smallest PAN is 120 and the city's
    # cohort 2918, so from 2918 / 120 = 24.3 every route holds its district's
    # whole cohort and no route capacity binds.
    "capacity_scale": [
        replace(DEFAULTS, capacity_scale=k)
        for k in (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    ],
    # P8MEA runs from -0.99 to 0.82 over the 12 secondary schools. At -0.72 or
    # below every disadvantaged district has a school above the threshold within
    # the radius and the route set empties, so the grid starts at -0.5; above
    # 0.82 no school is, so the condition is inert and the route set is whole.
    "max_local_p8": [replace(DEFAULTS, max_local_p8=p / 4) for p in range(-2, 5)],
    # Half miles in metres. Every disadvantaged district has a school above
    # MAX_LOCAL_P8 within 5.6km, so the route set empties from there.
    "local_radius": [
        replace(DEFAULTS, local_radius=r)
        for r in (0, 805, 1609, 2414, 3218, 4023, 4828)
    ],
    "performance_weight": [
        replace(DEFAULTS, performance_weight=w / 10) for w in range(11)
    ],
    "disadvantaged_performance_weight": [
        replace(DEFAULTS, disadvantaged_performance_weight=w / 10) for w in range(11)
    ],
    # Routes leave only the districts at or below the decile that makes a student
    # disadvantaged, so no other student is offered one and their route discount
    # is inert. Only the disadvantaged students' discount is swept.
    "disadvantaged_route_discount": [
        replace(DEFAULTS, disadvantaged_route_discount=d / 10) for d in range(11)
    ],
}

AXIS_LABELS = {
    "decile": "Disadvantaged at or below IMD decile",
    "min_distance": "Minimum route distance (m)",
    "capacity_scale": "Route capacity scale (× fair share of PAN)",
    "max_local_p8": "Highest Progress 8 allowed nearby",
    "local_radius": "Local performance radius (m)",
    "performance_weight": "Performance weight, other students",
    "disadvantaged_performance_weight": "Performance weight, disadvantaged students",
    "disadvantaged_route_discount": "Route discount, disadvantaged students",
}

COLOURS = {"with routes": "#2a78d6", "without routes": "#52514e"}
# Okabe-Ito hues for the survey modes, and the route in the darkest ink since
# it is the mode the routes add.
MODE_COLOURS = {
    "walk": "#0173b2",
    "cycle": "#029e73",
    "car": "#d55e00",
    "bus": "#de8f05",
    "other": "#cc78bc",
    "route": "#1f2933",
}
# The scenario hues again, the disadvantaged group in full ink and the rest
# washed out, so the unassigned panel reads against the scenario panels.
GROUP_COLOURS = {
    "disadvantaged, with routes": "#2a78d6",
    "disadvantaged, without routes": "#52514e",
    "other, with routes": "#7fb0e3",
    "other, without routes": "#a5a49f",
}
DEFAULT_LINE = {"color": "#898781", "linestyle": "--", "linewidth": 1}
# What the intake panels and the FSM strip show of each school: its term of the
# dissimilarity index, diverging either side of the city-wide mix, or the
# disadvantaged share of its intake, in one hue from light to dark.
INTAKE_MEASURES = {
    "dissimilarity": {
        "column": "dissimilarity_term",
        "cmap": "RdBu_r",
        "label": "Share of disadvantaged less share of other students",
    },
    "share": {
        "column": "share",
        "cmap": "Blues",
        "label": "Disadvantaged share of intake",
    },
}


def sweep(
    samples: list[tuple[np.ndarray, np.ndarray]],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    circuity: float = CIRCUITY,
    workers: int = 1,
    round_up: bool = ROUND_UP_SEATS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every cell of GRID on the same samples.

    Every cell matches the same student samples, so the scores of two cells
    differ by their parameter values alone, and the default cell of every
    parameter reproduces `dissimilarity.run` and `car_displacement.run`.

    Args:
        samples (list[tuple[np.ndarray, np.ndarray]]): Student samples, as
        yielded by `student_samples`, in seed order.

        areas (gpd.GeoDataFrame): Districts, as `route_network` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools, as `score_sample`
        takes them.

        shares (pd.DataFrame): Passed to `score_settings`.

        circuity (float, optional): Passed to `score_settings`. Defaults to
        CIRCUITY.

        workers (int, optional): Processes to score the cells in, one cell
        per task. A routed instance carries a priority array of up to
        150MB, so memory rather than cores bounds the gain: on a 24-core,
        32GB machine 4 workers ran the sweep 2.2x faster than one, 8 ran it
        3x faster, and 16 ran out of memory. Threads were slower than one
        process. Defaults to 1, which scores in this process.

        round_up (bool, optional): Passed to `score_settings`. Defaults to
        ROUND_UP_SEATS.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The scenario rows,
        one per (parameter, value, seed, scenario), the school rows, one per
        (parameter, value, seed, scenario, school), and the mode rows, one
        per (parameter, value, seed, scenario, mode), each with "parameter"
        and "value" ahead of the columns `score_settings` returns. Also
        written to RESULTS_CSV, SCHOOLS_CSV and MODES_CSV.
    """
    cells = [(parameter, s) for parameter, group in GRID.items() for s in group]
    score = functools.partial(
        score_settings,
        samples=samples,
        areas=areas,
        secondary_schools=secondary_schools,
        shares=shares,
        circuity=circuity,
        round_up=round_up,
    )
    if workers == 1:
        scored = [score(settings) for _, settings in cells]
    else:
        with ProcessPoolExecutor(workers) as pool:
            scored = list(pool.map(score, [settings for _, settings in cells]))

    rows, school_rows, mode_rows = [], [], []
    for (parameter, settings), cell_rows in zip(cells, scored):
        cell = {"parameter": parameter, "value": getattr(settings, parameter)}
        for tagged, untagged in zip((rows, school_rows, mode_rows), cell_rows):
            tagged += [{**cell, **r} for r in untagged]

    results, schools, modes = (
        pd.DataFrame(rows),
        pd.DataFrame(school_rows),
        pd.DataFrame(mode_rows),
    )
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    schools.to_csv(SCHOOLS_CSV, index=False)
    modes.to_csv(MODES_CSV, index=False)
    return results, schools, modes


def legend_handles(colours: dict[str, str]) -> list[Line2D]:
    """Proxy artists for the lines in `colours` and the default marker."""
    return [
        Line2D([], [], color=colour, marker="o", linewidth=2, label=label)
        for label, colour in colours.items()
    ] + [Line2D([], [], label="default", **DEFAULT_LINE)]


def positions(parameter: str) -> pd.Series:
    """The x position of every value `parameter` sweeps, indexed by value.

    Every panel of a column is drawn against these positions rather than
    against the values themselves, so the panels that can only be categorical
    — the stacked bars and the intake heatmaps — share one axis with the line
    panels. The positions are the cell centres `sns.heatmap` draws its columns
    at, so the bars and the lines land on the heatmap's own grid.

    Args:
        parameter (str): A key of GRID.

    Returns:
        pd.Series: The position of every value the parameter sweeps, in the
        order GRID holds them.
    """
    values = [getattr(settings, parameter) for settings in GRID[parameter]]
    return pd.Series(np.arange(len(values)) + 0.5, index=values)


def plot_parameter(
    ax: Axes,
    rows: pd.DataFrame,
    parameter: str,
    y: str,
    hue: str,
    colours: dict[str, str],
) -> None:
    """Draw one parameter's line panel: the mean over seeds of `y` for
    every level of `hue`, banded by the seeds' range.

    Args:
        ax (Axes): Axis to draw on.

        rows (pd.DataFrame): Rows `sweep` returns, carrying `y` and `hue`.

        parameter (str): A key of GRID.

        y (str): Column to draw.

        hue (str): Column giving one line per level.

        colours (dict[str, str]): Colour of every level of `hue`, in the
        order the lines are drawn.
    """
    at = positions(parameter)
    cell = rows[rows["parameter"] == parameter]
    sns.lineplot(
        cell.assign(position=cell["value"].map(at)),
        x="position",
        y=y,
        hue=hue,
        hue_order=list(colours),
        palette=colours,
        errorbar=("pi", 100),
        marker="o",
        linewidth=2,
        markersize=6,
        legend=False,
        ax=ax,
    )
    ax.axvline(at[asdict(DEFAULTS)[parameter]], **DEFAULT_LINE)
    ax.set_xlabel(AXIS_LABELS[parameter])
    ax.yaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def fsm_share(secondary_schools: pd.DataFrame) -> pd.Series:
    """The share of each school's pupils eligible for free school meals.

    The register's own measure of a school's disadvantage, so the point of
    comparison for the share the model seats. The two are not the same
    measure: the model counts students by the deprivation decile of their
    district, the register counts pupils by their own eligibility.

    Args:
        secondary_schools (pd.DataFrame): Schools carrying
        "EstablishmentName" and "PercentageFSM".

    Returns:
        pd.Series: The share, in [0, 1], indexed by establishment name. A
        school the register holds no figure for is named and left NaN.
    """
    fsm = secondary_schools.set_index("EstablishmentName")["PercentageFSM"] / 100
    if fsm.isna().any():
        print(
            "No FSM percentage recorded, so drawn blank: "
            + ", ".join(fsm.index[fsm.isna()])
        )
    return fsm


def fsm_term(secondary_schools: pd.DataFrame) -> pd.Series:
    """Each school's term of the dissimilarity index, pupils eligible for free
    school meals against the rest, as the register counts them.

    The register's own measure of how a school's intake departs from the
    city's, on the scale of the terms the model seats, though not the same
    measure: the model counts students by the deprivation decile of their
    district, the register counts pupils by their own eligibility. The
    register takes "PercentageFSM" over fewer pupils than "NumberOfPupils"
    at schools with a sixth form, so the pupils it was taken over are
    recovered as FSM * 100 / PercentageFSM, to within the percentage's
    rounding.

    Args:
        secondary_schools (pd.DataFrame): Schools carrying
        "EstablishmentName", "FSM" and "PercentageFSM".

    Returns:
        pd.Series: The term, indexed by establishment name. A school the
        register holds no figures for is named, left NaN and left out of the
        totals the other terms are taken over.

    Raises:
        ValueError: If a school records no pupil eligible, from which the
        pupils the percentage was taken over cannot be recovered.
    """
    fsm = secondary_schools.set_index("EstablishmentName")[["FSM", "PercentageFSM"]]
    missing = fsm.isna().any(axis=1)
    if missing.any():
        print(
            "No FSM count recorded, so drawn blank and left out of the FSM "
            "totals: " + ", ".join(fsm.index[missing])
        )
    held = fsm[~missing]
    if (held["PercentageFSM"] == 0).any():
        raise ValueError(
            "No pupil recorded eligible, so the pupils the percentage was taken "
            "over cannot be recovered: "
            + ", ".join(held.index[held["PercentageFSM"] == 0])
        )
    assessed = held["FSM"] * 100 / held["PercentageFSM"]
    terms = dissimilarity_terms(held["FSM"], assessed - held["FSM"])
    return pd.Series(terms, index=held.index).reindex(fsm.index)


def plot_stack(ax: Axes, rows: pd.DataFrame, parameter: str) -> None:
    """Draw one parameter's unassigned panel: a bar per scenario at every
    value, stacked by the group the unplaced students belong to.

    Args:
        ax (Axes): Axis to draw on.

        rows (pd.DataFrame): Rows `unassigned_rows` returns.

        parameter (str): A key of GRID.
    """
    at = positions(parameter)
    cell = rows[rows["parameter"] == parameter]
    stack = cell.pivot_table(
        index="value", columns="series", values="unassigned", aggfunc="mean"
    ).reindex(at.index)
    width = 0.38
    for offset, scenario in zip((-width / 2, width / 2), COLOURS):
        bottom = np.zeros(len(stack))
        for group in ("disadvantaged", "other"):
            series = f"{group}, {scenario}"
            ax.bar(
                at.to_numpy() + offset,
                stack[series],
                width,
                bottom=bottom,
                color=GROUP_COLOURS[series],
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += stack[series].to_numpy()
    ax.axvline(at[asdict(DEFAULTS)[parameter]], **DEFAULT_LINE)
    ax.set_xlabel(AXIS_LABELS[parameter])
    ax.yaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def unassigned_rows(results: pd.DataFrame) -> pd.DataFrame:
    """The students each group is left without a place, one row per group.

    A student unseated by the matching is scored by neither the index nor the
    intake panels, both of which count seated students alone, so the burden of
    going unplaced is only visible here.

    Args:
        results (pd.DataFrame): The scenario rows `sweep` returns, carrying
        "n_unmatched" and "unassigned_disadvantaged".

    Returns:
        pd.DataFrame: Two rows per scenario row, carrying "unassigned" (that
        group's students with no place) and "series", a key of GROUP_COLOURS
        naming the group and the scenario.
    """
    unmatched, disadvantaged = (
        results["n_unmatched"],
        results["unassigned_disadvantaged"],
    )
    rows = pd.concat(
        [
            results.assign(group="disadvantaged", unassigned=disadvantaged),
            results.assign(group="other", unassigned=unmatched - disadvantaged),
        ]
    )
    return rows.assign(series=rows["group"] + ", " + rows["scenario"])


def plot_intake(
    ax: Axes,
    schools: pd.DataFrame,
    parameter: str,
    scenario: str,
    order: pd.Index,
    measure: str,
    limits: tuple[float, float],
) -> None:
    """Draw one scenario's intake panel: `measure` of each school's intake,
    averaged over seeds, as a gradient over the values.

    Args:
        ax (Axes): Axis to draw on.

        schools (pd.DataFrame): The school rows `sweep` returns, carrying the
        measure's column.

        parameter (str): A key of GRID.

        scenario (str): A key of COLOURS.

        order (pd.Index): School names, top to bottom.

        measure (str): A key of INTAKE_MEASURES.

        limits (tuple[float, float]): The values the colour scale ends at.
    """
    cell = schools[
        (schools["parameter"] == parameter) & (schools["scenario"] == scenario)
    ]
    at = positions(parameter)
    intake = cell.pivot_table(
        index="school", columns="value", values=INTAKE_MEASURES[measure]["column"]
    )
    intake = intake.reindex(index=order, columns=at.index)
    vmin, vmax = limits
    sns.heatmap(
        intake,
        vmin=vmin,
        vmax=vmax,
        cmap=INTAKE_MEASURES[measure]["cmap"],
        cbar=False,
        ax=ax,
    )
    # The cell's left edge, since the position is its centre.
    default = at[asdict(DEFAULTS)[parameter]] - 0.5
    ax.add_patch(Rectangle((default, 0), 1, len(intake), fill=False, **DEFAULT_LINE))
    ax.set_xlabel(AXIS_LABELS[parameter])
    ax.set_ylabel(scenario)
    ax.tick_params(axis="y", rotation=0)


def plot_fsm(
    ax: Axes, fsm: pd.Series, order: pd.Index, measure: str, limits: tuple[float, float]
) -> None:
    """Draw the FSM strip: the register's measure for each school, on the
    intake panels' scale, as the fixed point of comparison for both.

    Args:
        ax (Axes): Axis to draw on.

        fsm (pd.Series): As returned by `fsm_term` or `fsm_share`, whichever
        `measure` is.

        order (pd.Index): School names, top to bottom.

        measure (str): A key of INTAKE_MEASURES.

        limits (tuple[float, float]): The values the colour scale ends at.
    """
    vmin, vmax = limits
    sns.heatmap(
        fsm.reindex(order).to_frame("FSM"),
        vmin=vmin,
        vmax=vmax,
        cmap=INTAKE_MEASURES[measure]["cmap"],
        cbar=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_ylabel("")


def draw_columns(
    fig: Figure,
    results: pd.DataFrame,
    schools: pd.DataFrame,
    modes: pd.DataFrame,
    fsm: pd.Series,
    parameters: list[str],
    measure: str = "dissimilarity",
) -> None:
    """Fill `fig` with a column per parameter: the index on top, then who the
    matching leaves unplaced, then each school's intake with routes and
    without, the FSM strip closing both intake rows, and the change in travel
    mode the routes induce at the foot.

    Every panel of a column shares one x-axis, drawn once at the foot, so the
    column reads top to bottom at a value. Every line row shares one y-axis
    and the intake panels one colour scale, so effects compare across
    columns. The dissimilarity scale is centred on 0 and reaches the largest
    term of any parameter or of the FSM strip, so every figure drawn from
    the same frames shares it too. Schools are ordered by their mean
    unrouted value, highest first.

    Args:
        fig (Figure): Figure to draw on, using constrained layout.

        results (pd.DataFrame): The scenario rows `sweep` returns.

        schools (pd.DataFrame): The school rows `sweep` returns.

        modes (pd.DataFrame): The mode rows `sweep` returns.

        fsm (pd.Series): As returned by `fsm_term` or `fsm_share`, whichever
        `measure` is.

        parameters (list[str]): Keys of GRID, one per column.

        measure (str, optional): A key of INTAKE_MEASURES, what the intake
        panels and the FSM strip show. Defaults to "dissimilarity".

    Raises:
        ValueError: If a frame carries a value its parameter does not sweep,
        which the panels would otherwise drop without saying so.
    """
    for parameter in parameters:
        swept = set(positions(parameter).index)
        for frame in (results, schools, modes):
            stray = set(frame.loc[frame["parameter"] == parameter, "value"]) - swept
            if stray:
                raise ValueError(
                    f"{parameter} carries values GRID does not sweep: {sorted(stray)}"
                )

    schools = schools.assign(
        share=schools["disadvantaged"] / (schools["disadvantaged"] + schools["other"])
    )
    shown = INTAKE_MEASURES[measure]["column"]
    unrouted = schools[schools["scenario"] == "without routes"]
    order = unrouted.groupby("school")[shown].mean().sort_values(ascending=False)
    if measure == "share":
        limits = (0.0, 1.0)
    else:
        means = schools.groupby(["parameter", "value", "scenario", "school"])[shown]
        bound = max(means.mean().abs().max(), fsm.abs().max())
        limits = (-bound, bound)
    change = mode_change(modes)
    unassigned = unassigned_rows(results)

    # The FSM strip takes a narrow last column of its own, so the intake
    # panels stay in the columns of the line panels above them.
    grid = fig.subplots(
        5,
        len(parameters) + 1,
        squeeze=False,
        width_ratios=[*[1] * len(parameters), 0.12],
    )
    axes, strips = grid[:, :-1], grid[:, -1]
    shared_rows = (0, 1, 4)
    for row in (axes[i] for i in shared_rows):
        for ax in row[1:]:
            ax.sharey(row[0])
    for column in axes.T:
        for ax in column[1:]:
            ax.sharex(column[0])
    for column, parameter in zip(axes.T, parameters):
        plot_parameter(
            column[0], results, parameter, "dissimilarity", "scenario", COLOURS
        )
        plot_stack(column[1], unassigned, parameter)
        for ax, scenario in zip(column[2:4], COLOURS):
            plot_intake(ax, schools, parameter, scenario, order.index, measure, limits)
        plot_parameter(column[4], change, parameter, "change", "mode", MODE_COLOURS)
        column[4].axhline(0, color="#898781", linewidth=1)
        # Last, so they replace the ticks the heatmaps set as they were drawn:
        # a shared axis carries one locator, so the foot sets the whole column.
        at = positions(parameter)
        column[0].set_xlim(0, len(at))
        column[4].set_xticks(
            at.to_numpy(),
            [f"{value:g}" for value in at.index],
            rotation=45,
            ha="right",
            fontsize="small",
        )
        for ax in column[:-1]:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
    axes[0, 0].set_ylabel("Dissimilarity index")
    axes[1, 0].set_ylabel("Students left unassigned")
    axes[4, 0].set_ylabel("Change in students with routes")
    for row in shared_rows:
        strips[row].axis("off")
    for ax in strips[2:4]:
        plot_fsm(ax, fsm, order.index, measure, limits)
    # Labels once per row and per column.
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    # The lines of the scenario panels and the patches the stacks are built
    # from share the top band: the stacks take their hue from the scenario
    # and their tone from the group.
    fig.legend(
        handles=legend_handles(COLOURS)
        + [Patch(color=colour, label=label) for label, colour in GROUP_COLOURS.items()],
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    fig.legend(
        handles=legend_handles(MODE_COLOURS),
        loc="outside lower center",
        ncol=len(MODE_COLOURS) + 1,
        frameon=False,
    )
    fig.colorbar(
        axes[2, 0].collections[0],
        ax=grid[2:4].ravel().tolist(),
        label=INTAKE_MEASURES[measure]["label"],
        fraction=0.04,
        pad=0.02,
    )


def plot(
    results: pd.DataFrame,
    schools: pd.DataFrame,
    modes: pd.DataFrame,
    fsm: pd.Series,
    measure: str = "dissimilarity",
) -> None:
    """Plot every parameter on its own and all of them side by side.

    Each parameter is written to SWEEP_DIR as its own PNG, and the matrix of
    every column to PLOT_PNG.

    Args:
        results (pd.DataFrame): The scenario rows `sweep` returns.

        schools (pd.DataFrame): The school rows `sweep` returns.

        modes (pd.DataFrame): The mode rows `sweep` returns.

        fsm (pd.Series): As returned by `fsm_term` or `fsm_share`, whichever
        `measure` is.

        measure (str, optional): Passed to `draw_columns`. Defaults to
        "dissimilarity".
    """
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for parameter in GRID:
        # A bare Figure draws without a display backend, which pyplot would need.
        fig = Figure(figsize=(9, 19), layout="constrained")
        draw_columns(fig, results, schools, modes, fsm, [parameter], measure)
        fig.savefig(SWEEP_DIR / f"{parameter}.png", dpi=200)

    fig = Figure(figsize=(4.5 * len(GRID), 20), layout="constrained")
    draw_columns(fig, results, schools, modes, fsm, list(GRID), measure)
    fig.savefig(PLOT_PNG, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep each parameter of the secondary matching and score it "
        "with and without routes over the same fresh samples."
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes to score cells in; bounded by memory, see `sweep`",
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
        help="round route seats up, so every route keeps one, rather than to "
        "the nearest seat, which leaves a route rounding to none unbuilt",
    )
    parser.add_argument(
        "--intake",
        choices=list(INTAKE_MEASURES),
        default="dissimilarity",
        help="what the intake panels show of each school: its term of the "
        "dissimilarity index, or the disadvantaged share of its intake",
    )
    args = parser.parse_args()

    areas = load_areas()
    _, secondary_schools = load_schools()
    samples = list(student_samples(areas, cohort_sizes(areas, "secondary"), args.seeds))
    shares = load_nts_mode_shares(args.years)

    results, schools, modes = sweep(
        samples,
        areas,
        secondary_schools,
        shares,
        args.circuity,
        args.workers,
        args.round_up_seats,
    )
    summary = results.pivot_table(
        index=["parameter", "value"],
        columns="scenario",
        values="dissimilarity",
        aggfunc="mean",
    )
    print(summary.loc[list(GRID)].round(3))
    change = mode_change(modes).pivot_table(
        index=["parameter", "value"], columns="mode", values="change", aggfunc="mean"
    )
    print("Change in students per mode with routes:")
    print(change.loc[list(GRID), MODES].round(1))
    unassigned = unassigned_rows(results).pivot_table(
        index=["parameter", "value"],
        columns=["group", "scenario"],
        values="unassigned",
        aggfunc="mean",
    )
    print("Students of each group left unassigned:")
    # Four columns under a two-level header fold at the default width.
    with pd.option_context("display.width", 100):
        print(unassigned.loc[list(GRID)].round(1))
    fsm = {"dissimilarity": fsm_term, "share": fsm_share}[args.intake]
    plot(results, schools, modes, fsm(secondary_schools), args.intake)
    print(
        f"Wrote {RESULTS_CSV}, {SCHOOLS_CSV}, {MODES_CSV} and the plots in {SWEEP_DIR}."
    )


if __name__ == "__main__":
    main()
