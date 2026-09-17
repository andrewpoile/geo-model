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
from matplotlib.patches import Rectangle

from geo_model.build_prefs import cohort_sizes
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
from geo_model.utils import CIRCUITY, MODES, mode_change

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
    # The largest disadvantaged cohort is 40 students, so 40 seats never bind.
    "capacity": [replace(DEFAULTS, capacity=c) for c in (1, 2, 5, 10, 20, 30, 40)],
    "performance_weight": [
        replace(DEFAULTS, performance_weight=w / 10) for w in range(11)
    ],
    "route_discount": [replace(DEFAULTS, route_discount=d / 10) for d in range(11)],
}

AXIS_LABELS = {
    "decile": "Disadvantaged at or below IMD decile",
    "min_distance": "Minimum route distance (m)",
    "capacity": "Route capacity (seats)",
    "performance_weight": "Performance weight",
    "route_discount": "Route discount",
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
DEFAULT_LINE = {"color": "#898781", "linestyle": "--", "linewidth": 1}
# One hue, light to dark, for the share of a school's intake that is disadvantaged.
SHARE_CMAP = "Blues"


def sweep(
    samples: list[tuple[np.ndarray, np.ndarray]],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    circuity: float = CIRCUITY,
    workers: int = 1,
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
    sns.lineplot(
        rows[rows["parameter"] == parameter],
        x="value",
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
    ax.axvline(float(asdict(DEFAULTS)[parameter]), **DEFAULT_LINE)
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


def plot_intake(
    ax: Axes, schools: pd.DataFrame, parameter: str, scenario: str, order: pd.Index
) -> None:
    """Draw one scenario's intake panel: the disadvantaged share of each
    school's intake, averaged over seeds, as a gradient over the values.

    Args:
        ax (Axes): Axis to draw on.

        schools (pd.DataFrame): The school rows `sweep` returns, carrying a
        "share" column.

        parameter (str): A key of GRID.

        scenario (str): A key of COLOURS.

        order (pd.Index): School names, top to bottom.
    """
    cell = schools[
        (schools["parameter"] == parameter) & (schools["scenario"] == scenario)
    ]
    share = cell.pivot_table(index="school", columns="value", values="share")
    share = share.reindex(order)
    sns.heatmap(
        share,
        vmin=0,
        vmax=1,
        cmap=SHARE_CMAP,
        cbar=False,
        xticklabels=[f"{value:g}" for value in share.columns],
        ax=ax,
    )
    (default,) = np.flatnonzero(share.columns == asdict(DEFAULTS)[parameter])
    ax.add_patch(
        Rectangle((int(default), 0), 1, len(share), fill=False, **DEFAULT_LINE)
    )
    ax.set_xlabel(AXIS_LABELS[parameter])
    ax.set_ylabel(scenario)
    ax.tick_params(axis="y", rotation=0)


def plot_fsm(ax: Axes, fsm: pd.Series, order: pd.Index) -> None:
    """Draw the FSM strip: the register's share for each school, on the
    intake panels' scale, as the fixed point of comparison for both.

    Args:
        ax (Axes): Axis to draw on.

        fsm (pd.Series): As returned by `fsm_share`.

        order (pd.Index): School names, top to bottom.
    """
    sns.heatmap(
        fsm.reindex(order).to_frame("FSM"),
        vmin=0,
        vmax=1,
        cmap=SHARE_CMAP,
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
) -> None:
    """Fill `fig` with a column per parameter: the index on top, then each
    school's intake with routes and without, the FSM strip closing both
    intake rows, and the change in travel mode the routes induce at the foot.

    The index panels share one y-axis, the intake panels one colour scale
    and the mode panels one y-axis, so effects compare across columns.
    Schools are ordered by their unrouted share, most disadvantaged first.

    Args:
        fig (Figure): Figure to draw on, using constrained layout.

        results (pd.DataFrame): The scenario rows `sweep` returns.

        schools (pd.DataFrame): The school rows `sweep` returns.

        modes (pd.DataFrame): The mode rows `sweep` returns.

        fsm (pd.Series): As returned by `fsm_share`.

        parameters (list[str]): Keys of GRID, one per column.
    """
    schools = schools.assign(
        share=schools["disadvantaged"] / (schools["disadvantaged"] + schools["other"])
    )
    unrouted = schools[schools["scenario"] == "without routes"]
    order = unrouted.groupby("school")["share"].mean().sort_values(ascending=False)
    change = mode_change(modes)

    # The FSM strip takes a narrow last column of its own, so the intake
    # panels stay in the columns of the index panels above them.
    grid = fig.subplots(
        4,
        len(parameters) + 1,
        squeeze=False,
        width_ratios=[*[1] * len(parameters), 0.12],
    )
    axes, strips = grid[:, :-1], grid[:, -1]
    for row in (axes[0], axes[3]):
        for ax in row[1:]:
            ax.sharey(row[0])
    for column, parameter in zip(axes.T, parameters):
        plot_parameter(
            column[0], results, parameter, "dissimilarity", "scenario", COLOURS
        )
        for ax, scenario in zip(column[1:3], COLOURS):
            plot_intake(ax, schools, parameter, scenario, order.index)
        plot_parameter(column[3], change, parameter, "change", "mode", MODE_COLOURS)
        column[3].axhline(0, color="#898781", linewidth=1)
    axes[0, 0].set_ylabel("Dissimilarity index")
    axes[3, 0].set_ylabel("Change in students with routes")
    strips[0].axis("off")
    strips[3].axis("off")
    for ax in strips[1:3]:
        plot_fsm(ax, fsm, order.index)
    # Labels once per row and per column. The line panels keep their own
    # axes, since they are numeric where the intake panels are categorical.
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    for ax in axes[1:3].flat:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    fig.legend(
        handles=legend_handles(COLOURS),
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    fig.legend(
        handles=legend_handles(MODE_COLOURS),
        loc="outside lower center",
        ncol=len(MODE_COLOURS) + 1,
        frameon=False,
    )
    fig.colorbar(
        axes[1, 0].collections[0],
        ax=grid[1:3].ravel().tolist(),
        label="Disadvantaged share of intake",
        fraction=0.04,
        pad=0.02,
    )


def plot(
    results: pd.DataFrame, schools: pd.DataFrame, modes: pd.DataFrame, fsm: pd.Series
) -> None:
    """Plot every parameter on its own and all of them side by side.

    Each parameter is written to SWEEP_DIR as its own PNG, and the matrix of
    every column to PLOT_PNG.

    Args:
        results (pd.DataFrame): The scenario rows `sweep` returns.

        schools (pd.DataFrame): The school rows `sweep` returns.

        modes (pd.DataFrame): The mode rows `sweep` returns.

        fsm (pd.Series): As returned by `fsm_share`.
    """
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    for parameter in GRID:
        # A bare Figure draws without a display backend, which pyplot would need.
        fig = Figure(figsize=(9, 15), layout="constrained")
        draw_columns(fig, results, schools, modes, fsm, [parameter])
        fig.savefig(SWEEP_DIR / f"{parameter}.png", dpi=200)

    fig = Figure(figsize=(4.5 * len(GRID), 16), layout="constrained")
    draw_columns(fig, results, schools, modes, fsm, list(GRID))
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
    args = parser.parse_args()

    areas = load_areas()
    _, secondary_schools = load_schools()
    samples = list(student_samples(areas, cohort_sizes(areas, "secondary"), args.seeds))
    shares = load_nts_mode_shares(args.years)

    results, schools, modes = sweep(
        samples, areas, secondary_schools, shares, args.circuity, args.workers
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
    plot(results, schools, modes, fsm_share(secondary_schools))
    print(
        f"Wrote {RESULTS_CSV}, {SCHOOLS_CSV}, {MODES_CSV} and the plots in {SWEEP_DIR}."
    )


if __name__ == "__main__":
    main()
