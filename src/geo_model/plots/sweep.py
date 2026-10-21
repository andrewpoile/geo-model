from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from geo_model.analysis.sweep import sweep_tests
from geo_model.measures.summaries import unassigned_rows
from geo_model.measures.travel import mode_change
from geo_model.model.settings import DEFAULTS, SCENARIOS, Settings
from geo_model.output import save
from geo_model.plots.style import (
    AXIS_LABELS,
    COLOURS,
    DEFAULT_LINE,
    GROUP_COLOURS,
    INTAKE_MEASURES,
    MODE_COLOURS,
    SIGNIFICANCE,
    SIGNIFICANCE_KEY,
)


def legend_handles(colours: dict[str, str]) -> list[Line2D]:
    """Proxy artists for the lines in `colours` and the default marker."""
    return [
        Line2D([], [], color=colour, marker="o", linewidth=2, label=label)
        for label, colour in colours.items()
    ] + [Line2D([], [], label="default", **DEFAULT_LINE)]


def positions(grid: dict[str, list[Settings]], parameter: str) -> pd.Series:
    """The x position of every value `parameter` sweeps, indexed by value.

    Every panel of a column is drawn against these positions rather than
    against the values themselves, so the panels that can only be categorical
    — the stacked bars and the intake heatmaps — share one axis with the line
    panels. The positions are the cell centres `sns.heatmap` draws its columns
    at, so the bars and the lines land on the heatmap's own grid.

    Args:
        grid (dict[str, list[Settings]]): The cells every parameter is swept
        over, as GRID holds them.

        parameter (str): A key of `grid`.

    Returns:
        pd.Series: The position of every value the parameter sweeps, in the
        order `grid` holds them.
    """
    values = [getattr(settings, parameter) for settings in grid[parameter]]
    return pd.Series(np.arange(len(values)) + 0.5, index=values)


def plot_parameter(
    ax: Axes,
    rows: pd.DataFrame,
    grid: dict[str, list[Settings]],
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

        grid (dict[str, list[Settings]]): The cells every parameter is swept
        over, as GRID holds them.

        parameter (str): A key of `grid`.

        y (str): Column to draw.

        hue (str): Column giving one line per level.

        colours (dict[str, str]): Colour of every level of `hue`, in the
        order the lines are drawn.
    """
    at = positions(grid, parameter)
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


def stars(p: float) -> str:
    """The stars of SIGNIFICANCE `p` reaches, "ns" short of every level, and
    "–" where the test is undefined."""
    if np.isnan(p):
        return "–"
    return next((mark for level, mark in SIGNIFICANCE if p < level), "ns")


def plot_significance(
    ax: Axes, tests: pd.DataFrame, grid: dict[str, list[Settings]], parameter: str
) -> None:
    """Mark every value of one parameter's index panel, above the panel,
    with the stars of its Holm-corrected p-value.

    Args:
        ax (Axes): The parameter's index panel.

        tests (pd.DataFrame): As returned by `sweep_tests`.

        grid (dict[str, list[Settings]]): The cells every parameter is swept
        over, as GRID holds them.

        parameter (str): A key of `grid`.
    """
    corrected = tests["p_holm"].xs(parameter)
    for x, p in zip(positions(grid, parameter)[corrected.index], corrected):
        ax.text(
            x,
            1,
            stars(p),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize="small",
        )


def plot_stack(
    ax: Axes, rows: pd.DataFrame, grid: dict[str, list[Settings]], parameter: str
) -> None:
    """Draw one parameter's unassigned panel: a bar per scenario at every
    value, stacked by the group the unplaced students belong to.

    Args:
        ax (Axes): Axis to draw on.

        rows (pd.DataFrame): Rows `unassigned_rows` returns.

        grid (dict[str, list[Settings]]): The cells every parameter is swept
        over, as GRID holds them.

        parameter (str): A key of `grid`.
    """
    at = positions(grid, parameter)
    cell = rows[rows["parameter"] == parameter]
    stack = cell.pivot_table(
        index="value", columns="series", values="unassigned", aggfunc="mean"
    ).reindex(at.index)
    width = 0.38
    for offset, scenario in zip((-width / 2, width / 2), SCENARIOS):
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


def plot_intake(
    ax: Axes,
    schools: pd.DataFrame,
    grid: dict[str, list[Settings]],
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

        grid (dict[str, list[Settings]]): The cells every parameter is swept
        over, as GRID holds them.

        parameter (str): A key of `grid`.

        scenario (str): One of SCENARIOS.

        order (pd.Index): School names, top to bottom.

        measure (str): A key of INTAKE_MEASURES.

        limits (tuple[float, float]): The values the colour scale ends at.
    """
    cell = schools[
        (schools["parameter"] == parameter) & (schools["scenario"] == scenario)
    ]
    at = positions(grid, parameter)
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
    grid: dict[str, list[Settings]],
    parameters: list[str],
    measure: str = "dissimilarity",
) -> None:
    """Fill `fig` with a column per parameter: the index on top, every value
    starred by its test from `sweep_tests`, then who the matching leaves
    unplaced, then each school's intake with routes and without, the FSM
    strip closing both intake rows, and the change in travel mode the routes
    induce at the foot.

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

        grid (dict[str, list[Settings]]): The cells every parameter is swept
        over, as GRID holds them.

        parameters (list[str]): Keys of `grid`, one per column.

        measure (str, optional): A key of INTAKE_MEASURES, what the intake
        panels and the FSM strip show. Defaults to "dissimilarity".

    Raises:
        ValueError: If a frame carries a value its parameter does not sweep,
        which the panels would otherwise drop without saying so.
    """
    for parameter in parameters:
        swept = set(positions(grid, parameter).index)
        for frame in (results, schools, modes):
            stray = set(frame.loc[frame["parameter"] == parameter, "value"]) - swept
            if stray:
                raise ValueError(
                    f"{parameter} carries values the grid does not sweep: {sorted(stray)}"
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
    tests = sweep_tests(results)

    # The FSM strip takes a narrow last column of its own, so the intake
    # panels stay in the columns of the line panels above them.
    panels = fig.subplots(
        5,
        len(parameters) + 1,
        squeeze=False,
        width_ratios=[*[1] * len(parameters), 0.12],
    )
    axes, strips = panels[:, :-1], panels[:, -1]
    shared_rows = (0, 1, 4)
    for row in (axes[i] for i in shared_rows):
        for ax in row[1:]:
            ax.sharey(row[0])
    for column in axes.T:
        for ax in column[1:]:
            ax.sharex(column[0])
    for column, parameter in zip(axes.T, parameters):
        plot_parameter(
            column[0], results, grid, parameter, "dissimilarity", "scenario", COLOURS
        )
        plot_significance(column[0], tests, grid, parameter)
        plot_stack(column[1], unassigned, grid, parameter)
        for ax, scenario in zip(column[2:4], SCENARIOS):
            plot_intake(
                ax, schools, grid, parameter, scenario, order.index, measure, limits
            )
        plot_parameter(
            column[4], change, grid, parameter, "change", "mode", MODE_COLOURS
        )
        column[4].axhline(0, color="#898781", linewidth=1)
        # Last, so they replace the ticks the heatmaps set as they were drawn:
        # a shared axis carries one locator, so the foot sets the whole column.
        at = positions(grid, parameter)
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
    # and their tone from the group. The stars' key heads it.
    fig.legend(
        handles=legend_handles(COLOURS)
        + [Patch(color=colour, label=label) for label, colour in GROUP_COLOURS.items()],
        loc="outside upper center",
        ncol=4,
        frameon=False,
        title=SIGNIFICANCE_KEY,
        title_fontsize="small",
    )
    fig.legend(
        handles=legend_handles(MODE_COLOURS),
        loc="outside lower center",
        ncol=len(MODE_COLOURS) + 1,
        frameon=False,
    )
    fig.colorbar(
        axes[2, 0].collections[0],
        ax=panels[2:4].ravel().tolist(),
        label=INTAKE_MEASURES[measure]["label"],
        fraction=0.04,
        pad=0.02,
    )


def plot_sweep(
    results: pd.DataFrame,
    schools: pd.DataFrame,
    modes: pd.DataFrame,
    fsm: pd.Series,
    grid: dict[str, list[Settings]],
    parameters: list[str],
    path: Path,
    measure: str = "dissimilarity",
) -> None:
    """Plot every parameter on its own and all of them side by side.

    Each parameter is written as its own PNG beside `path`, named for the
    parameter, and the matrix of every column to `path`.

    Args:
        results (pd.DataFrame): The scenario rows `sweep` returns.

        schools (pd.DataFrame): The school rows `sweep` returns.

        modes (pd.DataFrame): The mode rows `sweep` returns.

        fsm (pd.Series): As returned by `fsm_term` or `fsm_share`, whichever
        `measure` is.

        grid (dict[str, list[Settings]]): The cells every parameter is swept
        over, as GRID holds them.

        parameters (list[str]): Keys of `grid` the frames sweep, as `swept`
        names them.

        path (Path): PNG the matrix is written to.

        measure (str, optional): Passed to `draw_columns`. Defaults to
        "dissimilarity".
    """
    for parameter in parameters:
        # A bare Figure draws without a display backend, which pyplot would need.
        fig = Figure(figsize=(9, 19), layout="constrained")
        draw_columns(fig, results, schools, modes, fsm, grid, [parameter], measure)
        save(fig, path.parent / f"{parameter}.png")

    fig = Figure(figsize=(4.5 * len(parameters), 20), layout="constrained")
    draw_columns(fig, results, schools, modes, fsm, grid, parameters, measure)
    save(fig, path)
