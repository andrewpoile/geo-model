from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from geo_model.measures.segregation import gini, lorenz_curve
from geo_model.measures.statistics import routes_t_test, t_test_label
from geo_model.measures.summaries import (
    local_acceptance,
    local_difference,
    route_utilisation,
    unassigned_rows,
)
from geo_model.measures.travel import MODES, mode_change
from geo_model.model.settings import DEFAULTS, DISADVANTAGE, SCENARIOS
from geo_model.output import save
from geo_model.plots.style import COLOURS, GROUP_COLOURS, INK, INTAKE_MEASURES

# Past this many routes the utilisation bars go unlabelled, too narrow to name.
MAX_LABELLED_ROUTES = 40


def plot_dissimilarity(results: pd.DataFrame, path: Path) -> None:
    """Box-plot the index per scenario, one point per seed, to `path`, titled
    with the paired t-test of the index with routes against without.

    The y-axis is fitted to the indices drawn, so it need not start at 0.

    Args:
        results (pd.DataFrame): As returned by `run`, or the scenario rows of
        one setting as `score_settings` returns them.

        path (Path): PNG to write.
    """
    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.set_title(t_test_label(routes_t_test(results)), fontsize="small")
    # Every seed is drawn as a point, so the box draws no outliers of its own.
    sns.boxplot(
        results,
        x="scenario",
        y="dissimilarity",
        color="#d9dde3",
        width=0.5,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(results, x="scenario", y="dissimilarity", color="#1f2933", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Dissimilarity index")
    ax.yaxis.grid(True, color="#e5e7eb")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()

    save(fig, path)


def plot_modes(results: pd.DataFrame, path: Path) -> None:
    """Plot the expected modes of both scenarios and the change per mode, to `path`.

    Args:
        results (pd.DataFrame): As returned by `run`, or the mode rows of one
        setting as `score_settings` returns them.

        path (Path): PNG to write.
    """
    change = mode_change(results)

    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    modes, shift = fig.subplots(1, 2)
    sns.barplot(
        results,
        x="mode",
        y="students",
        hue="scenario",
        hue_order=list(SCENARIOS),
        palette=COLOURS,
        errorbar=("pi", 100),
        ax=modes,
    )
    modes.set_title("Expected mode")
    modes.set_ylabel("Students")
    modes.legend(frameon=False)
    # Every seed is drawn as a point, so the box draws no outliers of its own.
    sns.boxplot(
        change,
        x="mode",
        y="change",
        order=MODES,
        color="#d9dde3",
        width=0.5,
        showfliers=False,
        ax=shift,
    )
    sns.stripplot(change, x="mode", y="change", order=MODES, color="#1f2933", ax=shift)
    shift.axhline(0, color="#898781", linewidth=1)
    shift.set_title("Change with routes")
    shift.set_ylabel("Students, with routes minus without")
    for ax in (modes, shift):
        ax.set_xlabel("")
        ax.yaxis.grid(True, color="#e5e7eb")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    save(fig, path)


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
        .reindex(list(SCENARIOS))
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
        intake[list(SCENARIOS)]
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
    ax.axvline(len(SCENARIOS), color="white", linewidth=4)
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
    `plot_modes` draws the change per mode. Schools are ordered
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


def draw_lorenz(
    fig: Figure, schools: pd.DataFrame, disadvantage: str = DISADVANTAGE
) -> None:
    """Fill `fig` with a panel per scenario: the Lorenz curve of disadvantaged
    against seated students over schools at one setting, a line per seed, and
    its Gini coefficient over the seeds.

    A student left unassigned belongs to no school's intake, so the curves
    count seated students alone. Both panels share both axes, so the panels
    compare.

    Args:
        fig (Figure): Figure to draw on, using constrained layout.

        schools (pd.DataFrame): The school rows of one setting, as
        `score_settings` returns them or `default_cell` picks them from
        `sweep`'s.

        disadvantage (str, optional): The choice of DISADVANTAGE_CHOICES the
        rows were scored under, which the axis label names. Under "decile"
        the setting's decile is taken to be DEFAULTS.decile. Defaults to
        DISADVANTAGE.
    """
    axes = fig.subplots(1, len(SCENARIOS), sharex=True, sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        colour = COLOURS[scenario]
        coefficients = []
        for seed, intake in schools[schools["scenario"] == scenario].groupby("seed"):
            x, y = lorenz_curve(intake["disadvantaged"], intake["other"])
            ax.plot(x, y, color=colour, linewidth=1, alpha=0.5, label=f"seed {seed}")
            coefficients.append(gini(x, y))
        ax.plot([0, 1], [0, 1], color="#898781", linewidth=1, label="equality")
        # The curve never rises above the diagonal, so the upper left is clear.
        ax.text(
            0.04,
            0.96,
            f"Gini {np.mean(coefficients):.3f} (mean)\n"
            f"seeds {min(coefficients):.3f} to {max(coefficients):.3f}, "
            f"n = {len(coefficients)}",
            transform=ax.transAxes,
            va="top",
            color=INK,
        )
        ax.set_title(scenario)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Cumulative share of seated students")
        ax.grid(True, color="#e1e0d9")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    group = (
        f"IDACI decile ≤ {DEFAULTS.decile}"
        if disadvantage == "decile"
        else "drawn from LSOA IDACI scores"
    )
    axes[0].set_ylabel(f"Cumulative share of disadvantaged students ({group})")


def plot_lorenz(
    schools: pd.DataFrame, path: Path, disadvantage: str = DISADVANTAGE
) -> None:
    """Plot the Lorenz curves with `draw_lorenz`, to `path`.

    Args:
        schools (pd.DataFrame): Passed to `draw_lorenz`.

        path (Path): PNG to write.

        disadvantage (str, optional): Passed to `draw_lorenz`. Defaults to
        DISADVANTAGE.
    """
    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(12, 6.5), layout="constrained")
    draw_lorenz(fig, schools, disadvantage)
    save(fig, path)
