from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from geo_model.analysis.search import OBJECTIVES
from geo_model.output import save
from geo_model.plots.style import COLOURS, INK


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
