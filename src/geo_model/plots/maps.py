from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from geo_model.output import save
from geo_model.plots.style import DECILE_CMAP, INK, LINK_COLOURS, P8_CMAP, SCALE_BAR


def draw_map(
    fig: Figure,
    areas: gpd.GeoDataFrame,
    secondary_schools: pd.DataFrame,
    student_xy: np.ndarray,
    matchings: dict[str, np.ndarray],
) -> None:
    """Fill `fig` with a map per scenario: the districts by IDACI decile, a
    link from every seated student to their school, and the schools by
    Progress 8 on top.

    A routed student's link takes a colour of its own, and a student the
    matching leaves unassigned is marked by a cross, having no link. Every
    panel shares both colour scales, so the panels compare.

    Args:
        fig (Figure): Figure to draw on, using constrained layout.

        areas (gpd.GeoDataFrame): Districts carrying "IDACI Decile" and a
        polygon geometry.

        secondary_schools (pd.DataFrame): Schools carrying "Easting",
        "Northing" and "P8MEA", in the order the matchings index them.

        student_xy (np.ndarray): Student coordinates, shape (n_students, 2),
        in the CRS of `areas`.

        matchings (dict[str, np.ndarray]): The matching of each scenario, as
        `settings_matchings` returns them.
    """
    school_xy = secondary_schools[["Easting", "Northing"]].to_numpy()
    p8 = secondary_schools["P8MEA"].to_numpy()
    p8_norm = Normalize(-np.abs(p8).max(), np.abs(p8).max())
    axes = fig.subplots(1, len(matchings), squeeze=False)[0]
    for ax, (scenario, matching) in zip(axes, matchings.items()):
        # Half-step limits give every whole decile its own step of the scale.
        areas.plot(
            ax=ax,
            column="IDACI Decile",
            cmap=DECILE_CMAP,
            vmin=0.5,
            vmax=10.5,
            edgecolor="white",
            linewidth=0.3,
            label="districts",
        )
        school, route = matching[:, 0], matching[:, 1]
        # The routed links are the fewer, so they are drawn over the rest.
        for label, linked, width, alpha in (
            ("without a route", (school >= 0) & (route < 0), 0.3, 0.4),
            ("on a route", route >= 0, 0.6, 0.7),
        ):
            # One (2, 2) segment per student, from their home to their school.
            segments = list(
                np.stack((student_xy[linked], school_xy[school[linked]]), axis=1)
            )
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=LINK_COLOURS[label],
                    linewidths=width,
                    alpha=alpha,
                    label=label,
                )
            )
        ax.scatter(*student_xy[school >= 0].T, s=1, color=INK, label="seated")
        ax.scatter(
            *student_xy[school < 0].T,
            s=12,
            marker="x",
            color=INK,
            linewidths=0.8,
            label="unassigned",
        )
        schools = ax.scatter(
            *school_xy.T,
            c=p8,
            cmap=P8_CMAP,
            norm=p8_norm,
            s=90,
            edgecolors=INK,
            linewidths=1,
            zorder=3,
            label="schools",
        )
        ax.set_title(scenario)
        ax.set_axis_off()
        draw_scale(ax)

    fig.colorbar(
        axes[0].collections[0],
        ax=axes.tolist(),
        ticks=range(1, 11),
        label="LSOA IDACI decile (1 is most deprived)",
    )
    fig.colorbar(schools, ax=axes.tolist(), label="School Progress 8 (P8MEA)")
    fig.legend(
        handles=[
            Line2D([], [], color=INK, marker="o", markersize=3, linestyle="none"),
            Line2D([], [], color=INK, marker="x", markersize=6, linestyle="none"),
        ]
        + [
            Line2D([], [], color=colour, linewidth=2)
            for colour in LINK_COLOURS.values()
        ],
        labels=["student", "student left unassigned"]
        + [f"link to school {label}" for label in LINK_COLOURS],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )


def draw_scale(ax: Axes) -> None:
    """Draw SCALE_BAR in the lower left of a map panel, its first unit ticked
    above the line and its second below, both from the same origin.

    Lengths are British National Grid metres, the metres every distance in the
    model is measured in; over Southampton a grid metre is 0.9996 of a metre
    on the ground. The bar is placed by the panel's limits and leaves them as
    they are, so it is drawn last.

    Args:
        ax (Axes): A map panel, drawn in metres at equal aspect.
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    x, y = x0 + 0.03 * (x1 - x0), y0 + 0.07 * (y1 - y0)
    tick = 0.012 * (y1 - y0)
    for (unit, (metres, length)), side in zip(SCALE_BAR.items(), (1, -1)):
        at = x + metres * np.arange(length + 1)
        ax.add_collection(
            LineCollection(
                [[(x, y), (at[-1], y)], *[[(t, y), (t, y + side * tick)] for t in at]],
                colors=INK,
                linewidths=1,
                label=unit,
            ),
            autolim=False,
        )
        for i, t in enumerate(at):
            ax.text(
                t,
                y + side * 2 * tick,
                f"{i} {unit}" if i == length else str(i),
                ha="center",
                va="bottom" if side > 0 else "top",
                fontsize="small",
                color=INK,
            )


def reflect(
    areas: gpd.GeoDataFrame, secondary_schools: pd.DataFrame, student_xy: np.ndarray
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, np.ndarray]:
    """A map's districts, schools and students reflected in the line easting
    = northing, so the outline the map draws is not the region's.

    The reflection keeps every distance, so the scale bar drawn over it stays
    true. It is for drawing alone: the model samples and matches on the
    coordinates as they are.

    Args:
        areas (gpd.GeoDataFrame): Districts, their active geometry the one
        drawn.

        secondary_schools (pd.DataFrame): Schools carrying "Easting" and
        "Northing".

        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

    Returns:
        tuple[gpd.GeoDataFrame, pd.DataFrame, np.ndarray]: The three with
        every easting and northing swapped.
    """
    return (
        areas.set_geometry(areas.geometry.affine_transform([0, 1, 1, 0, 0, 0])),
        secondary_schools.assign(
            Easting=secondary_schools["Northing"],
            Northing=secondary_schools["Easting"],
        ),
        student_xy[:, ::-1],
    )


def plot_map(
    areas: gpd.GeoDataFrame,
    secondary_schools: pd.DataFrame,
    student_xy: np.ndarray,
    matchings: dict[str, np.ndarray],
    path: Path,
    reflected: bool = False,
) -> None:
    """Map `matchings` with `draw_map`, to `path`.

    Args:
        areas (gpd.GeoDataFrame): Passed to `draw_map`.

        secondary_schools (pd.DataFrame): Passed to `draw_map`.

        student_xy (np.ndarray): Passed to `draw_map`.

        matchings (dict[str, np.ndarray]): Passed to `draw_map`.

        path (Path): PNG to write.

        reflected (bool, optional): Draw the map reflected, as `reflect`
        gives it. Defaults to False.
    """
    if reflected:
        areas, secondary_schools, student_xy = reflect(
            areas, secondary_schools, student_xy
        )
    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(16, 8), layout="constrained")
    draw_map(fig, areas, secondary_schools, student_xy, matchings)
    save(fig, path)
