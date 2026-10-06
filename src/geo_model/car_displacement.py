"""Estimate the change in travel mode that routed matching induces.

The routeless matching is taken to reproduce the existing statistics: a
student seated without a route travels by the National Travel Survey's mode
split for school trips of their length (NTS0614a, aged 11 to 16), so the
count of every mode is an expected value and the only randomness is the
student sample per seed. A student seated with a route takes it with
certainty, counted as a mode of its own, so the change in each existing mode
shows where routed students are drawn from. Trip length is the straight-line
distance to the seated school, scaled by a circuity factor for the road
distance the survey records, and an unseated student makes no trip. The
counting itself is `utils.expected_modes`, which `dissimilarity.score_sample`
applies to every matching it scores, so the parameter sweep in `__main__`
carries the same estimate.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from geo_model.__main__ import COLOURS
from geo_model.build_prefs import cohort_sizes
from geo_model.build_routes import DISADVANTAGE, route_network
from geo_model.dissimilarity import (
    DEFAULTS,
    N_SEEDS,
    score_sample,
    student_samples,
)
from geo_model.load_data import NTS_YEARS, load_nts_mode_shares, load_schools
from geo_model.utils import (
    CIRCUITY,
    MODES,
    add_region_arguments,
    disadvantaged_group,
    mode_change,
    region_dir,
    regions,
)

# Written to each region's folder.
RESULTS_CSV = "car_displacement.csv"
PLOT_PNG = "car_displacement.png"


def run(
    n_seeds: int,
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    out_dir: Path,
    years: list[int] = NTS_YEARS,
    circuity: float = CIRCUITY,
) -> pd.DataFrame:
    """Count the modes of the secondary matching with and without routes over fresh samples.

    Args:
        n_seeds (int): Number of student samples to draw.

        areas (gpd.GeoDataFrame): The region's districts, as `route_network`
        takes them.

        secondary_schools (gpd.GeoDataFrame): The region's schools, as
        `score_sample` takes them.

        out_dir (Path): Folder the rows are written to.

        years (list[int], optional): Passed to `load_nts_mode_shares`.
        Defaults to NTS_YEARS.

        circuity (float, optional): Passed to `score_sample`. Defaults to
        CIRCUITY.

    Returns:
        pd.DataFrame: One row per (seed, scenario, mode), with columns "seed",
        "scenario", "mode" and "students" (expected). Also written to
        RESULTS_CSV in `out_dir`.
    """
    sizes = cohort_sizes(areas, "secondary")
    routes = route_network(
        areas,
        secondary_schools[["Easting", "Northing"]].to_numpy(),
        secondary_schools["P8MEA"].to_numpy(),
        secondary_schools["PlacesOffered"].to_numpy(),
        sizes,
    )
    shares = load_nts_mode_shares(years)

    rows = []
    for seed, (student_xy, student_lsoa, drawn) in enumerate(
        student_samples(areas, sizes, n_seeds)
    ):
        _, _, mode_rows = score_sample(
            student_xy,
            student_lsoa,
            areas,
            secondary_schools,
            routes,
            disadvantaged_group(
                student_lsoa, drawn, areas, DEFAULTS.decile, DISADVANTAGE
            ),
            shares,
            circuity=circuity,
        )
        rows += [{"seed": seed, **r} for r in mode_rows]
        print(f"Seed {seed + 1} of {n_seeds}: {len(student_xy)} students.")

    results = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / RESULTS_CSV, index=False)
    return results


def plot(results: pd.DataFrame, path: Path) -> None:
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
        hue_order=list(COLOURS),
        palette=COLOURS,
        errorbar=("pi", 100),
        ax=modes,
    )
    modes.set_title("Expected mode")
    modes.set_ylabel("Students")
    modes.legend(frameon=False)
    sns.boxplot(
        change, x="mode", y="change", order=MODES, color="#d9dde3", width=0.5, ax=shift
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

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the change in travel mode the routed secondary "
        "matching induces, from NTS mode shares by trip length."
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
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
    add_region_arguments(parser)
    args = parser.parse_args()

    for las, areas in regions(args.la, args.merge):
        out_dir = region_dir(las)
        _, secondary_schools = load_schools(las)
        results = run(
            args.seeds, areas, secondary_schools, out_dir, args.years, args.circuity
        )
        summary = results.pivot_table(
            index="mode", columns="scenario", values="students"
        )
        summary = summary.reindex(MODES)
        summary["change"] = summary["with routes"] - summary["without routes"]
        summary["share with"] = summary["with routes"] / summary["with routes"].sum()
        summary["share without"] = (
            summary["without routes"] / summary["without routes"].sum()
        )
        print(" + ".join(las) + ", mean over seeds:")
        print(summary.round(3))
        plot(results, out_dir / PLOT_PNG)
        print(f"Wrote {RESULTS_CSV} and {PLOT_PNG} in {out_dir}.")


if __name__ == "__main__":
    main()
