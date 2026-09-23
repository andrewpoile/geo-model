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

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from geo_model.__main__ import COLOURS
from geo_model.build_prefs import cohort_sizes
from geo_model.build_routes import route_network
from geo_model.dissimilarity import N_SEEDS, score_sample, student_samples
from geo_model.load_data import (
    NTS_YEARS,
    load_areas,
    load_nts_mode_shares,
    load_schools,
)
from geo_model.utils import CIRCUITY, MODES, mode_change

RESULTS_CSV = Path("temp/car_displacement.csv")
PLOT_PNG = Path("temp/car_displacement.png")


def run(
    n_seeds: int, years: list[int] = NTS_YEARS, circuity: float = CIRCUITY
) -> pd.DataFrame:
    """Count the modes of the secondary matching with and without routes over fresh samples.

    Args:
        n_seeds (int): Number of student samples to draw.

        years (list[int], optional): Passed to `load_nts_mode_shares`.
        Defaults to NTS_YEARS.

        circuity (float, optional): Passed to `score_sample`. Defaults to
        CIRCUITY.

    Returns:
        pd.DataFrame: One row per (seed, scenario, mode), with columns "seed",
        "scenario", "mode" and "students" (expected). Also written to
        RESULTS_CSV.
    """
    areas = load_areas()
    _, secondary_schools = load_schools()
    sizes = cohort_sizes(areas, "secondary")
    routes = route_network(
        areas,
        secondary_schools[["Easting", "Northing"]].to_numpy(),
        secondary_schools["P8MEA"].to_numpy(),
        secondary_schools["PAN"].to_numpy(),
        sizes,
    )
    shares = load_nts_mode_shares(years)

    rows = []
    for seed, (student_xy, student_lsoa) in enumerate(
        student_samples(areas, sizes, n_seeds)
    ):
        _, _, mode_rows = score_sample(
            student_xy,
            student_lsoa,
            areas,
            secondary_schools,
            routes,
            shares,
            circuity=circuity,
        )
        rows += [{"seed": seed, **r} for r in mode_rows]
        print(f"Seed {seed + 1} of {n_seeds}: {len(student_xy)} students.")

    results = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    return results


def plot(results: pd.DataFrame) -> None:
    """Plot the expected modes of both scenarios and the change per mode, to PLOT_PNG.

    Args:
        results (pd.DataFrame): As returned by `run`.
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

    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=200)


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
    args = parser.parse_args()

    results = run(args.seeds, args.years, args.circuity)
    summary = results.pivot_table(index="mode", columns="scenario", values="students")
    summary = summary.reindex(MODES)
    summary["change"] = summary["with routes"] - summary["without routes"]
    summary["share with"] = summary["with routes"] / summary["with routes"].sum()
    summary["share without"] = (
        summary["without routes"] / summary["without routes"].sum()
    )
    print("Mean over seeds:")
    print(summary.round(3))
    plot(results)
    print(f"Wrote {RESULTS_CSV} and {PLOT_PNG}.")


if __name__ == "__main__":
    main()
