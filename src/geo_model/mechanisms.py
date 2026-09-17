"""Score the mechanism variants against the paper's DAT on the same samples.

Two levers change who a route seats without changing the instance: a walking
threshold withholds routes from students who can walk (`utils.route_eligibility`
under either rule), and reserved seats let a routed student ride without
taking a seat from a routeless one (`utils.reserve_routes`). Each lever is
scored alone and together, at the model's defaults otherwise, and the
routeless matching, which no lever touches, is scored once as the baseline.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from geo_model.__main__ import MODE_COLOURS
from geo_model.build_prefs import cohort_sizes
from geo_model.dissimilarity import (
    DEFAULTS,
    N_SEEDS,
    Settings,
    score_settings,
    student_samples,
)
from geo_model.load_data import (
    NTS_YEARS,
    load_areas,
    load_nts_mode_shares,
    load_schools,
)
from geo_model.utils import CIRCUITY, MILE, MODES

RESULTS_CSV = Path("temp/mechanisms.csv")
MODES_CSV = Path("temp/mechanisms_modes.csv")
PLOT_PNG = Path("temp/mechanisms.png")

# Metres a student is taken to walk: the NTS band edge below which 86% of
# school trips are walked.
WALK_DISTANCE = MILE
BASELINE = "without routes"


def variants(walk_distance: float = WALK_DISTANCE) -> dict[str, Settings]:
    """The mechanism variants, each lever alone and together, by name.

    Args:
        walk_distance (float, optional): Threshold of the gated variants, in
        metres. Defaults to WALK_DISTANCE.

    Returns:
        dict[str, Settings]: The paper's mechanism first, under "current".
    """
    gated = {
        rule: replace(DEFAULTS, walk_distance=walk_distance, walk_rule=rule)
        for rule in ("routeless", "nearest")
    }
    return {
        "current": DEFAULTS,
        "gated (routeless)": gated["routeless"],
        "gated (nearest)": gated["nearest"],
        "reserved": replace(DEFAULTS, reserved=True),
        "gated (routeless) + reserved": replace(gated["routeless"], reserved=True),
        "gated (nearest) + reserved": replace(gated["nearest"], reserved=True),
    }


def run(
    n_seeds: int,
    walk_distance: float = WALK_DISTANCE,
    years: list[int] = NTS_YEARS,
    circuity: float = CIRCUITY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every variant's routed matching, and the routeless baseline, over fresh samples.

    Args:
        n_seeds (int): Number of student samples to draw.

        walk_distance (float, optional): Passed to `variants`. Defaults to
        WALK_DISTANCE.

        years (list[int], optional): Passed to `load_nts_mode_shares`.
        Defaults to NTS_YEARS.

        circuity (float, optional): Passed to `score_settings`. Defaults to
        CIRCUITY.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The scenario rows, one per (seed,
        variant), with columns "seed", "variant", "dissimilarity",
        "n_matched" and "n_unmatched", and the mode rows, one per (seed,
        variant, mode), with "seed", "variant", "mode" and "students"
        (expected). The baseline is the variant BASELINE of both. Also
        written to RESULTS_CSV and MODES_CSV.
    """
    areas = load_areas()
    _, secondary_schools = load_schools()
    samples = list(student_samples(areas, cohort_sizes(areas, "secondary"), n_seeds))
    shares = load_nts_mode_shares(years)

    results, modes = [], []
    for name, settings in variants(walk_distance).items():
        rows, _, mode_rows = score_settings(
            settings, samples, areas, secondary_schools, shares, circuity
        )
        rows = pd.DataFrame(rows).drop(columns="n_routes")
        for frame, sink in ((rows, results), (pd.DataFrame(mode_rows), modes)):
            routeless = frame[frame["scenario"] == BASELINE].assign(variant=BASELINE)
            # No lever reaches the routeless instance, so every variant scores
            # the same baseline. It is kept once, and a difference is a fault.
            if sink:
                pd.testing.assert_frame_equal(sink[0], routeless)
            else:
                sink.append(routeless)
            sink.append(frame[frame["scenario"] != BASELINE].assign(variant=name))
        print(f"Scored {name}.")

    results, modes = (
        pd.concat(frames, ignore_index=True).drop(columns="scenario")
        for frames in (results, modes)
    )
    results = results[["seed", "variant", "dissimilarity", "n_matched", "n_unmatched"]]
    modes = modes[["seed", "variant", "mode", "students"]]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    modes.to_csv(MODES_CSV, index=False)
    return results, modes


def summary(results: pd.DataFrame, modes: pd.DataFrame) -> pd.DataFrame:
    """The mean expected students per mode, the car and active shares and the
    index, one row per variant in the order scored.

    Args:
        results (pd.DataFrame): The scenario rows `run` returns.

        modes (pd.DataFrame): The mode rows `run` returns.

    Returns:
        pd.DataFrame: Indexed by variant, with a column per mode in MODES,
        "car share", "active share" and "dissimilarity".
    """
    order = modes["variant"].unique()
    table = modes.pivot_table(index="variant", columns="mode", values="students")
    table = table.reindex(index=order, columns=MODES)
    seated = table.sum(axis=1)
    table["car share"] = table["car"] / seated
    table["active share"] = (table["walk"] + table["cycle"]) / seated
    table["dissimilarity"] = results.groupby("variant")["dissimilarity"].mean()
    return table


def plot(results: pd.DataFrame, modes: pd.DataFrame) -> None:
    """Plot the expected modes and the index of every variant, to PLOT_PNG.

    Args:
        results (pd.DataFrame): The scenario rows `run` returns.

        modes (pd.DataFrame): The mode rows `run` returns.
    """
    order = list(modes["variant"].unique())

    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(13, 5), layout="constrained")
    students, index = fig.subplots(1, 2, width_ratios=[2, 1])
    sns.barplot(
        modes,
        x="variant",
        y="students",
        hue="mode",
        order=order,
        hue_order=MODES,
        palette=MODE_COLOURS,
        errorbar=("pi", 100),
        legend=False,
        ax=students,
    )
    students.set_title("Expected mode")
    students.set_ylabel("Students")
    sns.boxplot(
        results,
        x="variant",
        y="dissimilarity",
        order=order,
        color="#d9dde3",
        width=0.5,
        ax=index,
    )
    sns.stripplot(
        results, x="variant", y="dissimilarity", order=order, color="#1f2933", ax=index
    )
    index.set_title("Dissimilarity index")
    index.set_ylabel("")
    index.set_ylim(0, 1)
    for ax in (students, index):
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.yaxis.grid(True, color="#e5e7eb")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    fig.legend(
        handles=[Patch(color=MODE_COLOURS[mode], label=mode) for mode in MODES],
        loc="outside upper center",
        ncol=len(MODES),
        frameon=False,
    )

    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the mechanism variants of the secondary matching "
        "against the paper's, with and without routes over the same fresh samples."
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument(
        "--walk-distance",
        type=float,
        default=WALK_DISTANCE,
        help="metres within which a student is taken to walk, so is offered no route",
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

    results, modes = run(args.seeds, args.walk_distance, args.years, args.circuity)
    print("Mean over seeds:")
    print(summary(results, modes).round(3).to_string())
    plot(results, modes)
    print(f"Wrote {RESULTS_CSV}, {MODES_CSV} and {PLOT_PNG}.")


if __name__ == "__main__":
    main()
