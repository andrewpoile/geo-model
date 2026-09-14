import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from geo_model.build_prefs import SEED, cohort_sizes, secondary_instance
from geo_model.build_routes import DISADVANTAGED_DECILE, route_network
from geo_model.load_data import load_areas, load_schools
from geo_model.matching import fast_DAT
from geo_model.utils import dissimilarity_index, sample_students

N_SEEDS = 30
RESULTS_CSV = Path("temp/dissimilarity.csv")
PLOT_PNG = Path("temp/dissimilarity.png")


def disadvantaged_students(
    student_lsoa: np.ndarray, areas: pd.DataFrame, decile: int = DISADVANTAGED_DECILE
) -> np.ndarray:
    """Whether each student lives in a disadvantaged district.

    Args:
        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        areas (pd.DataFrame): Districts carrying an "IMD Decile" column, in
        the order students were sampled from.

        decile (int, optional): Districts at or below this IMD decile are
        disadvantaged. Defaults to DISADVANTAGED_DECILE.

    Returns:
        np.ndarray: Boolean, one per student.
    """
    return areas["IMD Decile"].to_numpy()[student_lsoa] <= decile


def run(n_seeds: int) -> pd.DataFrame:
    """Score the secondary matching with and without routes over fresh samples.

    Each seed draws a new student sample, which is matched twice: as the
    transport instance with its routes, and as the same students with no
    routes at all. Both matchings are scored for segregation on the same
    students, so the two scores of a seed differ by the routes alone.

    Args:
        n_seeds (int): Number of student samples to draw.

    Returns:
        pd.DataFrame: One row per (seed, scenario), with columns "seed",
        "scenario", "dissimilarity", "n_matched" and "n_unmatched". Also
        written to RESULTS_CSV.
    """
    areas = load_areas()
    _, secondary_schools = load_schools()
    sizes = cohort_sizes(areas, "secondary")
    routes = route_network(areas, secondary_schools[["Easting", "Northing"]].to_numpy())
    n_schools = len(secondary_schools)

    rows = []
    for seed, stream in enumerate(np.random.SeedSequence(SEED).spawn(n_seeds)):
        student_xy, student_lsoa = sample_students(
            areas["Borders"], areas["Centroids"], sizes, np.random.default_rng(stream)
        )
        disadvantaged = disadvantaged_students(student_lsoa, areas)

        for scenario, route_set in [("with routes", routes), ("without routes", None)]:
            instance = secondary_instance(
                student_xy, student_lsoa, areas, secondary_schools, route_set
            )
            matched_school = fast_DAT(*instance)[:, 0]
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "dissimilarity": dissimilarity_index(
                        matched_school, disadvantaged, n_schools
                    ),
                    "n_matched": int((matched_school >= 0).sum()),
                    "n_unmatched": int((matched_school < 0).sum()),
                }
            )
        print(f"Seed {seed + 1} of {n_seeds}: {len(student_xy)} students.")

    results = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    return results


def plot(results: pd.DataFrame) -> None:
    """Box-plot the index per scenario, one point per seed, to PLOT_PNG.

    Args:
        results (pd.DataFrame): As returned by `run`.
    """
    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    sns.boxplot(
        results, x="scenario", y="dissimilarity", color="#d9dde3", width=0.5, ax=ax
    )
    sns.stripplot(results, x="scenario", y="dissimilarity", color="#1f2933", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Dissimilarity index")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color="#e5e7eb")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()

    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the secondary matching with and without routes over fresh samples."
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    results = run(args.seeds)
    print(results.groupby("scenario")["dissimilarity"].describe())
    plot(results)
    print(f"Wrote {RESULTS_CSV} and {PLOT_PNG}.")


if __name__ == "__main__":
    main()
