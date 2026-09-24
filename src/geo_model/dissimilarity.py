import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from geo_model.build_prefs import (
    DISADVANTAGED_PERFORMANCE_WEIGHT,
    DISADVANTAGED_ROUTE_DISCOUNT,
    PERFORMANCE_WEIGHT,
    ROUTE_DISCOUNT,
    SEED,
    cohort_sizes,
    secondary_instance,
)
from geo_model.build_routes import (
    DISADVANTAGED_DECILE,
    LOCAL_RADIUS,
    MAX_LOCAL_P8,
    MIN_ROUTE_DISTANCE,
    ROUND_UP_SEATS,
    ROUTE_CAPACITY_SCALE,
    route_network,
)
from geo_model.load_data import load_areas, load_nts_mode_shares, load_schools
from geo_model.matching import fast_DAT
from geo_model.utils import (
    CIRCUITY,
    disadvantaged_students,
    dissimilarity_index,
    dissimilarity_terms,
    expected_modes,
    sample_students,
    school_intake,
)

N_SEEDS = 30
RESULTS_CSV = Path("temp/dissimilarity.csv")
PLOT_PNG = Path("temp/dissimilarity.png")


@dataclass(frozen=True)
class Settings:
    """The parameters a matching can be scored under, at the model's defaults."""

    decile: int = DISADVANTAGED_DECILE
    min_distance: float = MIN_ROUTE_DISTANCE
    capacity_scale: float = ROUTE_CAPACITY_SCALE
    max_local_p8: float = MAX_LOCAL_P8
    local_radius: float = LOCAL_RADIUS
    performance_weight: float = PERFORMANCE_WEIGHT
    route_discount: float = ROUTE_DISCOUNT
    disadvantaged_performance_weight: float = DISADVANTAGED_PERFORMANCE_WEIGHT
    disadvantaged_route_discount: float = DISADVANTAGED_ROUTE_DISCOUNT


DEFAULTS = Settings()


def student_samples(
    areas: pd.DataFrame, sizes: np.ndarray, n_seeds: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Draw one secondary student sample per seed, reproducibly from SEED.

    Args:
        areas (pd.DataFrame): Districts carrying "Borders" and "Centroids".

        sizes (np.ndarray): Students to sample in each area, aligned with
        `areas`.

        n_seeds (int): Number of samples to draw.

    Yields:
        tuple[np.ndarray, np.ndarray]: Coordinates and positional district
        index of each student, as returned by `sample_students`.
    """
    for stream in np.random.SeedSequence(SEED).spawn(n_seeds):
        yield sample_students(
            areas["Borders"], areas["Centroids"], sizes, np.random.default_rng(stream)
        )


def match_sample(
    student_xy: np.ndarray,
    student_lsoa: np.ndarray,
    areas: pd.DataFrame,
    secondary_schools: gpd.GeoDataFrame,
    routes: pd.DataFrame,
    disadvantaged: np.ndarray,
    performance_weight: float = PERFORMANCE_WEIGHT,
    route_discount: float = ROUTE_DISCOUNT,
    disadvantaged_performance_weight: float = DISADVANTAGED_PERFORMANCE_WEIGHT,
    disadvantaged_route_discount: float = DISADVANTAGED_ROUTE_DISCOUNT,
) -> Iterator[tuple[str, np.ndarray]]:
    """Match one student sample with and without routes.

    The sample is matched twice: as the transport instance with its routes,
    and as the same students with no routes at all, so whatever is measured
    on the two matchings differs by the routes alone.

    Args:
        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        areas (pd.DataFrame): Districts, as `secondary_instance` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools, as `secondary_instance`
        takes them.

        routes (pd.DataFrame): The route set, as returned by `route_network`.

        disadvantaged (np.ndarray): Passed to `secondary_instance`.

        performance_weight (float, optional): Passed to `secondary_instance`.
        Defaults to PERFORMANCE_WEIGHT.

        route_discount (float, optional): Passed to `secondary_instance`.
        Defaults to ROUTE_DISCOUNT.

        disadvantaged_performance_weight (float, optional): Passed to
        `secondary_instance`. Defaults to DISADVANTAGED_PERFORMANCE_WEIGHT.

        disadvantaged_route_discount (float, optional): Passed to
        `secondary_instance`. Defaults to DISADVANTAGED_ROUTE_DISCOUNT.

    Yields:
        tuple[str, np.ndarray]: The scenario, "with routes" then "without
        routes", and its matching as returned by `fast_DAT`.
    """
    for scenario, route_set in [("with routes", routes), ("without routes", None)]:
        instance = secondary_instance(
            student_xy,
            student_lsoa,
            areas,
            secondary_schools,
            route_set,
            disadvantaged,
            performance_weight,
            route_discount,
            disadvantaged_performance_weight,
            disadvantaged_route_discount,
        )
        yield scenario, fast_DAT(*instance)


def score_sample(
    student_xy: np.ndarray,
    student_lsoa: np.ndarray,
    areas: pd.DataFrame,
    secondary_schools: gpd.GeoDataFrame,
    routes: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    decile: int = DISADVANTAGED_DECILE,
    performance_weight: float = PERFORMANCE_WEIGHT,
    route_discount: float = ROUTE_DISCOUNT,
    disadvantaged_performance_weight: float = DISADVANTAGED_PERFORMANCE_WEIGHT,
    disadvantaged_route_discount: float = DISADVANTAGED_ROUTE_DISCOUNT,
    circuity: float = CIRCUITY,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Match one student sample with and without routes and score both.

    Both matchings are scored for segregation on the same students, so the
    two scores differ by the routes alone. Each school's intake is counted by
    group as well, so the composition behind the index can be seen, and the
    students seated are counted by expected travel mode, so the modal shift
    the routes induce can be seen too.

    Args:
        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        areas (pd.DataFrame): Districts, in the order students were sampled
        from, carrying the columns `secondary_instance` and
        `disadvantaged_students` read.

        secondary_schools (gpd.GeoDataFrame): Schools, as `secondary_instance`
        takes them.

        routes (pd.DataFrame): The route set, as returned by `route_network`.

        shares (pd.DataFrame): NTS mode share within each trip-length band,
        as returned by `load_nts_mode_shares`.

        decile (int, optional): Districts at or below this IDACI decile are the
        disadvantaged group the index measures and that ranks with its own
        weights. Defaults to DISADVANTAGED_DECILE.

        performance_weight (float, optional): Passed to `secondary_instance`.
        Defaults to PERFORMANCE_WEIGHT.

        route_discount (float, optional): Passed to `secondary_instance`.
        Defaults to ROUTE_DISCOUNT.

        disadvantaged_performance_weight (float, optional): Passed to
        `secondary_instance`. Defaults to DISADVANTAGED_PERFORMANCE_WEIGHT.

        disadvantaged_route_discount (float, optional): Passed to
        `secondary_instance`. Defaults to DISADVANTAGED_ROUTE_DISCOUNT.

        circuity (float, optional): Passed to `expected_modes`. Defaults to
        CIRCUITY.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: One row per scenario, with
        keys "scenario", "dissimilarity", "n_matched", "n_unmatched",
        "n_disadvantaged" (students drawn from a disadvantaged district) and
        "unassigned_disadvantaged" (those of them left without a place); one
        row per (scenario, school), with keys "scenario", "school" (the
        establishment name), "disadvantaged" and "other" (students seated)
        and "dissimilarity_term" (the school's term of the index, as
        `dissimilarity_terms` gives it);
        and one row per (scenario, mode), with keys "scenario", "mode" and
        "students" (expected).
    """
    disadvantaged = disadvantaged_students(student_lsoa, areas, decile)
    n_schools = len(secondary_schools)
    school_xy = secondary_schools[["Easting", "Northing"]].to_numpy()
    rows, school_rows, mode_rows = [], [], []
    for scenario, matching in match_sample(
        student_xy,
        student_lsoa,
        areas,
        secondary_schools,
        routes,
        disadvantaged,
        performance_weight,
        route_discount,
        disadvantaged_performance_weight,
        disadvantaged_route_discount,
    ):
        matched_school = matching[:, 0]
        rows.append(
            {
                "scenario": scenario,
                "dissimilarity": dissimilarity_index(
                    matched_school, disadvantaged, n_schools
                ),
                "n_matched": int((matched_school >= 0).sum()),
                "n_unmatched": int((matched_school < 0).sum()),
                "n_disadvantaged": int(disadvantaged.sum()),
                "unassigned_disadvantaged": int(
                    (disadvantaged & (matched_school < 0)).sum()
                ),
            }
        )
        group_a, group_b = school_intake(matched_school, disadvantaged, n_schools)
        school_rows += [
            {
                "scenario": scenario,
                "school": name,
                "disadvantaged": a,
                "other": b,
                "dissimilarity_term": term,
            }
            for name, a, b, term in zip(
                secondary_schools["EstablishmentName"],
                group_a,
                group_b,
                dissimilarity_terms(group_a, group_b),
            )
        ]
        modes = expected_modes(matching, student_xy, school_xy, shares, circuity)
        mode_rows += [
            {"scenario": scenario, "mode": mode, "students": students}
            for mode, students in modes.items()
        ]
    return rows, school_rows, mode_rows


def score_settings(
    settings: Settings,
    samples: list[tuple[np.ndarray, np.ndarray]],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    circuity: float = CIRCUITY,
    round_up: bool = ROUND_UP_SEATS,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Score every sample at one setting of the parameters.

    Lives here rather than in `__main__` so worker processes can import it:
    multiprocessing never re-imports a `__main__` module into its workers.

    Args:
        settings (Settings): The parameter values to build routes and rank
        preferences with.

        samples (list[tuple[np.ndarray, np.ndarray]]): Student samples, as
        yielded by `student_samples`, in seed order.

        areas (gpd.GeoDataFrame): Districts, as `route_network` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools, as `score_sample`
        takes them.

        shares (pd.DataFrame): Passed to `score_sample`.

        circuity (float, optional): Passed to `score_sample`. Defaults to
        CIRCUITY.

        round_up (bool, optional): Passed to `route_network`. Defaults to
        ROUND_UP_SEATS.

    Returns:
        tuple[list[dict], list[dict], list[dict]]: The scenario, school and
        mode rows `score_sample` returns for every sample, each tagged with
        "seed", the scenario rows with "n_routes" as well.
    """
    print(f"Scoring {settings}")
    routes = route_network(
        areas,
        secondary_schools[["Easting", "Northing"]].to_numpy(),
        secondary_schools["P8MEA"].to_numpy(),
        secondary_schools["PAN"].to_numpy(),
        cohort_sizes(areas, "secondary"),
        decile=settings.decile,
        min_distance=settings.min_distance,
        capacity_scale=settings.capacity_scale,
        max_local_p8=settings.max_local_p8,
        local_radius=settings.local_radius,
        round_up=round_up,
    )
    rows, school_rows, mode_rows = [], [], []
    for seed, (student_xy, student_lsoa) in enumerate(samples):
        scenario_rows, intake_rows, travel_rows = score_sample(
            student_xy,
            student_lsoa,
            areas,
            secondary_schools,
            routes,
            shares,
            decile=settings.decile,
            performance_weight=settings.performance_weight,
            route_discount=settings.route_discount,
            disadvantaged_performance_weight=settings.disadvantaged_performance_weight,
            disadvantaged_route_discount=settings.disadvantaged_route_discount,
            circuity=circuity,
        )
        rows += [{"seed": seed, "n_routes": len(routes), **r} for r in scenario_rows]
        school_rows += [{"seed": seed, **r} for r in intake_rows]
        mode_rows += [{"seed": seed, **r} for r in travel_rows]
    return rows, school_rows, mode_rows


def run(n_seeds: int) -> pd.DataFrame:
    """Score the secondary matching with and without routes over fresh samples.

    Each seed draws a new student sample and scores it with `score_sample`.

    Args:
        n_seeds (int): Number of student samples to draw.

    Returns:
        pd.DataFrame: One row per (seed, scenario), with column "seed" ahead
        of the scenario-row keys `score_sample` returns. Also written to
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
    shares = load_nts_mode_shares()

    rows = []
    for seed, (student_xy, student_lsoa) in enumerate(
        student_samples(areas, sizes, n_seeds)
    ):
        scenario_rows, _, _ = score_sample(
            student_xy, student_lsoa, areas, secondary_schools, routes, shares
        )
        rows += [{"seed": seed, **r} for r in scenario_rows]
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
