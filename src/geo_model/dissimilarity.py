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
    DISADVANTAGE,
    DISADVANTAGED_DECILE,
    LINEAR_PROGRESSIVITY,
    LOCAL_RADIUS,
    MAX_LOCAL_P8,
    MIN_ROUTE_DISTANCE,
    ROUND_UP_SEATS,
    ROUTE_CAPACITY_SCALE,
    ROUTE_PROGRESSIVITY,
    SHORTEST_ROUTE_ONLY,
    route_network,
)
from geo_model.load_data import load_nts_mode_shares, load_schools
from geo_model.matching import fast_DAT
from geo_model.utils import (
    CIRCUITY,
    add_region_arguments,
    disadvantaged_group,
    disadvantaged_students,
    dissimilarity_index,
    dissimilarity_terms,
    expected_modes,
    region_dir,
    regions,
    sample_students,
    school_intake,
)

N_SEEDS = 30
# Written to each region's folder.
RESULTS_CSV = "dissimilarity.csv"
PLOT_PNG = "dissimilarity.png"

# A student sample: the coordinates of every student, the positional index of
# the district each was drawn in, and whether each was drawn disadvantaged.
Sample = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Settings:
    """The parameters a matching can be scored under, at the model's defaults."""

    decile: int = DISADVANTAGED_DECILE
    min_distance: float = MIN_ROUTE_DISTANCE
    capacity_scale: float = ROUTE_CAPACITY_SCALE
    progressivity: float = ROUTE_PROGRESSIVITY
    max_local_p8: float = MAX_LOCAL_P8
    local_radius: float = LOCAL_RADIUS
    performance_weight: float = PERFORMANCE_WEIGHT
    route_discount: float = ROUTE_DISCOUNT
    disadvantaged_performance_weight: float = DISADVANTAGED_PERFORMANCE_WEIGHT
    disadvantaged_route_discount: float = DISADVANTAGED_ROUTE_DISCOUNT
    shortest_only: bool = SHORTEST_ROUTE_ONLY


DEFAULTS = Settings()


def student_samples(
    areas: pd.DataFrame, sizes: np.ndarray, n_seeds: int
) -> Iterator[Sample]:
    """Draw one secondary student sample per seed, reproducibly from SEED.

    Which students are disadvantaged is drawn after where they live, from the
    same stream, so the locations of a seed do not depend on it.

    Args:
        areas (pd.DataFrame): Districts carrying "Borders", "Centroids" and
        "IDACI Score".

        sizes (np.ndarray): Students to sample in each area, aligned with
        `areas`.

        n_seeds (int): Number of samples to draw.

    Yields:
        Sample: Coordinates and positional district index of each student, as
        returned by `sample_students`, and whether each is disadvantaged, as
        drawn by `disadvantaged_students`.
    """
    for stream in np.random.SeedSequence(SEED).spawn(n_seeds):
        rng = np.random.default_rng(stream)
        student_xy, student_lsoa = sample_students(
            areas["Borders"], areas["Centroids"], sizes, rng
        )
        yield student_xy, student_lsoa, disadvantaged_students(student_lsoa, areas, rng)


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
    disadvantage: str = DISADVANTAGE,
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

        disadvantage (str, optional): Passed to `secondary_instance`. Defaults
        to DISADVANTAGE.

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
            disadvantage,
        )
        yield scenario, fast_DAT(*instance)


def score_sample(
    student_xy: np.ndarray,
    student_lsoa: np.ndarray,
    areas: pd.DataFrame,
    secondary_schools: gpd.GeoDataFrame,
    routes: pd.DataFrame,
    disadvantaged: np.ndarray,
    shares: pd.DataFrame,
    *,
    performance_weight: float = PERFORMANCE_WEIGHT,
    route_discount: float = ROUTE_DISCOUNT,
    disadvantaged_performance_weight: float = DISADVANTAGED_PERFORMANCE_WEIGHT,
    disadvantaged_route_discount: float = DISADVANTAGED_ROUTE_DISCOUNT,
    circuity: float = CIRCUITY,
    disadvantage: str = DISADVANTAGE,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Match one student sample with and without routes and score both.

    Both matchings are scored for segregation on the same students, so the
    two scores differ by the routes alone. Each school's intake is counted by
    group as well, so the composition behind the index can be seen, the
    students seated are counted by expected travel mode, so the modal shift
    the routes induce can be seen too, and each route's riders by group, so
    the use made of its seats can be seen.

    Args:
        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        areas (pd.DataFrame): Districts, in the order students were sampled
        from, carrying the columns `secondary_instance` reads.

        secondary_schools (gpd.GeoDataFrame): Schools, as `secondary_instance`
        takes them.

        routes (pd.DataFrame): The route set, as returned by `route_network`.

        disadvantaged (np.ndarray): Whether each student is disadvantaged, as
        `disadvantaged_group` gives it: the group the index measures and that
        ranks with its own weights.

        shares (pd.DataFrame): NTS mode share within each trip-length band,
        as returned by `load_nts_mode_shares`.

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

        disadvantage (str, optional): Passed to `secondary_instance`. Defaults
        to DISADVANTAGE.

    Returns:
        tuple[list[dict], list[dict], list[dict], list[dict]]: One row per
        scenario, with keys "scenario", "dissimilarity", "n_matched",
        "n_unmatched", "n_disadvantaged" (students disadvantaged) and
        "unassigned_disadvantaged" (those of them left without a place); one
        row per (scenario, school), with keys "scenario", "school" (the
        establishment name), "disadvantaged" and "other" (students seated)
        and "dissimilarity_term" (the school's term of the index, as
        `dissimilarity_terms` gives it); one row per (scenario, mode), with
        keys "scenario", "mode" and "students" (expected); and one row per
        route of the routed matching, with keys "route" (its route_id),
        "LSOA21CD", "school" (the establishment name), "capacity" and
        "disadvantaged" and "other" (students riding it).
    """
    n_schools = len(secondary_schools)
    school_xy = secondary_schools[["Easting", "Northing"]].to_numpy()
    rows, school_rows, mode_rows, route_rows = [], [], [], []
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
        disadvantage,
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
        if scenario == "with routes":
            # A route's riders are counted as a school's intake is, over the
            # route axis of the matching.
            riders_a, riders_b = school_intake(
                matching[:, 1], disadvantaged, len(routes)
            )
            route_rows = [
                {
                    "route": route,
                    "LSOA21CD": lsoa,
                    "school": name,
                    "capacity": capacity,
                    "disadvantaged": a,
                    "other": b,
                }
                for route, lsoa, name, capacity, a, b in zip(
                    routes["route_id"],
                    routes["LSOA21CD"],
                    secondary_schools["EstablishmentName"].to_numpy()[
                        routes["school_idx"]
                    ],
                    routes["capacity"],
                    riders_a,
                    riders_b,
                )
            ]
    return rows, school_rows, mode_rows, route_rows


def settings_routes(
    settings: Settings,
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
) -> pd.DataFrame:
    """The route set at one setting of the parameters.

    Args:
        settings (Settings): The parameter values to build routes with.

        areas (gpd.GeoDataFrame): Districts, as `route_network` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools carrying "Easting",
        "Northing", "P8MEA" and "PlacesOffered".

        round_up (bool, optional): Passed to `route_network`. Defaults to
        ROUND_UP_SEATS.

        linear (bool, optional): Passed to `route_network`. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): Passed to `route_network`. Defaults to
        DISADVANTAGE.

    Returns:
        pd.DataFrame: The route set, as returned by `route_network`.
    """
    return route_network(
        areas,
        secondary_schools[["Easting", "Northing"]].to_numpy(),
        secondary_schools["P8MEA"].to_numpy(),
        secondary_schools["PlacesOffered"].to_numpy(),
        cohort_sizes(areas, "secondary"),
        decile=settings.decile,
        min_distance=settings.min_distance,
        capacity_scale=settings.capacity_scale,
        max_local_p8=settings.max_local_p8,
        local_radius=settings.local_radius,
        round_up=round_up,
        progressivity=settings.progressivity,
        linear=linear,
        disadvantage=disadvantage,
        shortest_only=settings.shortest_only,
    )


def score_settings(
    settings: Settings,
    samples: list[Sample],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    circuity: float = CIRCUITY,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
    utilisation: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Score every sample at one setting of the parameters.

    Lives here rather than in `__main__` so worker processes can import it:
    multiprocessing never re-imports a `__main__` module into its workers.

    Args:
        settings (Settings): The parameter values to build routes and rank
        preferences with.

        samples (list[Sample]): Student samples, as yielded by
        `student_samples`, in seed order.

        areas (gpd.GeoDataFrame): Districts, as `route_network` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools, as `score_sample`
        takes them.

        shares (pd.DataFrame): Passed to `score_sample`.

        circuity (float, optional): Passed to `score_sample`. Defaults to
        CIRCUITY.

        round_up (bool, optional): Passed to `settings_routes`. Defaults to
        ROUND_UP_SEATS.

        linear (bool, optional): Passed to `settings_routes`. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): Passed to `settings_routes` and
        `score_sample`, and sets each sample's disadvantaged group with
        `disadvantaged_group` at `settings.decile`. Defaults to DISADVANTAGE.

        utilisation (bool, optional): Keep the route rows. A row per seed
        and route would run to tens of millions over a sweep of a merged
        region's whole network, so they are kept only where read. Defaults
        to False.

    Returns:
        tuple[list[dict], list[dict], list[dict], list[dict]]: The scenario,
        school, mode and route rows `score_sample` returns for every sample,
        each tagged with "seed", the scenario rows with "n_routes" as well.
        The route rows are empty unless `utilisation`.
    """
    print(f"Scoring {settings}")
    routes = settings_routes(
        settings, areas, secondary_schools, round_up, linear, disadvantage
    )
    rows, school_rows, mode_rows, route_rows = [], [], [], []
    for seed, (student_xy, student_lsoa, drawn) in enumerate(samples):
        scenario_rows, intake_rows, travel_rows, rider_rows = score_sample(
            student_xy,
            student_lsoa,
            areas,
            secondary_schools,
            routes,
            disadvantaged_group(
                student_lsoa, drawn, areas, settings.decile, disadvantage
            ),
            shares,
            performance_weight=settings.performance_weight,
            route_discount=settings.route_discount,
            disadvantaged_performance_weight=settings.disadvantaged_performance_weight,
            disadvantaged_route_discount=settings.disadvantaged_route_discount,
            circuity=circuity,
            disadvantage=disadvantage,
        )
        rows += [{"seed": seed, "n_routes": len(routes), **r} for r in scenario_rows]
        school_rows += [{"seed": seed, **r} for r in intake_rows]
        mode_rows += [{"seed": seed, **r} for r in travel_rows]
        if utilisation:
            route_rows += [{"seed": seed, **r} for r in rider_rows]
    return rows, school_rows, mode_rows, route_rows


def run(
    n_seeds: int,
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Score the secondary matching with and without routes over fresh samples.

    Each seed draws a new student sample and scores it with `score_sample`.

    Args:
        n_seeds (int): Number of student samples to draw.

        areas (gpd.GeoDataFrame): The region's districts, as `route_network`
        takes them.

        secondary_schools (gpd.GeoDataFrame): The region's schools, as
        `score_sample` takes them.

        out_dir (Path): Folder the rows are written to.

    Returns:
        pd.DataFrame: One row per (seed, scenario), with column "seed" ahead
        of the scenario-row keys `score_sample` returns. Also written to
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
    shares = load_nts_mode_shares()

    rows = []
    for seed, (student_xy, student_lsoa, drawn) in enumerate(
        student_samples(areas, sizes, n_seeds)
    ):
        scenario_rows, _, _, _ = score_sample(
            student_xy,
            student_lsoa,
            areas,
            secondary_schools,
            routes,
            disadvantaged_group(
                student_lsoa, drawn, areas, DEFAULTS.decile, DISADVANTAGE
            ),
            shares,
        )
        rows += [{"seed": seed, **r} for r in scenario_rows]
        print(f"Seed {seed + 1} of {n_seeds}: {len(student_xy)} students.")

    results = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / RESULTS_CSV, index=False)
    return results


def plot(results: pd.DataFrame, path: Path) -> None:
    """Box-plot the index per scenario, one point per seed, to `path`.

    Args:
        results (pd.DataFrame): As returned by `run`, or the scenario rows of
        one setting as `score_settings` returns them.

        path (Path): PNG to write.
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

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the secondary matching with and without routes over fresh samples."
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    add_region_arguments(parser)
    args = parser.parse_args()

    for las, areas in regions(args.la, args.merge):
        out_dir = region_dir(las)
        _, secondary_schools = load_schools(las)
        results = run(args.seeds, areas, secondary_schools, out_dir)
        print(" + ".join(las) + ":")
        print(results.groupby("scenario")["dissimilarity"].describe())
        plot(results, out_dir / PLOT_PNG)
        print(f"Wrote {RESULTS_CSV} and {PLOT_PNG} in {out_dir}.")


if __name__ == "__main__":
    main()
