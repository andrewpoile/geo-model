from collections.abc import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd

from geo_model.measures.segregation import (
    dissimilarity_index,
    dissimilarity_terms,
    district_groups,
    local_intake,
    school_intake,
)
from geo_model.measures.travel import CIRCUITY, expected_modes
from geo_model.model.matching import fast_DAT
from geo_model.model.preferences import district_index, secondary_instance
from geo_model.model.routes import route_network
from geo_model.model.settings import (
    DISADVANTAGE,
    DISADVANTAGED_PERFORMANCE_WEIGHT,
    DISADVANTAGED_ROUTE_DISCOUNT,
    LINEAR_PROGRESSIVITY,
    PERFORMANCE_WEIGHT,
    PRIORITY,
    PRIORITY_DISCOUNT,
    ROUND_UP_SEATS,
    ROUTE_DISCOUNT,
    SCENARIOS,
    Settings,
)
from geo_model.model.students import (
    Sample,
    cohort_sizes,
    disadvantaged_group,
)


def match_sample(
    student_distance: np.ndarray,
    student_lsoa: np.ndarray,
    areas: pd.DataFrame,
    secondary_schools: gpd.GeoDataFrame,
    routes: pd.DataFrame,
    disadvantaged: np.ndarray,
    noise_seed: np.random.SeedSequence,
    performance_weight: float = PERFORMANCE_WEIGHT,
    route_discount: float = ROUTE_DISCOUNT,
    disadvantaged_performance_weight: float = DISADVANTAGED_PERFORMANCE_WEIGHT,
    disadvantaged_route_discount: float = DISADVANTAGED_ROUTE_DISCOUNT,
    disadvantage: str = DISADVANTAGE,
    priority: str = PRIORITY,
    priority_discount: float = PRIORITY_DISCOUNT,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Match one student sample with and without routes.

    The sample is matched twice: as the transport instance with its routes,
    and as the same students, with the same tastes, with no routes at all, so
    whatever is measured on the two matchings differs by the routes alone.

    Args:
        student_distance (np.ndarray): Distance from every student to every
        school, shape (n_students, n_schools).

        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        areas (pd.DataFrame): Districts, as `secondary_instance` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools, as `secondary_instance`
        takes them.

        routes (pd.DataFrame): The route set, as returned by `route_network`.

        disadvantaged (np.ndarray): Passed to `secondary_instance`.

        noise_seed (np.random.SeedSequence): Passed to `secondary_instance`
        for both scenarios, so both rank with the same tastes.

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

        priority (str, optional): Passed to `secondary_instance`. Defaults to
        PRIORITY.

        priority_discount (float, optional): Passed to `secondary_instance`.
        Defaults to PRIORITY_DISCOUNT.

    Yields:
        tuple[str, np.ndarray, np.ndarray]: The scenario, "with routes" then
        "without routes", the students' preferences in it, as
        `secondary_instance` ranks them, and its matching as returned by
        `fast_DAT`.
    """
    for scenario, route_set in zip(SCENARIOS, (routes, None)):
        instance = secondary_instance(
            student_distance,
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
            priority=priority,
            priority_discount=priority_discount,
            noise_seed=noise_seed,
        )
        yield scenario, instance[0], fast_DAT(*instance)


def score_sample(
    student_distance: np.ndarray,
    student_lsoa: np.ndarray,
    areas: pd.DataFrame,
    secondary_schools: gpd.GeoDataFrame,
    routes: pd.DataFrame,
    disadvantaged: np.ndarray,
    shares: pd.DataFrame,
    noise_seed: np.random.SeedSequence,
    *,
    performance_weight: float = PERFORMANCE_WEIGHT,
    route_discount: float = ROUTE_DISCOUNT,
    disadvantaged_performance_weight: float = DISADVANTAGED_PERFORMANCE_WEIGHT,
    disadvantaged_route_discount: float = DISADVANTAGED_ROUTE_DISCOUNT,
    circuity: float = CIRCUITY,
    disadvantage: str = DISADVANTAGE,
    priority: str = PRIORITY,
    priority_discount: float = PRIORITY_DISCOUNT,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Match one student sample with and without routes and score both.

    Both matchings are scored for segregation on the same students, so the
    two scores differ by the routes alone. Each school's intake is counted by
    group as well, so the composition behind the index can be seen, and its
    local applicants and those of them seated, so any displacement of local
    students can be seen. The students seated are counted by expected travel
    mode, so the modal shift the routes induce can be seen too, and each
    route's riders by group, so the use made of its seats can be seen.

    Args:
        student_distance (np.ndarray): Distance from every student to every
        school, shape (n_students, n_schools).

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

        noise_seed (np.random.SeedSequence): Passed to `match_sample`.

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

        priority (str, optional): Passed to `secondary_instance`. Defaults to
        PRIORITY.

        priority_discount (float, optional): Passed to `secondary_instance`.
        Defaults to PRIORITY_DISCOUNT.

    Returns:
        tuple[list[dict], list[dict], list[dict], list[dict]]: One row per
        scenario, with keys "scenario", "dissimilarity", "n_matched",
        "n_unmatched", "n_disadvantaged" (students disadvantaged) and
        "unassigned_disadvantaged" (those of them left without a place); one
        row per (scenario, school), with keys "scenario", "school" (the
        establishment name), "disadvantaged" and "other" (students seated),
        "dissimilarity_term" (the school's term of the index, as
        `dissimilarity_terms` gives it), "local_applicants" (students of the
        school's district who applied to it) and "local_disadvantaged" and
        "local_other" (those of them seated, as `local_intake` counts them);
        one row per (scenario, mode), with keys "scenario", "mode" and
        "students" (expected); and one row per
        route of the routed matching, with keys "route" (its route_id),
        "LSOA21CD", "school" (the establishment name), "capacity" and
        "disadvantaged" and "other" (students riding it).
    """
    n_schools = len(secondary_schools)
    school_district = district_index(secondary_schools, areas)
    rows, school_rows, mode_rows, route_rows = [], [], [], []
    for scenario, preferences, matching in match_sample(
        student_distance,
        student_lsoa,
        areas,
        secondary_schools,
        routes,
        disadvantaged,
        noise_seed,
        performance_weight,
        route_discount,
        disadvantaged_performance_weight,
        disadvantaged_route_discount,
        disadvantage,
        priority=priority,
        priority_discount=priority_discount,
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
        applicants, local_a, local_b = local_intake(
            preferences, matching, student_lsoa, school_district, disadvantaged
        )
        school_rows += [
            {
                "scenario": scenario,
                "school": name,
                "disadvantaged": a,
                "other": b,
                "dissimilarity_term": term,
                "local_applicants": applied,
                "local_disadvantaged": seated_a,
                "local_other": seated_b,
            }
            for name, a, b, term, applied, seated_a, seated_b in zip(
                secondary_schools["EstablishmentName"],
                group_a,
                group_b,
                dissimilarity_terms(group_a, group_b),
                applicants,
                local_a,
                local_b,
            )
        ]
        modes = expected_modes(matching, student_distance, shares, circuity)
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
                    "minutes": minutes,
                    "disadvantaged": a,
                    "other": b,
                }
                for route, lsoa, name, capacity, minutes, a, b in zip(
                    routes["route_id"],
                    routes["LSOA21CD"],
                    secondary_schools["EstablishmentName"].to_numpy()[
                        routes["school_idx"]
                    ],
                    routes["capacity"],
                    routes["minutes"],
                    riders_a,
                    riders_b,
                )
            ]
    return rows, school_rows, mode_rows, route_rows


def settings_routes(
    settings: Settings,
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    district_distance: np.ndarray,
    district_minutes: np.ndarray | None,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
) -> pd.DataFrame:
    """The route set at one setting of the parameters.

    Args:
        settings (Settings): The parameter values to build routes with.

        areas (gpd.GeoDataFrame): Districts, as `route_network` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools carrying "P8MEA" and
        "PlacesOffered".

        district_distance (np.ndarray): Passed to `route_network`.

        district_minutes (np.ndarray): Passed to `route_network`.

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
        district_distance,
        district_minutes,
        secondary_schools["P8MEA"].to_numpy(),
        secondary_schools["PlacesOffered"].to_numpy(),
        cohort_sizes(areas, "secondary"),
        decile=settings.decile,
        min_distance=settings.min_distance,
        max_minutes=settings.max_route_minutes,
        capacity_scale=settings.capacity_scale,
        max_local_p8=settings.max_local_p8,
        local_radius=settings.local_radius,
        round_up=round_up,
        progressivity=settings.progressivity,
        linear=linear,
        disadvantage=disadvantage,
        shortest_only=settings.shortest_only,
        bottleneck=settings.bottleneck,
        largest_first=settings.largest_first,
    )


def settings_matchings(
    settings: Settings,
    sample: Sample,
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    district_distance: np.ndarray,
    district_minutes: np.ndarray | None,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
    priority: str = PRIORITY,
    routes: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Match one student sample at `settings`, with routes and without.

    The tastes are drawn from the sample's own seed and the mechanism is
    deterministic, so these are the matchings `score_settings` scores for the
    sample at `settings`.

    Args:
        settings (Settings): The parameter values to build routes and rank
        preferences with.

        sample (Sample): One student sample, as yielded by
        `student_samples`.

        areas (gpd.GeoDataFrame): Districts, as `score_settings` takes them.

        secondary_schools (gpd.GeoDataFrame): Schools, as `score_settings`
        takes them.

        district_distance (np.ndarray): Passed to `settings_routes`.

        district_minutes (np.ndarray): Passed to `settings_routes`.

        round_up (bool, optional): Passed to `settings_routes`. Defaults to
        ROUND_UP_SEATS.

        linear (bool, optional): Passed to `settings_routes`. Defaults to
        LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): As `score_settings` takes it. Defaults
        to DISADVANTAGE.

        priority (str, optional): Passed to `match_sample`. Defaults to
        PRIORITY.

        routes (pd.DataFrame | None, optional): The route set at `settings`,
        as `settings_routes` builds it, where the caller has built it already.
        Defaults to None, building it here.

    Returns:
        dict[str, np.ndarray]: The matching of each scenario, "with routes"
        then "without routes", as returned by `fast_DAT`.
    """
    _, student_lsoa, drawn, noise_seed, student_distance = sample
    if routes is None:
        routes = settings_routes(
            settings,
            areas,
            secondary_schools,
            district_distance,
            district_minutes,
            round_up,
            linear,
            disadvantage,
        )
    return {
        scenario: matching
        for scenario, _, matching in match_sample(
            student_distance,
            student_lsoa,
            areas,
            secondary_schools,
            routes,
            disadvantaged_group(
                student_lsoa, drawn, areas, settings.decile, disadvantage
            ),
            noise_seed,
            settings.performance_weight,
            settings.route_discount,
            settings.disadvantaged_performance_weight,
            settings.disadvantaged_route_discount,
            disadvantage,
            priority=priority,
            priority_discount=settings.priority_discount,
        )
    }


def district_rows(
    samples: list[Sample], areas: pd.DataFrame, decile: int, disadvantage: str
) -> pd.DataFrame:
    """Each district's disadvantaged and advantaged students in every
    sample, as `district_groups` gives them, a seed's cohorts being drawn
    afresh.

    Args:
        samples (list[Sample]): Student samples, as yielded by
        `student_samples`, in seed order.

        areas (pd.DataFrame): Districts, as `district_groups` takes them.

        decile (int): Passed to `disadvantaged_group`.

        disadvantage (str): Passed to `disadvantaged_group`.

    Returns:
        pd.DataFrame: One row per (seed, district), "seed" ahead of the
        columns `district_groups` returns.
    """
    return pd.concat(
        [
            district_groups(
                areas,
                student_lsoa,
                disadvantaged_group(student_lsoa, drawn, areas, decile, disadvantage),
            ).assign(seed=seed)
            for seed, (_, student_lsoa, drawn, _, _) in enumerate(samples)
        ],
        ignore_index=True,
    ).pipe(lambda rows: rows[["seed", *rows.columns.drop("seed")]])
