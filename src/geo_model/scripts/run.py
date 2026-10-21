"""Run one setting of the parameters, the atom every other analysis builds on.

Fresh student samples are matched with and without the setting's routes and
both matchings are scored, so whatever differs between them differs by the
routes alone. The route set, the score rows and every figure of the setting
are written to the region's folder, and the two scenarios are reported against
each other. Every parameter is an option, so any setting can be run from the
command line. The scenarios are presets of it; the sweep scores each of its
cells, and the search each candidate, with `score_settings`, and the search
runs its compromise through `run`.
"""

import argparse
from dataclasses import fields
from pathlib import Path

import geopandas as gpd
import pandas as pd
from matplotlib.figure import Figure

from geo_model.analysis.scoring import (
    district_rows,
    score_sample,
    settings_matchings,
    settings_routes,
)
from geo_model.cli import (
    add_disadvantage_argument,
    add_intake_argument,
    add_progressivity_argument,
    add_region_arguments,
    add_seat_arguments,
    add_seeds_argument,
    add_settings_arguments,
    add_travel_arguments,
)
from geo_model.data.regions import regions
from geo_model.data.schools import load_schools
from geo_model.data.travel import load_nts_mode_shares
from geo_model.measures.segregation import fsm_share, fsm_term
from geo_model.measures.statistics import routes_t_test, t_test_label
from geo_model.measures.summaries import (
    local_acceptance,
    local_summary,
    route_utilisation,
    utilisation_summary,
)
from geo_model.measures.travel import CIRCUITY, MODES
from geo_model.model.settings import (
    DEFAULTS,
    DISADVANTAGE,
    LINEAR_PROGRESSIVITY,
    N_SEEDS,
    ROUND_UP_SEATS,
    Settings,
)
from geo_model.model.students import Sample, disadvantaged_group, student_samples
from geo_model.output import region_dir, save, save_routes
from geo_model.plots.comparison import (
    draw_local_acceptance,
    draw_route_utilisation,
    draw_school_intake,
    draw_unassigned,
    plot_dissimilarity,
    plot_lorenz,
    plot_modes,
)
from geo_model.plots.maps import plot_map
from geo_model.plots.style import INTAKE_MEASURES

# The run's folder within each region's, and the score rows written to it.
OUT_DIR = Path("run")
RESULTS_CSV = "results.csv"
SCHOOLS_CSV = "schools.csv"
MODES_CSV = "modes.csv"
ROUTE_UTILISATION_CSV = "route_utilisation.csv"
DISTRICTS_CSV = "districts.csv"
# The setting against the same students without routes: every metric the sweep
# draws, the Lorenz curves and the map.
DISSIMILARITY_PNG = "dissimilarity.png"
UNASSIGNED_PNG = "unassigned.png"
INTAKE_PNG = "intake.png"
MODES_PNG = "car_displacement.png"
LORENZ_PNG = "lorenz.png"
MAP_PNG = "map.png"
ROUTES_PNG = "route_utilisation.png"
LOCALS_PNG = "local_acceptance.png"

DESCRIPTION = (
    "Match fresh samples with and without the routes of one setting of the "
    "parameters, and plot and map the two against each other."
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
    routes: pd.DataFrame | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Score every sample at one setting of the parameters.

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

        routes (pd.DataFrame | None, optional): The route set at `settings`,
        as `settings_routes` builds it, where the caller has built it already.
        Defaults to None, building it here.

    Returns:
        tuple[list[dict], list[dict], list[dict], list[dict]]: The scenario,
        school, mode and route rows `score_sample` returns for every sample,
        each tagged with "seed", the scenario rows with "n_routes" as well.
        The route rows are empty unless `utilisation`.
    """
    print(f"Scoring {settings}")
    if routes is None:
        routes = settings_routes(
            settings, areas, secondary_schools, round_up, linear, disadvantage
        )
    rows, school_rows, mode_rows, route_rows = [], [], [], []
    for seed, (student_xy, student_lsoa, drawn, noise_seed) in enumerate(samples):
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
            noise_seed,
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
    settings: Settings,
    samples: list[Sample],
    areas: gpd.GeoDataFrame,
    secondary_schools: gpd.GeoDataFrame,
    shares: pd.DataFrame,
    out_dir: Path,
    circuity: float = CIRCUITY,
    round_up: bool = ROUND_UP_SEATS,
    linear: bool = LINEAR_PROGRESSIVITY,
    disadvantage: str = DISADVANTAGE,
    measure: str = "dissimilarity",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score `settings` over `samples` with routes and without, and write,
    plot and report the two against each other in `out_dir`.

    Writes the route set, as `save_routes` does, the scenario, school, mode
    and route rows to RESULTS_CSV, SCHOOLS_CSV, MODES_CSV and
    ROUTE_UTILISATION_CSV, and each district's groups in every sample, as
    `district_rows` gives them, to DISTRICTS_CSV. Draws every metric the
    sweep draws: the index per scenario and its paired t-test to
    DISSIMILARITY_PNG, who is left unassigned to UNASSIGNED_PNG, each
    school's intake to INTAKE_PNG and the expected modes and their change to
    MODES_PNG. Draws the Lorenz curves to LORENZ_PNG as well, a map of the
    first sample's two matchings to MAP_PNG, the seats each route provides
    and its riders take to ROUTES_PNG, and the local students each school
    accepts of those who applied, with the change the routes make to them
    per seed, to LOCALS_PNG. Then prints the two scenarios' means and their
    paired t-test, the change per mode, each route's seats and riders, and
    each school's local applicants.

    Args:
        settings (Settings): The parameter values to build routes and rank
        preferences with.

        samples (list[Sample]): Student samples, as yielded by
        `student_samples`, in seed order.

        areas (gpd.GeoDataFrame): Passed to `score_settings`.

        secondary_schools (gpd.GeoDataFrame): Passed to `score_settings`.

        shares (pd.DataFrame): Passed to `score_settings`.

        out_dir (Path): Folder everything is written to.

        circuity (float, optional): Passed to `score_settings`. Defaults to
        CIRCUITY.

        round_up (bool, optional): Passed to `settings_routes` and
        `score_settings`. Defaults to ROUND_UP_SEATS.

        linear (bool, optional): Passed to `settings_routes` and
        `score_settings`. Defaults to LINEAR_PROGRESSIVITY.

        disadvantage (str, optional): Passed to `settings_routes`,
        `score_settings`, `district_rows`, `plot_lorenz` and
        `settings_matchings`. Defaults to DISADVANTAGE.

        measure (str, optional): A key of INTAKE_MEASURES, what the intake
        heatmap shows. Defaults to "dissimilarity".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The
        scenario, school, mode and route rows `score_settings` returns with
        `utilisation`.
    """
    if not samples:
        raise ValueError("No student sample to compare on.")
    network = settings_routes(
        settings, areas, secondary_schools, round_up, linear, disadvantage
    )
    save_routes(network, out_dir)
    results, schools, modes, routes = (
        pd.DataFrame(rows)
        for rows in score_settings(
            settings,
            samples,
            areas,
            secondary_schools,
            shares,
            circuity,
            round_up,
            linear,
            disadvantage,
            utilisation=True,
            routes=network,
        )
    )
    plot_dissimilarity(results, out_dir / DISSIMILARITY_PNG)
    # A bare Figure draws without a display backend, which pyplot would need.
    fig = Figure(figsize=(5, 4.5), layout="constrained")
    draw_unassigned(fig.subplots(), results)
    save(fig, out_dir / UNASSIGNED_PNG)
    fig = Figure(
        figsize=(8, 1.5 + 0.4 * schools["school"].nunique()), layout="constrained"
    )
    fsm = {"dissimilarity": fsm_term, "share": fsm_share}[measure]
    draw_school_intake(fig, schools, fsm(secondary_schools), measure)
    save(fig, out_dir / INTAKE_PNG)
    plot_modes(modes, out_dir / MODES_PNG)
    plot_lorenz(schools, out_dir / LORENZ_PNG, disadvantage)
    plot_map(
        areas,
        secondary_schools,
        samples[0][0],
        settings_matchings(
            settings,
            samples[0],
            areas,
            secondary_schools,
            round_up,
            linear,
            disadvantage,
            network,
        ),
        out_dir / MAP_PNG,
    )
    n_routes = routes["route"].nunique()
    fig = Figure(
        figsize=(min(max(8.0, 2 + 0.3 * n_routes), 30.0), 9), layout="constrained"
    )
    draw_route_utilisation(fig, routes)
    save(fig, out_dir / ROUTES_PNG)
    n_schools = schools["school"].nunique()
    fig = Figure(
        figsize=(min(max(8.0, 2 + 0.4 * n_schools), 30.0), 11), layout="constrained"
    )
    draw_local_acceptance(fig, schools)
    save(fig, out_dir / LOCALS_PNG)

    districts = district_rows(samples, areas, settings.decile, disadvantage)
    for frame, name in (
        (results, RESULTS_CSV),
        (schools, SCHOOLS_CSV),
        (modes, MODES_CSV),
        (routes, ROUTE_UTILISATION_CSV),
        (districts, DISTRICTS_CSV),
    ):
        frame.to_csv(out_dir / name, index=False)

    print(f"With routes and without, mean over {len(samples)} samples:")
    print(
        results.groupby("scenario")[
            ["dissimilarity", "n_unmatched", "unassigned_disadvantaged"]
        ]
        .mean()
        .round(3)
        .to_string()
    )
    print(t_test_label(routes_t_test(results)))
    print("The index over samples:")
    print(results.groupby("scenario")["dissimilarity"].describe().to_string())
    # Paired by seed, so the change in the means is the mean of the changes.
    per_mode = modes.pivot_table(index="mode", columns="scenario", values="students")
    per_mode = per_mode.reindex(MODES)
    per_mode["change"] = per_mode["with routes"] - per_mode["without routes"]
    per_mode["share with"] = per_mode["with routes"] / per_mode["with routes"].sum()
    per_mode["share without"] = (
        per_mode["without routes"] / per_mode["without routes"].sum()
    )
    print("Students per mode, mean over samples, and each scenario's shares:")
    print(per_mode.round(3))
    print("Seats on each route and the riders taking them, mean over samples:")
    print(
        route_utilisation(routes).drop(columns="used").round(2).to_string(index=False)
    )
    print(utilisation_summary(routes))
    print(
        "Local students applying to each school and those it accepts, "
        "mean over samples:"
    )
    print(local_acceptance(schools).round(2).to_string(index=False))
    print(local_summary(schools))
    print(
        f"Wrote the routes, {RESULTS_CSV}, {SCHOOLS_CSV}, {MODES_CSV}, "
        f"{ROUTE_UTILISATION_CSV}, {DISTRICTS_CSV} and the plots in {out_dir}."
    )
    return results, schools, modes, routes


def main(
    settings: Settings = DEFAULTS,
    out_dir: Path = OUT_DIR,
    description: str = DESCRIPTION,
    *,
    seeds: int = N_SEEDS,
    round_up: bool = ROUND_UP_SEATS,
) -> None:
    """Run one setting in every region named, as the command line sets out.

    A preset calls it with values of its own, which its options then default
    to, so a preset takes every option a run takes.

    Args:
        settings (Settings, optional): The setting every parameter's option
        defaults to. Defaults to DEFAULTS.

        out_dir (Path, optional): The folder within each region's the run
        writes to. Defaults to OUT_DIR.

        description (str, optional): The command line's description. Defaults
        to DESCRIPTION.

        seeds (int, optional): Samples drawn unless --seeds says otherwise.
        Defaults to N_SEEDS.

        round_up (bool, optional): Whether route seats are rounded up unless
        --no-round-up-seats says otherwise. Defaults to ROUND_UP_SEATS.
    """
    parser = argparse.ArgumentParser(description=description)
    add_seeds_argument(parser, seeds)
    add_settings_arguments(parser, settings)
    add_travel_arguments(parser)
    add_seat_arguments(parser, round_up)
    add_progressivity_argument(parser)
    add_disadvantage_argument(parser)
    add_intake_argument(parser, list(INTAKE_MEASURES))
    add_region_arguments(parser)
    args = parser.parse_args()
    # Checked here as well as by route_network, so a run fails before its
    # region is loaded and sampled rather than after.
    if sum((args.shortest_only, args.bottleneck, args.largest_first)) > 1:
        parser.error(
            "--shortest-only, --bottleneck and --largest-first each keep one route "
            "per school by a rule of their own, so only one may be set"
        )

    settings = Settings(**{f.name: getattr(args, f.name) for f in fields(Settings)})
    shares = load_nts_mode_shares(args.years)
    for las, areas in regions(args.la, args.merge):
        print(" + ".join(las) + ":")
        _, secondary_schools = load_schools(las)
        samples = list(student_samples(areas, args.seeds))
        run(
            settings,
            samples,
            areas,
            secondary_schools,
            shares,
            region_dir(las) / out_dir,
            args.circuity,
            args.round_up_seats,
            args.linear_progressivity,
            args.disadvantage,
            args.intake,
        )


if __name__ == "__main__":
    main()
