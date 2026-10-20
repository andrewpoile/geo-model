from dataclasses import asdict, replace
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from geo_model import optimise as op
from geo_model.build_routes import DISADVANTAGE_CHOICES
from geo_model.dissimilarity import DEFAULTS, Settings, score_settings
from geo_model.sweep import COLOURS, GROUP_COLOURS, INTAKE_MEASURES

# A sample of no students, for the checks made before anything is scored.
NO_STUDENTS = [
    (
        np.empty((0, 2)),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
        np.random.SeedSequence(0),
    )
]


def scored_rows(dissimilarity: float, car: tuple[float, float]):
    """Rows shaped like `score_settings` returns for one seed: the routed
    index `dissimilarity`, and `car` students by car without and with routes."""
    results = [
        {"seed": 0, "scenario": "with routes", "dissimilarity": dissimilarity},
        {"seed": 0, "scenario": "without routes", "dissimilarity": 0.5},
    ]
    modes = [
        {"seed": 0, "scenario": scenario, "mode": "car", "students": students}
        for scenario, students in zip(("without routes", "with routes"), car)
    ]
    return results, [], modes, []


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------


def test_routed_dissimilarity_is_the_mean_routed_index_over_seeds():
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "dissimilarity": [0.3, 0.4, 0.2, 0.5],
        }
    )

    assert op.routed_dissimilarity(
        results, pd.DataFrame(), pd.DataFrame()
    ) == pytest.approx(0.25)


def test_car_displacement_is_the_mean_car_students_the_routes_remove():
    # Seed 0 loses 10 car students to routes and seed 1 loses 4; walking
    # changes too, which the car count must not pick up.
    modes = pd.DataFrame(
        {
            "seed": [0] * 4 + [1] * 4,
            "scenario": ["without routes", "with routes"] * 4,
            "mode": ["car", "car", "walk", "walk"] * 2,
            "students": [100.0, 90.0, 50.0, 20.0, 80.0, 76.0, 40.0, 45.0],
        }
    )

    assert op.car_displacement(pd.DataFrame(), pd.DataFrame(), modes) == 7.0


# --------------------------------------------------------------------------
# bounds
# --------------------------------------------------------------------------


def test_bounds_hold_the_default_of_every_optimisable_parameter():
    assert set(op.LEVERS) <= set(op.OPTIMISABLE)
    for parameter in op.OPTIMISABLE:
        low, high = op.bounds(parameter)
        assert low < high
        assert low <= asdict(DEFAULTS)[parameter] <= high


def test_bounds_search_the_decile_over_the_deciles_swept():
    assert op.bounds("decile") == (1.0, 9.0)


def test_bounds_rejects_a_parameter_that_is_not_swept():
    with pytest.raises(ValueError, match="circuity is not optimisable"):
        op.bounds("circuity")


# --------------------------------------------------------------------------
# searchable
# --------------------------------------------------------------------------


def test_searchable_leaves_the_decile_out_only_where_it_sets_the_group():
    assert "decile" in op.searchable("score")
    assert "decile" in op.searchable("score-open")
    assert "decile" not in op.searchable("decile")


def test_searchable_leaves_the_other_students_route_discount_out_unless_they_ride():
    assert "route_discount" in op.searchable("score-open")
    assert "route_discount" not in op.searchable("score")
    assert "route_discount" not in op.searchable("decile")


@pytest.mark.parametrize("disadvantage", DISADVANTAGE_CHOICES)
def test_every_lever_is_searchable_under_every_choice(disadvantage):
    assert set(op.LEVERS) <= set(op.searchable(disadvantage))


# --------------------------------------------------------------------------
# cast
# --------------------------------------------------------------------------


def test_cast_rounds_the_decile_and_makes_every_other_value_a_float():
    cast = op.cast({"decile": np.float64(2.6), "min_distance": np.float64(1609.4)})

    assert cast == {"decile": 3, "min_distance": 1609.4}
    assert type(cast["decile"]) is int
    assert type(cast["min_distance"]) is float


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def test_pipeline_negates_maximised_objectives_and_marks_infeasible_settings():
    def score(settings: Settings):
        # Only a short minimum distance leaves a route here.
        if settings.min_distance > 4000:
            return None
        return scored_rows(settings.capacity_scale / 100, (100.0, 88.0))

    problem = op.Pipeline(
        ["min_distance", "capacity_scale"], op.DEFAULT_OBJECTIVES, score, map
    )
    x = np.array([[1000.0, 20.0], [6000.0, 5.0]])

    F, G = problem.evaluate(x)

    np.testing.assert_array_equal(F, [[0.2, -12.0], [np.inf, np.inf]])
    np.testing.assert_array_equal(G, [[0.0], [1.0]])
    feasible, infeasible = problem.evaluations
    assert feasible == {
        "generation": 1,
        "min_distance": 1000.0,
        "capacity_scale": 20.0,
        "dissimilarity": 0.2,
        "car_displacement": 12.0,
        "feasible": True,
    }
    assert infeasible["feasible"] is False
    assert np.isnan(infeasible["dissimilarity"])
    assert np.isnan(infeasible["car_displacement"])


def test_pipeline_scores_and_records_a_searched_decile_rounded():
    scored_deciles = []

    def score(settings: Settings):
        scored_deciles.append(settings.decile)
        return scored_rows(0.1, (100.0, 90.0))

    problem = op.Pipeline(["decile"], op.DEFAULT_OBJECTIVES, score, map)
    problem.evaluate(np.array([[2.4], [2.6]]))

    assert scored_deciles == [2, 3]
    assert [row["decile"] for row in problem.evaluations] == [2, 3]


# --------------------------------------------------------------------------
# optimise
# --------------------------------------------------------------------------


def test_optimise_rejects_a_parameter_named_twice(tmp_path):
    with pytest.raises(ValueError, match="named once"):
        op.optimise(
            NO_STUDENTS,
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            pd.DataFrame(),
            tmp_path,
            parameters=["min_distance", "min_distance"],
        )


@pytest.mark.parametrize(
    ("parameter", "disadvantage"),
    [("decile", "decile"), ("route_discount", "score"), ("route_discount", "decile")],
)
def test_optimise_rejects_a_parameter_the_choice_of_disadvantage_bars(
    parameter, disadvantage, tmp_path
):
    with pytest.raises(ValueError, match=rf"\['{parameter}'\] cannot be searched"):
        op.optimise(
            NO_STUDENTS,
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            pd.DataFrame(),
            tmp_path,
            parameters=[parameter],
            disadvantage=disadvantage,
        )


def test_optimise_rejects_a_search_where_no_setting_leaves_a_route(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(op, "score_feasible", lambda settings, **kwargs: None)

    with pytest.raises(ValueError, match="None of the 8 settings scored"):
        op.optimise(
            NO_STUDENTS,
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            pd.DataFrame(),
            tmp_path,
            pop_size=4,
            generations=2,
        )


# --------------------------------------------------------------------------
# score_feasible and optimise: against the real data folder
# --------------------------------------------------------------------------


def test_score_feasible_names_a_setting_that_leaves_no_route(
    samples, areas, secondary, shares, capsys
):
    # Every Southampton district lies within 9.9km of every school.
    settings = replace(DEFAULTS, min_distance=20_000)

    scored = op.score_feasible(
        settings,
        samples=samples,
        areas=areas,
        secondary_schools=secondary,
        shares=shares,
    )

    assert scored is None
    out = capsys.readouterr().out
    assert f"Infeasible, no route at {settings}" in out
    assert "so the route set is empty" in out


def test_optimise_finds_the_front_of_every_setting_scored(
    samples, areas, secondary, shares, tmp_path
):
    parameters = ["capacity_scale", "progressivity"]

    evaluations, front = op.optimise(
        samples,
        areas,
        secondary,
        shares,
        tmp_path,
        parameters,
        pop_size=4,
        generations=2,
    )

    assert len(evaluations) == 4 * 2
    assert list(evaluations["generation"]) == [1] * 4 + [2] * 4
    for parameter in parameters:
        low, high = op.bounds(parameter)
        assert evaluations[parameter].between(low, high).all()
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / op.EVALUATIONS_CSV), evaluations
    )
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / op.FRONT_CSV), front)

    # No feasible setting beats a front setting on one objective without
    # losing on the other, and every feasible setting off the front is beaten.
    feasible = evaluations[evaluations["feasible"]]
    values = feasible[["dissimilarity", "car_displacement"]].to_numpy()

    def dominated(point):
        no_worse = (values[:, 0] <= point[0]) & (values[:, 1] >= point[1])
        better = (values[:, 0] < point[0]) | (values[:, 1] > point[1])
        return (no_worse & better).any()

    front_settings = set(zip(*(front[p] for p in parameters)))
    on_front = np.array(
        [s in front_settings for s in zip(*(feasible[p] for p in parameters))]
    )
    assert on_front.sum() == len(front)
    assert not any(dominated(point) for point in values[on_front])
    assert all(dominated(point) for point in values[~on_front])
    assert front["dissimilarity"].is_monotonic_increasing

    # A front setting scored again on the same sample scores the same.
    best = front.iloc[0]
    changes: dict[str, Any] = {p: float(best[p]) for p in parameters}
    settings = replace(DEFAULTS, **changes)
    rescored = op.objective_values(
        score_settings(settings, samples, areas, secondary, shares),
        op.DEFAULT_OBJECTIVES,
    )
    assert rescored == {name: best[name] for name in op.DEFAULT_OBJECTIVES}

    # Scoring the candidates in worker processes changes nothing but the time.
    pooled, _ = op.optimise(
        samples,
        areas,
        secondary,
        shares,
        tmp_path / "pooled",
        parameters,
        pop_size=4,
        generations=2,
        workers=2,
    )
    pd.testing.assert_frame_equal(pooled, evaluations)


# --------------------------------------------------------------------------
# compromise
# --------------------------------------------------------------------------


def test_compromise_is_the_front_setting_nearest_the_rescaled_ideal():
    # Rescaled over the front, 0 best and 1 worst, the settings sit at (0, 1),
    # (1/3, 2/9) and (1, 0), so the middle one is nearest (0, 0).
    front = pd.DataFrame(
        {
            "min_distance": [100.0, 200.0, 300.0],
            "dissimilarity": [0.2, 0.3, 0.5],
            "car_displacement": [10.0, 80.0, 100.0],
        }
    )

    chosen = op.compromise(front, op.DEFAULT_OBJECTIVES)

    assert chosen["min_distance"] == 200.0


def test_compromise_ignores_an_objective_the_whole_front_ties_on():
    front = pd.DataFrame(
        {
            "min_distance": [100.0, 200.0],
            "dissimilarity": [0.3, 0.3],
            "car_displacement": [10.0, 80.0],
        }
    )

    assert op.compromise(front, op.DEFAULT_OBJECTIVES)["min_distance"] == 200.0
    assert op.compromise(front, ["dissimilarity"])["min_distance"] == 100.0


# --------------------------------------------------------------------------
# draw_unassigned
# --------------------------------------------------------------------------


def test_draw_unassigned_stacks_each_groups_mean_over_seeds_per_scenario():
    # With routes 3 disadvantaged and 8 others go unplaced on average, without
    # them 7 and 4.
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "n_unmatched": [10, 10, 12, 12],
            "unassigned_disadvantaged": [2, 8, 4, 6],
        }
    )
    ax = Figure().subplots()

    op.draw_unassigned(ax, results)

    bars = [
        (patch.get_x() + patch.get_width() / 2, patch.get_y(), patch.get_height())
        for patch in ax.patches
        if isinstance(patch, Rectangle)
    ]
    # Disadvantaged at the foot of both bars, then the others on top.
    assert bars == [(0, 0, 3), (1, 0, 7), (0, 3, 8), (1, 7, 4)]
    assert [t.get_text() for t in ax.get_xticklabels()] == list(COLOURS)
    colours = [to_hex(patch.get_facecolor()) for patch in ax.patches]
    assert colours == [
        GROUP_COLOURS[f"{group}, {scenario}"]
        for group in ("disadvantaged", "other")
        for scenario in COLOURS
    ]


# --------------------------------------------------------------------------
# draw_school_intake
# --------------------------------------------------------------------------


def school_rows() -> pd.DataFrame:
    """Two seeds of three schools, each school's disadvantaged share of its
    intake averaging 0.4, 0.25 and 0 with routes, 0.15, 0.75 and 0.4
    without."""
    counts = {
        # (seed, scenario): disadvantaged and other at Alpha, Beta and Gamma.
        (0, "with routes"): ([10, 5, 0], [10, 15, 20]),
        (1, "with routes"): ([6, 5, 0], [14, 15, 20]),
        (0, "without routes"): ([2, 16, 8], [18, 4, 12]),
        (1, "without routes"): ([4, 14, 8], [16, 6, 12]),
    }
    rows = pd.DataFrame(
        {
            "seed": seed,
            "scenario": scenario,
            "school": school,
            "disadvantaged": a,
            "other": b,
        }
        for (seed, scenario), (group_a, group_b) in counts.items()
        for school, a, b in zip(("Alpha", "Beta", "Gamma"), group_a, group_b)
    )
    totals = rows.groupby(["seed", "scenario"])[["disadvantaged", "other"]]
    totals = totals.transform("sum")
    return rows.assign(
        dissimilarity_term=rows["disadvantaged"] / totals["disadvantaged"]
        - rows["other"] / totals["other"]
    )


def drawn_table(fig: Figure) -> tuple[np.ndarray, list[str], list[str]]:
    """The heatmap's cells, row and column labels, as drawn."""
    ax = fig.axes[0]
    cells = ax.collections[0].get_array()
    assert isinstance(cells, np.ma.MaskedArray)
    return (
        cells.filled(np.nan).reshape(3, 3),
        [t.get_text() for t in ax.get_yticklabels()],
        [t.get_text() for t in ax.get_xticklabels()],
    )


def test_draw_school_intake_orders_schools_by_their_share_without_routes():
    fsm = pd.Series({"Alpha": 0.3, "Beta": np.nan, "Gamma": 0.5})
    fig = Figure(layout="constrained")

    op.draw_school_intake(fig, school_rows(), fsm, "share")

    cells, schools, columns = drawn_table(fig)
    assert schools == ["Beta", "Gamma", "Alpha"]
    assert columns == ["with\nroutes", "without\nroutes", "FSM"]
    np.testing.assert_allclose(
        cells, [[0.25, 0.75, np.nan], [0.0, 0.4, 0.5], [0.4, 0.15, 0.3]]
    )
    norm = fig.axes[0].collections[0].norm
    assert (norm.vmin, norm.vmax) == (0, 1)
    assert fig.axes[-1].get_ylabel() == INTAKE_MEASURES["share"]["label"]


def test_draw_school_intake_centres_the_terms_on_the_largest_drawn():
    rows = school_rows()
    # Larger than any school's term, so the register's figure sets the scale.
    fsm = pd.Series({"Alpha": -0.9, "Beta": 0.4, "Gamma": 0.5})
    fig = Figure(layout="constrained")

    op.draw_school_intake(fig, rows, fsm, "dissimilarity")

    cells, schools, _ = drawn_table(fig)
    means = rows.pivot_table(
        index="school", columns="scenario", values="dissimilarity_term"
    )
    np.testing.assert_allclose(
        cells[:, :2], means.loc[schools, list(COLOURS)].to_numpy()
    )
    np.testing.assert_allclose(cells[:, 2], fsm[schools])
    norm = fig.axes[0].collections[0].norm
    assert (norm.vmin, norm.vmax) == pytest.approx((-0.9, 0.9))


# --------------------------------------------------------------------------
# route_utilisation, utilisation_summary and draw_route_utilisation
# --------------------------------------------------------------------------


def route_rows(n_routes: int = 2) -> pd.DataFrame:
    """Route rows over two seeds: route r holds 4 + r seats, and in seed s
    carries r + s disadvantaged riders and 1 other."""
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "route": route,
                "LSOA21CD": f"E{route:08d}",
                "school": f"School {route}",
                "capacity": 4 + route,
                "disadvantaged": route + seed,
                "other": 1,
            }
            for seed in (0, 1)
            for route in range(n_routes)
        ]
    )


def test_route_utilisation_averages_each_routes_riders_over_seeds():
    per_route = op.route_utilisation(route_rows())

    # Route 1 holds more seats, so it comes first.
    np.testing.assert_array_equal(per_route["route"], [1, 0])
    np.testing.assert_array_equal(per_route["capacity"], [5, 4])
    np.testing.assert_array_equal(per_route["disadvantaged"], [1.5, 0.5])
    np.testing.assert_array_equal(per_route["other"], [1.0, 1.0])
    np.testing.assert_array_equal(per_route["used"], [2.5, 1.5])
    np.testing.assert_array_equal(per_route["unused"], [2.5, 2.5])
    np.testing.assert_array_equal(per_route["utilisation"], [0.5, 0.375])


def test_utilisation_summary_totals_the_seats_and_those_used():
    assert op.utilisation_summary(route_rows()) == (
        "2 routes provide 9 seats, 4.0 of them used, the mean over seeds (44.4%)."
    )


@pytest.mark.parametrize(
    ("n_routes", "labelled"), [(2, True), (op.MAX_LABELLED_ROUTES + 1, False)]
)
def test_draw_route_utilisation_stacks_the_riders_below_the_unused_seats(
    n_routes, labelled
):
    fig = Figure()
    op.draw_route_utilisation(fig, route_rows(n_routes))
    ax = fig.axes[0]

    per_route = op.route_utilisation(route_rows(n_routes))
    bars = ax.containers[:3]
    heights = np.array([[bar.get_height() for bar in stack] for stack in bars])
    # Disadvantaged riders, then the others, then the unused seats, so every
    # bar stands at its route's seats.
    np.testing.assert_allclose(heights[0], per_route["disadvantaged"])
    np.testing.assert_allclose(heights[1], per_route["other"])
    np.testing.assert_allclose(heights.sum(axis=0), per_route["capacity"])
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    if labelled:
        assert labels == ["E00000001 → School 1", "E00000000 → School 0"]
    else:
        assert labels == []
        assert ax.get_xlabel() == "Routes, by seats provided"


# --------------------------------------------------------------------------
# local_acceptance, local_summary and draw_local_acceptance
# --------------------------------------------------------------------------


def local_rows() -> pd.DataFrame:
    """School rows over two seeds of three schools' local applicants and the
    disadvantaged and other of them seated."""
    counts = {
        # (seed, scenario): applicants, disadvantaged and other seated at
        # Alpha, Beta and Gamma.
        (0, "with routes"): ([4, 8, 0], [1, 3, 0], [2, 4, 0]),
        (1, "with routes"): ([6, 8, 0], [1, 3, 0], [2, 5, 0]),
        (0, "without routes"): ([4, 8, 0], [2, 4, 0], [2, 4, 0]),
        (1, "without routes"): ([6, 8, 0], [2, 4, 0], [2, 4, 0]),
    }
    return pd.DataFrame(
        {
            "seed": seed,
            "scenario": scenario,
            "school": school,
            "local_applicants": n,
            "local_disadvantaged": a,
            "local_other": b,
        }
        for (seed, scenario), columns in counts.items()
        for school, n, a, b in zip(("Alpha", "Beta", "Gamma"), *columns)
    )


def test_local_acceptance_averages_each_schools_locals_over_seeds():
    per_school = op.local_acceptance(local_rows())

    # Beta has the most local applicants without routes, so it comes first,
    # and Gamma, with none, has no acceptance rate.
    assert list(per_school["school"]) == ["Beta"] * 2 + ["Alpha"] * 2 + ["Gamma"] * 2
    assert list(per_school["scenario"]) == list(COLOURS) * 3
    np.testing.assert_array_equal(per_school["local_applicants"], [8, 8, 5, 5, 0, 0])
    np.testing.assert_array_equal(per_school["accepted"], [7.5, 8, 3, 4, 0, 0])
    np.testing.assert_array_equal(per_school["rejected"], [0.5, 0, 2, 1, 0, 0])
    np.testing.assert_allclose(
        per_school["acceptance"], [0.9375, 1, 0.6, 0.8, np.nan, np.nan]
    )


def test_local_summary_totals_the_applicants_and_those_accepted():
    assert op.local_summary(local_rows()) == (
        "With routes: 13.0 local students applied to their own school, 10.5 of "
        "them accepted, the mean over seeds (80.8%).\n"
        "Without routes: 13.0 local students applied to their own school, 12.0 "
        "of them accepted, the mean over seeds (92.3%)."
    )


def test_draw_local_acceptance_stacks_the_accepted_below_the_rejected():
    fig = Figure()
    op.draw_local_acceptance(fig, local_rows())
    ax = fig.axes[0]

    per_school = op.local_acceptance(local_rows())
    bars = ax.containers[:3]
    heights = np.array([[bar.get_height() for bar in stack] for stack in bars])
    # Disadvantaged accepted, then the others, then those rejected, so every
    # bar stands at its school's local applicants.
    np.testing.assert_allclose(heights[0], per_school["local_disadvantaged"])
    np.testing.assert_allclose(heights[1], per_school["local_other"])
    np.testing.assert_allclose(heights.sum(axis=0), per_school["local_applicants"])
    # Each school's bars with routes then without, in the scenario's colours.
    assert [to_hex(bar.get_facecolor()) for bar in bars[0][:2]] == [
        GROUP_COLOURS[f"disadvantaged, {scenario}"] for scenario in COLOURS
    ]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "Beta",
        "Alpha",
        "Gamma",
    ]


# --------------------------------------------------------------------------
# compare: against the real data folder
# --------------------------------------------------------------------------


def test_compare_scores_and_plots_the_setting_against_no_routes(
    samples, areas, secondary, shares, tmp_path
):
    settings = replace(DEFAULTS, capacity_scale=2.0, progressivity=0.5)

    results, intake, modes, routes = op.compare(
        settings, samples, areas, secondary, shares, tmp_path
    )

    rows, school_rows, mode_rows, route_rows = score_settings(
        settings, samples, areas, secondary, shares, utilisation=True
    )
    pd.testing.assert_frame_equal(results, pd.DataFrame(rows))
    pd.testing.assert_frame_equal(intake, pd.DataFrame(school_rows))
    pd.testing.assert_frame_equal(modes, pd.DataFrame(mode_rows))
    pd.testing.assert_frame_equal(routes, pd.DataFrame(route_rows))
    assert len(routes) == results["n_routes"].iloc[0] * len(samples)
    assert list(results["scenario"]) == ["with routes", "without routes"]
    for name in (
        op.DISSIMILARITY_PNG,
        op.UNASSIGNED_PNG,
        op.INTAKE_PNG,
        op.MODES_PNG,
        op.LORENZ_PNG,
        op.MAP_PNG,
        op.ROUTES_PNG,
        op.LOCALS_PNG,
    ):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name


def test_compare_rejects_no_samples(tmp_path):
    with pytest.raises(ValueError, match="No student sample"):
        op.compare(
            DEFAULTS,
            [],
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            pd.DataFrame(),
            tmp_path,
        )


# --------------------------------------------------------------------------
# plot_front
# --------------------------------------------------------------------------


def test_plot_front_writes_a_png_of_the_front(tmp_path):
    png = tmp_path / "out" / op.FRONT_PNG
    evaluations = pd.DataFrame(
        {
            "generation": [1, 1, 1, 2],
            "min_distance": [1000.0, 2000.0, 9000.0, 3000.0],
            "dissimilarity": [0.30, 0.35, np.nan, 0.40],
            "car_displacement": [5.0, 12.0, np.nan, 8.0],
            "feasible": [True, True, False, True],
        }
    )
    front = evaluations.iloc[[0, 1]].drop(columns="feasible")

    op.plot_front(
        evaluations,
        front,
        front.iloc[1],
        {"dissimilarity": 0.33, "car_displacement": 9.0},
        op.DEFAULT_OBJECTIVES,
        png,
    )

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_front_rejects_anything_but_two_objectives(tmp_path):
    with pytest.raises(ValueError, match="plotted in two objectives"):
        op.plot_front(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(),
            {},
            ["dissimilarity"],
            tmp_path / op.FRONT_PNG,
        )
