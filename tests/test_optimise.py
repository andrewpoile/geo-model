from dataclasses import asdict, replace
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from geo_model import build_prefs as bp
from geo_model import load_data as ld
from geo_model import optimise as op
from geo_model.__main__ import COLOURS, GROUP_COLOURS, INTAKE_MEASURES
from geo_model.dissimilarity import DEFAULTS, Settings, score_settings, student_samples

DATA = ld.POPULATION_XLSX.parent.parent


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
    return results, [], modes


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
    assert "decile" not in op.OPTIMISABLE
    assert set(op.LEVERS) <= set(op.OPTIMISABLE)
    for parameter in op.OPTIMISABLE:
        low, high = op.bounds(parameter)
        assert low < high
        assert low <= asdict(DEFAULTS)[parameter] <= high


def test_bounds_rejects_the_decile():
    with pytest.raises(ValueError, match="decile is not optimisable"):
        op.bounds("decile")


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


# --------------------------------------------------------------------------
# optimise
# --------------------------------------------------------------------------


def test_optimise_rejects_a_parameter_named_twice():
    with pytest.raises(ValueError, match="named once"):
        op.optimise(
            [(np.empty((0, 2)), np.empty(0))],
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            pd.DataFrame(),
            parameters=["min_distance", "min_distance"],
        )


def test_optimise_rejects_a_search_where_no_setting_leaves_a_route(monkeypatch):
    monkeypatch.setattr(op, "score_feasible", lambda settings, **kwargs: None)

    with pytest.raises(ValueError, match="None of the 8 settings scored"):
        op.optimise(
            [(np.empty((0, 2)), np.empty(0))],
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            pd.DataFrame(),
            pop_size=4,
            generations=2,
        )


# --------------------------------------------------------------------------
# score_feasible and optimise: against the real data folder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def secondary():
    """One sample and the frames it is scored with, loaded once for the module."""
    areas = ld.load_areas()
    _, schools = ld.load_schools()
    samples = list(student_samples(areas, bp.cohort_sizes(areas, "secondary"), 1))
    return samples, areas, schools, ld.load_nts_mode_shares()


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_score_feasible_names_a_setting_that_leaves_no_route(secondary, capsys):
    samples, areas, schools, shares = secondary
    # Every district lies within 9.9km of every school.
    settings = replace(DEFAULTS, min_distance=20_000)

    scored = op.score_feasible(
        settings,
        samples=samples,
        areas=areas,
        secondary_schools=schools,
        shares=shares,
    )

    assert scored is None
    out = capsys.readouterr().out
    assert f"Infeasible, no route at {settings}" in out
    assert "so the route set is empty" in out


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_optimise_finds_the_front_of_every_setting_scored(
    secondary, tmp_path, monkeypatch
):
    monkeypatch.setattr(op, "OUT_DIR", tmp_path)
    monkeypatch.setattr(op, "EVALUATIONS_CSV", tmp_path / "evaluations.csv")
    monkeypatch.setattr(op, "FRONT_CSV", tmp_path / "front.csv")
    samples, areas, schools, shares = secondary
    parameters = ["capacity_scale", "progressivity"]

    evaluations, front = op.optimise(
        samples,
        areas,
        schools,
        shares,
        parameters,
        pop_size=4,
        generations=2,
    )

    assert len(evaluations) == 4 * 2
    assert list(evaluations["generation"]) == [1] * 4 + [2] * 4
    for parameter in parameters:
        low, high = op.bounds(parameter)
        assert evaluations[parameter].between(low, high).all()
    pd.testing.assert_frame_equal(pd.read_csv(op.EVALUATIONS_CSV), evaluations)
    pd.testing.assert_frame_equal(pd.read_csv(op.FRONT_CSV), front)

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
        score_settings(settings, samples, areas, schools, shares),
        op.DEFAULT_OBJECTIVES,
    )
    assert rescored == {name: best[name] for name in op.DEFAULT_OBJECTIVES}

    # Scoring the candidates in worker processes changes nothing but the time.
    pooled, _ = op.optimise(
        samples,
        areas,
        schools,
        shares,
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
# compare: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_compare_scores_and_plots_the_setting_against_no_routes(
    secondary, tmp_path, monkeypatch
):
    pngs = {
        name: tmp_path / f"{name.lower()}.png"
        for name in (
            "DISSIMILARITY_PNG",
            "UNASSIGNED_PNG",
            "INTAKE_PNG",
            "MODES_PNG",
            "LORENZ_PNG",
            "MAP_PNG",
        )
    }
    for name, png in pngs.items():
        monkeypatch.setattr(op, name, png)
    samples, areas, schools, shares = secondary
    settings = replace(DEFAULTS, capacity_scale=2.0, progressivity=0.5)

    results, intake, modes = op.compare(settings, samples, areas, schools, shares)

    rows, school_rows, mode_rows = score_settings(
        settings, samples, areas, schools, shares
    )
    pd.testing.assert_frame_equal(results, pd.DataFrame(rows))
    pd.testing.assert_frame_equal(intake, pd.DataFrame(school_rows))
    pd.testing.assert_frame_equal(modes, pd.DataFrame(mode_rows))
    assert list(results["scenario"]) == ["with routes", "without routes"]
    for png in pngs.values():
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", png


def test_compare_rejects_no_samples():
    with pytest.raises(ValueError, match="No student sample"):
        op.compare(DEFAULTS, [], gpd.GeoDataFrame(), gpd.GeoDataFrame(), pd.DataFrame())


# --------------------------------------------------------------------------
# plot_front
# --------------------------------------------------------------------------


def test_plot_front_writes_a_png_of_the_front(tmp_path, monkeypatch):
    monkeypatch.setattr(op, "FRONT_PNG", tmp_path / "out" / "front.png")
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
    )

    assert op.FRONT_PNG.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_front_rejects_anything_but_two_objectives():
    with pytest.raises(ValueError, match="plotted in two objectives"):
        op.plot_front(
            pd.DataFrame(), pd.DataFrame(), pd.Series(), {}, ["dissimilarity"]
        )
