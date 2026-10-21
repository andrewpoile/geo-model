from dataclasses import replace
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geo_model import optimise as op
from geo_model.analysis.search import DEFAULT_OBJECTIVES, bounds, objective_values
from geo_model.dissimilarity import score_settings
from geo_model.model.settings import DEFAULTS

# A sample of no students, for the checks made before anything is scored.
NO_STUDENTS = [
    (
        np.empty((0, 2)),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
        np.random.SeedSequence(0),
    )
]


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
        low, high = bounds(parameter)
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
    rescored = objective_values(
        score_settings(settings, samples, areas, secondary, shares),
        DEFAULT_OBJECTIVES,
    )
    assert rescored == {name: best[name] for name in DEFAULT_OBJECTIVES}

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
