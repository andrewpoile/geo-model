from dataclasses import replace
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geo_model.analysis.search import DEFAULT_OBJECTIVES, bounds, objective_values
from geo_model.model.settings import DEFAULTS
from geo_model.scripts import optimise as op
from geo_model.scripts.run import score_settings

# A sample of no students, for the checks made before anything is scored.
NO_STUDENTS = [
    (
        np.empty((0, 2)),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
        np.random.SeedSequence(0),
        np.empty((0, 0)),
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
            np.empty((0, 0)),
            np.empty((0, 0)),
            pd.DataFrame(),
            tmp_path,
            parameters=["min_distance", "min_distance"],
            distance="euclidean",
        )


@pytest.mark.parametrize(
    ("parameter", "disadvantage", "priority"),
    [
        ("decile", "decile", "distance"),
        ("route_discount", "score", "distance"),
        ("route_discount", "decile", "distance"),
        ("priority_discount", "score", "bracket"),
    ],
)
def test_optimise_rejects_a_parameter_the_choices_bar(
    parameter, disadvantage, priority, tmp_path
):
    with pytest.raises(ValueError, match=rf"\['{parameter}'\] cannot be searched"):
        op.optimise(
            NO_STUDENTS,
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            np.empty((0, 0)),
            np.empty((0, 0)),
            pd.DataFrame(),
            tmp_path,
            parameters=[parameter],
            rules=replace(DEFAULTS, disadvantage=disadvantage, priority=priority),
            distance="euclidean",
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
            np.empty((0, 0)),
            np.empty((0, 0)),
            pd.DataFrame(),
            tmp_path,
            pop_size=4,
            generations=2,
            distance="euclidean",
        )


def test_optimise_scores_every_candidate_under_the_rule_given(monkeypatch, tmp_path):
    rules = []

    def score(settings, **kwargs):
        rules.append(settings.priority)

    monkeypatch.setattr(op, "score_feasible", score)

    with pytest.raises(ValueError, match="None of the 4 settings scored"):
        op.optimise(
            NO_STUDENTS,
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            np.empty((0, 0)),
            np.empty((0, 0)),
            pd.DataFrame(),
            tmp_path,
            pop_size=4,
            generations=1,
            rules=replace(DEFAULTS, priority="bracket"),
            distance="euclidean",
        )
    assert rules == ["bracket"] * 4


# --------------------------------------------------------------------------
# score_feasible and optimise: against the real data folder
# --------------------------------------------------------------------------


def test_score_feasible_names_a_setting_that_leaves_no_route(
    samples, areas, secondary, district_distance, district_minutes, shares, capsys
):
    # Every Southampton district lies within 9.9km of every school.
    settings = replace(DEFAULTS, min_distance=20_000)

    scored = op.score_feasible(
        settings,
        samples=samples,
        areas=areas,
        secondary_schools=secondary,
        district_distance=district_distance,
        district_minutes=district_minutes,
        shares=shares,
        distance="euclidean",
    )

    assert scored is None
    out = capsys.readouterr().out
    assert f"Infeasible, no route at {settings}" in out
    assert "so the route set is empty" in out


def test_optimise_finds_the_front_of_every_setting_scored(
    samples, areas, secondary, district_distance, district_minutes, shares, tmp_path
):
    parameters = ["capacity_scale", "progressivity"]

    evaluations, front = op.optimise(
        samples,
        areas,
        secondary,
        district_distance,
        district_minutes,
        shares,
        tmp_path,
        parameters,
        pop_size=4,
        generations=2,
        distance="euclidean",
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
        score_settings(
            settings,
            samples,
            areas,
            secondary,
            district_distance,
            district_minutes,
            shares,
            distance="euclidean",
        ),
        DEFAULT_OBJECTIVES,
    )
    assert rescored == {name: best[name] for name in DEFAULT_OBJECTIVES}

    # Scoring the candidates in worker processes changes nothing but the time.
    pooled, _ = op.optimise(
        samples,
        areas,
        secondary,
        district_distance,
        district_minutes,
        shares,
        tmp_path / "pooled",
        parameters,
        pop_size=4,
        generations=2,
        workers=2,
        distance="euclidean",
    )
    pd.testing.assert_frame_equal(pooled, evaluations)
