import functools

import numpy as np
import pandas as pd
import pytest

from geo_model.analysis.search import (
    DEFAULT_OBJECTIVES,
    LEVERS,
    OPTIMISABLE,
    Pipeline,
    bounds,
    car_displacement,
    cast,
    compromise,
    routed_dissimilarity,
    searchable,
)
from geo_model.model.settings import (
    DEFAULTS,
    DISADVANTAGE_CHOICES,
    PRIORITY,
    Settings,
)


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

    assert routed_dissimilarity(
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

    assert car_displacement(pd.DataFrame(), pd.DataFrame(), modes) == 7.0


# --------------------------------------------------------------------------
# bounds
# --------------------------------------------------------------------------


def test_bounds_hold_the_default_of_every_optimisable_parameter():
    assert set(LEVERS) <= set(OPTIMISABLE)
    for parameter in OPTIMISABLE:
        low, high = bounds(parameter)
        assert low < high
        assert low <= getattr(DEFAULTS, parameter) <= high


def test_bounds_search_the_decile_over_the_deciles_swept():
    assert bounds("decile") == (1.0, 9.0)


def test_bounds_rejects_a_parameter_that_is_not_swept():
    with pytest.raises(ValueError, match="circuity is not optimisable"):
        bounds("circuity")


# --------------------------------------------------------------------------
# searchable
# --------------------------------------------------------------------------


def test_searchable_leaves_the_decile_out_only_where_it_sets_the_group():
    assert "decile" in searchable("score", PRIORITY)
    assert "decile" in searchable("score-open", PRIORITY)
    assert "decile" not in searchable("decile", PRIORITY)


def test_searchable_leaves_the_other_students_route_discount_out_unless_they_ride():
    assert "route_discount" in searchable("score-open", PRIORITY)
    assert "route_discount" not in searchable("score", PRIORITY)
    assert "route_discount" not in searchable("decile", PRIORITY)


def test_searchable_leaves_the_priority_discount_out_under_brackets():
    assert "priority_discount" in searchable("score", "distance")
    assert "priority_discount" not in searchable("score", "bracket")


@pytest.mark.parametrize("disadvantage", DISADVANTAGE_CHOICES)
def test_every_lever_is_searchable_under_every_choice(disadvantage):
    assert set(LEVERS) <= set(searchable(disadvantage, PRIORITY))


# --------------------------------------------------------------------------
# cast
# --------------------------------------------------------------------------


def test_cast_rounds_the_decile_and_makes_every_other_value_a_float():
    values = cast({"decile": np.float64(2.6), "min_distance": np.float64(1609.4)})

    assert values == {"decile": 3, "min_distance": 1609.4}
    assert type(values["decile"]) is int
    assert type(values["min_distance"]) is float


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def test_pipeline_negates_maximised_objectives_and_marks_infeasible_settings():
    def score(settings: Settings):
        # Only a short minimum distance leaves a route here.
        if settings.min_distance > 4000:
            return None
        return scored_rows(settings.capacity_scale / 100, (100.0, 88.0))

    problem = Pipeline(
        ["min_distance", "capacity_scale"],
        DEFAULT_OBJECTIVES,
        functools.partial(map, score),
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

    problem = Pipeline(["decile"], DEFAULT_OBJECTIVES, functools.partial(map, score))
    problem.evaluate(np.array([[2.4], [2.6]]))

    assert scored_deciles == [2, 3]
    assert [row["decile"] for row in problem.evaluations] == [2, 3]


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

    chosen = compromise(front, DEFAULT_OBJECTIVES)

    assert chosen["min_distance"] == 200.0


def test_compromise_ignores_an_objective_the_whole_front_ties_on():
    front = pd.DataFrame(
        {
            "min_distance": [100.0, 200.0],
            "dissimilarity": [0.3, 0.3],
            "car_displacement": [10.0, 80.0],
        }
    )

    assert compromise(front, DEFAULT_OBJECTIVES)["min_distance"] == 200.0
    assert compromise(front, ["dissimilarity"])["min_distance"] == 100.0
