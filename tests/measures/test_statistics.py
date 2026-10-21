import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from geo_model.measures.statistics import holm, routes_t_test, t_test_label

# --------------------------------------------------------------------------
# routes_t_test and holm
# --------------------------------------------------------------------------


def scenario_rows(with_routes: list[float], without_routes: list[float]):
    """Scenario rows of one setting, seed s scoring `with_routes[s]` with
    routes and `without_routes[s]` without."""
    return pd.DataFrame(
        {"seed": seed, "scenario": scenario, "dissimilarity": index}
        for seed, pair in enumerate(zip(with_routes, without_routes))
        for scenario, index in zip(("with routes", "without routes"), pair)
    )


def test_routes_t_test_is_the_paired_test_over_each_seeds_difference():
    with_routes, without_routes = [0.30, 0.28, 0.33], [0.35, 0.36, 0.34]
    difference = np.subtract(with_routes, without_routes)
    t = difference.mean() / (difference.std(ddof=1) / np.sqrt(3))

    test = routes_t_test(scenario_rows(with_routes, without_routes))

    assert test["seeds"] == 3
    assert test["difference"] == pytest.approx(difference.mean())
    assert test["t"] == pytest.approx(t)
    assert test["p"] == pytest.approx(2 * stats.t.sf(abs(t), 2))


def test_routes_t_test_pairs_the_scenarios_by_seed_not_by_row():
    rows = scenario_rows([0.30, 0.28, 0.33], [0.35, 0.36, 0.34])

    shuffled = rows.sample(frac=1, random_state=0)

    pd.testing.assert_series_equal(routes_t_test(shuffled), routes_t_test(rows))


def test_routes_t_test_is_undefined_where_routes_change_no_seed():
    test = routes_t_test(scenario_rows([0.30, 0.28], [0.30, 0.28]))

    assert test["difference"] == 0
    assert np.isnan(test["t"]) and np.isnan(test["p"])


def test_routes_t_test_is_undefined_over_one_seed_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test = routes_t_test(scenario_rows([0.30], [0.35]))

    assert test["seeds"] == 1
    assert test["difference"] == pytest.approx(-0.05)
    assert np.isnan(test["t"]) and np.isnan(test["p"])


def test_routes_t_test_rejects_a_seed_scored_in_one_scenario_alone():
    rows = scenario_rows([0.30, 0.28], [0.35, 0.36]).iloc[:-1]

    with pytest.raises(ValueError, match="one scenario alone have no pair to test: 1"):
        routes_t_test(rows)


def test_routes_t_test_rejects_a_seed_scored_twice():
    rows = scenario_rows([0.30, 0.28], [0.35, 0.36])

    with pytest.raises(ValueError, match="duplicate"):
        routes_t_test(pd.concat([rows, rows]))


def test_holm_steps_down_over_the_ordered_p_values():
    # Ordered 0.005, 0.01, 0.03, 0.04 they scale to 0.02, 0.03, 0.06, 0.04,
    # and the last takes the 0.06 before it.
    adjusted = holm(pd.Series([0.01, 0.04, 0.03, 0.005]))

    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_keeps_an_undefined_test_out_of_the_count():
    adjusted = holm(pd.Series([0.01, np.nan, 0.02], index=["a", "b", "c"]))

    assert list(adjusted.index) == ["a", "b", "c"]
    assert np.isnan(adjusted["b"])
    assert adjusted[["a", "c"]].tolist() == pytest.approx([0.02, 0.02])


def test_holm_caps_the_adjusted_p_values_at_one():
    assert holm(pd.Series([0.6, 0.9])).tolist() == [1.0, 1.0]


# --------------------------------------------------------------------------
# t_test_label
# --------------------------------------------------------------------------


def test_t_test_label_gives_the_seeds_difference_t_and_p():
    test = pd.Series({"seeds": 30, "difference": -0.0123, "t": -4.214, "p": 0.00023})

    assert t_test_label(test) == (
        "Paired t-test over 30 seeds\nmean difference -0.0123, t = -4.21, p = 0.00023"
    )


def test_t_test_label_says_why_a_test_is_undefined():
    one_seed = pd.Series({"seeds": 1, "difference": -0.05, "t": np.nan, "p": np.nan})
    unchanged = pd.Series({"seeds": 5, "difference": 0.0, "t": np.nan, "p": np.nan})

    assert t_test_label(one_seed) == ("Paired t-test over 1 seed\nundefined: one seed")
    assert t_test_label(unchanged).endswith("undefined: routes change no seed's index")
