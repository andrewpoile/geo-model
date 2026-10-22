import sys
from dataclasses import replace

import geopandas as gpd
import pandas as pd
import pytest

from geo_model.analysis.sweep import GRID, swept
from geo_model.measures.travel import MODES
from geo_model.model.settings import DEFAULTS, DISADVANTAGE
from geo_model.scripts import sweep as sw

# --------------------------------------------------------------------------
# sweep: against the real data folder
# --------------------------------------------------------------------------


def test_sweep_scores_every_cell_on_the_same_samples(
    samples, areas, secondary, shares, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sw,
        "GRID",
        {
            "capacity_scale": [
                DEFAULTS,
                replace(DEFAULTS, capacity_scale=25.0),
            ],
            "decile": [DEFAULTS],
        },
    )
    results, intake, modes = sw.sweep(samples, areas, secondary, shares, tmp_path)

    assert len(results) == 3 * 1 * 2
    assert len(intake) == 3 * 1 * 2 * len(secondary)
    assert len(modes) == 3 * 1 * 2 * len(MODES)
    for frame in (results, intake, modes):
        assert list(frame.columns[:2]) == ["parameter", "value"]
    assert results["dissimilarity"].between(0, 1).all()
    for name, frame in (
        (sw.RESULTS_CSV, results),
        (sw.SCHOOLS_CSV, intake),
        (sw.MODES_CSV, modes),
    ):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), frame)

    # Every seated student sits in exactly one school's intake.
    keys = ["parameter", "value", "seed", "scenario"]
    seated = intake.groupby(keys)[["disadvantaged", "other"]].sum().sum(axis=1)
    pd.testing.assert_series_equal(
        seated, results.set_index(keys)["n_matched"], check_names=False
    )

    # Every seated student travels by exactly one expected mode.
    travelling = modes.groupby(keys)["students"].sum()
    pd.testing.assert_series_equal(
        travelling,
        results.set_index(keys)["n_matched"].astype(float),
        check_names=False,
    )

    # Scoring the cells in worker processes changes nothing but the wall time.
    pooled = sw.sweep(samples, areas, secondary, shares, tmp_path, workers=2)
    for pooled_frame, frame in zip(pooled, (results, intake, modes)):
        pd.testing.assert_frame_equal(pooled_frame, frame)

    # Route capacity plays no part in the unrouted instance, so on the same
    # sample its score cannot move with it; only the routed score can.
    unrouted = results[results["scenario"] == "without routes"]
    assert unrouted["dissimilarity"].nunique() == 1
    routed = results[results["scenario"] == "with routes"]
    scales = routed.groupby("parameter")["dissimilarity"].nunique()
    assert scales["capacity_scale"] == 2


def test_sweep_scores_every_cell_under_the_rule_given(monkeypatch, tmp_path):
    scored = []

    def score(settings, **kwargs):
        scored.append((settings, kwargs["priority"]))
        return [], [], [], []

    monkeypatch.setattr(sw, "score_settings", score)

    sw.sweep(
        [],
        gpd.GeoDataFrame(),
        gpd.GeoDataFrame(),
        pd.DataFrame(),
        tmp_path,
        priority="bracket",
    )

    # Every cell is scored under the rule, and none of the priority
    # discount's, which the rule leaves inert.
    assert [settings for settings, _ in scored] == [
        cell
        for parameter in swept(GRID, DISADVANTAGE, "bracket")
        for cell in GRID[parameter]
    ]
    assert {priority for _, priority in scored} == {"bracket"}


def test_main_rejects_a_map_of_an_anonymised_sweep(monkeypatch):
    def load(*args):
        pytest.fail("loaded a region for a sweep it cannot run")

    monkeypatch.setattr(sw, "regions", load)
    monkeypatch.setattr(sys, "argv", ["sweep", "--anon", "--map"])

    with pytest.raises(SystemExit):
        sw.main()
