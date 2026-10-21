from dataclasses import replace

import pandas as pd

from geo_model.measures.travel import MODES
from geo_model.model.settings import DEFAULTS
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
