from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from geo_model import __main__ as sweep_main
from geo_model import build_prefs as bp
from geo_model import load_data as ld

DATA = ld.POPULATION_XLSX.parent.parent


# --------------------------------------------------------------------------
# GRID
# --------------------------------------------------------------------------


def test_every_grid_varies_its_own_parameter_alone_and_holds_the_default():
    for parameter, cells in sweep_main.GRID.items():
        assert sweep_main.DEFAULTS in cells
        for cell in cells:
            others = {k: v for k, v in asdict(cell).items() if k != parameter}
            defaults = {
                k: v for k, v in asdict(sweep_main.DEFAULTS).items() if k != parameter
            }
            assert others == defaults


# --------------------------------------------------------------------------
# sweep: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_sweep_scores_every_cell_on_the_same_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_main, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(sweep_main, "RESULTS_CSV", tmp_path / "sweep.csv")
    monkeypatch.setattr(sweep_main, "SCHOOLS_CSV", tmp_path / "schools.csv")
    monkeypatch.setattr(
        sweep_main,
        "GRID",
        {
            "capacity": [
                replace(sweep_main.DEFAULTS, capacity=1),
                sweep_main.DEFAULTS,
            ],
            "decile": [sweep_main.DEFAULTS],
        },
    )
    areas = ld.load_areas()
    _, schools = ld.load_schools()
    samples = list(
        sweep_main.student_samples(areas, bp.cohort_sizes(areas, "secondary"), 1)
    )

    results, intake = sweep_main.sweep(samples, areas, schools)

    assert len(results) == 3 * 1 * 2
    assert len(intake) == 3 * 1 * 2 * len(schools)
    for frame in (results, intake):
        assert list(frame.columns[:2]) == ["parameter", "value"]
    assert results["dissimilarity"].between(0, 1).all()
    pd.testing.assert_frame_equal(pd.read_csv(sweep_main.RESULTS_CSV), results)
    pd.testing.assert_frame_equal(pd.read_csv(sweep_main.SCHOOLS_CSV), intake)

    # Every seated student sits in exactly one school's intake.
    keys = ["parameter", "value", "seed", "scenario"]
    seated = intake.groupby(keys)[["disadvantaged", "other"]].sum().sum(axis=1)
    pd.testing.assert_series_equal(
        seated, results.set_index(keys)["n_matched"], check_names=False
    )

    # Scoring the cells in worker processes changes nothing but the wall time.
    pooled_results, pooled_intake = sweep_main.sweep(samples, areas, schools, workers=2)
    pd.testing.assert_frame_equal(pooled_results, results)
    pd.testing.assert_frame_equal(pooled_intake, intake)

    # Route capacity plays no part in the unrouted instance, so on the same
    # sample its score cannot move with it; only the routed score can.
    unrouted = results[results["scenario"] == "without routes"]
    assert unrouted["dissimilarity"].nunique() == 1
    routed = results[results["scenario"] == "with routes"]
    assert routed.groupby("parameter")["dissimilarity"].nunique()["capacity"] == 2


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def test_plot_writes_a_png_per_parameter_and_the_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_main, "SWEEP_DIR", tmp_path / "out")
    monkeypatch.setattr(sweep_main, "PLOT_PNG", tmp_path / "out" / "sweep.png")
    rng = np.random.default_rng(0)
    cells = [
        {
            "parameter": parameter,
            "value": getattr(cell, parameter),
            "seed": seed,
            "scenario": scenario,
        }
        for parameter, grid in sweep_main.GRID.items()
        for cell in grid
        for seed in range(2)
        for scenario in ("with routes", "without routes")
    ]
    results = pd.DataFrame(
        {**cell, "dissimilarity": rng.uniform(0, 1)} for cell in cells
    )
    intake = pd.DataFrame(
        {
            **cell,
            "school": school,
            "disadvantaged": rng.integers(0, 50),
            "other": rng.integers(1, 200),
        }
        for cell in cells
        for school in ("Alpha School", "Beta School", "Gamma School")
    )

    # A school without an FSM figure draws blank rather than failing.
    fsm = pd.Series({"Alpha School": 0.4, "Beta School": np.nan, "Gamma School": 0.1})

    sweep_main.plot(results, intake, fsm)

    for name in [*sweep_main.GRID, "sweep"]:
        png = sweep_main.SWEEP_DIR / f"{name}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name


# --------------------------------------------------------------------------
# fsm_share
# --------------------------------------------------------------------------


def test_fsm_share_scales_the_register_percentage_and_names_a_missing_one(capsys):
    schools = pd.DataFrame(
        {
            "EstablishmentName": ["Alpha School", "Beta School"],
            "PercentageFSM": [55.7, np.nan],
        }
    )

    fsm = sweep_main.fsm_share(schools)

    assert fsm["Alpha School"] == pytest.approx(0.557)
    assert np.isnan(fsm["Beta School"])
    assert "drawn blank: Beta School" in capsys.readouterr().out
