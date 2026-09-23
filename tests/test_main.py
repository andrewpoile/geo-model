from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from geo_model import __main__ as sweep_main
from geo_model import build_prefs as bp
from geo_model import load_data as ld
from geo_model.utils import MODES

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
    monkeypatch.setattr(sweep_main, "MODES_CSV", tmp_path / "modes.csv")
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

    shares = ld.load_nts_mode_shares()

    results, intake, modes = sweep_main.sweep(samples, areas, schools, shares)

    assert len(results) == 3 * 1 * 2
    assert len(intake) == 3 * 1 * 2 * len(schools)
    assert len(modes) == 3 * 1 * 2 * len(MODES)
    for frame in (results, intake, modes):
        assert list(frame.columns[:2]) == ["parameter", "value"]
    assert results["dissimilarity"].between(0, 1).all()
    pd.testing.assert_frame_equal(pd.read_csv(sweep_main.RESULTS_CSV), results)
    pd.testing.assert_frame_equal(pd.read_csv(sweep_main.SCHOOLS_CSV), intake)
    pd.testing.assert_frame_equal(pd.read_csv(sweep_main.MODES_CSV), modes)

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
    pooled = sweep_main.sweep(samples, areas, schools, shares, workers=2)
    for pooled_frame, frame in zip(pooled, (results, intake, modes)):
        pd.testing.assert_frame_equal(pooled_frame, frame)

    # Route capacity plays no part in the unrouted instance, so on the same
    # sample its score cannot move with it; only the routed score can.
    unrouted = results[results["scenario"] == "without routes"]
    assert unrouted["dissimilarity"].nunique() == 1
    routed = results[results["scenario"] == "with routes"]
    assert routed.groupby("parameter")["dissimilarity"].nunique()["capacity"] == 2
    # More seats on every route carry more students by route.
    riders = modes[(modes["scenario"] == "with routes") & (modes["mode"] == "route")]
    riders = riders[riders["parameter"] == "capacity"].set_index("value")["students"]
    assert riders[1] < riders[sweep_main.DEFAULTS.capacity]


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def synthetic_frames():
    """Frames shaped like the ones `sweep` returns, covering every cell of GRID.

    A school without an FSM figure is included, so the plots are exercised on
    one that draws blank rather than failing.
    """
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
        {
            **cell,
            "dissimilarity": rng.uniform(0, 1),
            "n_matched": 900,
            "n_unmatched": 100,
            "n_disadvantaged": 300,
            "unassigned_disadvantaged": int(rng.integers(0, 100)),
        }
        for cell in cells
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
    # Each school's term of its matching's index, as `score_sample` gives it.
    totals = intake.groupby(["parameter", "value", "seed", "scenario"])[
        ["disadvantaged", "other"]
    ].transform("sum")
    intake["dissimilarity_term"] = (
        intake["disadvantaged"] / totals["disadvantaged"]
        - intake["other"] / totals["other"]
    )

    modes = pd.DataFrame(
        {**cell, "mode": mode, "students": rng.uniform(0, 500)}
        for cell in cells
        for mode in MODES
    )

    fsm = pd.Series({"Alpha School": 0.4, "Beta School": np.nan, "Gamma School": 0.1})
    return results, intake, modes, fsm


def draw(parameters: list[str]) -> np.ndarray:
    """Draw `parameters` on a bare figure and return its panels, row by row."""
    results, intake, modes, fsm = synthetic_frames()
    fig = Figure(figsize=(4.5 * len(parameters), 20), layout="constrained")
    sweep_main.draw_columns(fig, results, intake, modes, fsm, parameters)
    # The subplots come first and in row order; the colorbar follows them.
    grid = np.array(fig.axes[: 5 * (len(parameters) + 1)]).reshape(5, -1)
    return grid[:, :-1]  # the FSM strips take the last column


@pytest.mark.parametrize("measure", list(sweep_main.INTAKE_MEASURES))
def test_plot_writes_a_png_per_parameter_and_the_matrix(tmp_path, monkeypatch, measure):
    monkeypatch.setattr(sweep_main, "SWEEP_DIR", tmp_path / "out")
    monkeypatch.setattr(sweep_main, "PLOT_PNG", tmp_path / "out" / "sweep.png")

    sweep_main.plot(*synthetic_frames(), measure)

    for name in [*sweep_main.GRID, "sweep"]:
        png = sweep_main.SWEEP_DIR / f"{name}.png"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name


def test_draw_columns_shares_one_x_axis_down_every_column():
    parameters = list(sweep_main.GRID)

    panels = draw(parameters)

    for column, parameter in zip(panels.T, parameters):
        at = sweep_main.positions(parameter)
        shared = column[0].get_shared_x_axes()
        assert all(shared.joined(column[0], ax) for ax in column[1:])
        # One axis, so one set of limits and one set of ticks for the column.
        for ax in column:
            assert ax.get_xlim() == (0, len(at))
            assert list(ax.get_xticks()) == list(at)
        # Drawn once, at the foot of the column.
        assert column[-1].get_xlabel() == sweep_main.AXIS_LABELS[parameter]
        assert all(ax.get_xlabel() == "" for ax in column[:-1])
        assert column[-1].xaxis.get_tick_params()["labelbottom"]
        assert not any(ax.xaxis.get_tick_params()["labelbottom"] for ax in column[:-1])


@pytest.mark.parametrize("measure", list(sweep_main.INTAKE_MEASURES))
def test_draw_columns_puts_the_intake_panels_and_fsm_strip_on_one_scale(measure):
    results, intake, modes, fsm = synthetic_frames()
    fig = Figure(figsize=(9, 20), layout="constrained")

    sweep_main.draw_columns(fig, results, intake, modes, fsm, ["decile"], measure)

    grid = np.array(fig.axes[:10]).reshape(5, 2)
    norms = [ax.collections[0].norm for ax in grid[2:4].flat]
    assert len({(norm.vmin, norm.vmax) for norm in norms}) == 1
    norm = norms[0]
    if measure == "share":
        assert (norm.vmin, norm.vmax) == (0, 1)
    else:
        # Centred on 0 and reaching the largest seed-mean term of any
        # parameter, not only the one drawn.
        keys = ["parameter", "value", "scenario", "school"]
        largest = intake.groupby(keys)["dissimilarity_term"].mean().abs().max()
        assert norm.vmin == -norm.vmax
        assert norm.vmax == pytest.approx(max(largest, fsm.abs().max()))
    label = fig.axes[-1].get_ylabel()
    assert label == sweep_main.INTAKE_MEASURES[measure]["label"]


def test_draw_columns_rejects_a_value_the_grid_does_not_sweep():
    results, intake, modes, fsm = synthetic_frames()
    stray = results[results["parameter"] == "decile"].head(1).assign(value=-1)

    with pytest.raises(ValueError, match="decile carries values GRID does not sweep"):
        sweep_main.draw_columns(
            Figure(), pd.concat([results, stray]), intake, modes, fsm, ["decile"]
        )


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------


def test_positions_centre_every_grid_value_on_its_heatmap_cell():
    for parameter, cells in sweep_main.GRID.items():
        at = sweep_main.positions(parameter)

        assert list(at.index) == [getattr(cell, parameter) for cell in cells]
        assert list(at) == [i + 0.5 for i in range(len(cells))]


# --------------------------------------------------------------------------
# unassigned_rows
# --------------------------------------------------------------------------


def test_unassigned_rows_splits_each_scenario_into_its_two_groups():
    results = pd.DataFrame(
        {
            "scenario": ["with routes", "without routes"],
            "n_matched": [900, 800],
            "n_unmatched": [100, 200],
            "n_disadvantaged": [400, 400],
            "unassigned_disadvantaged": [40, 150],
        }
    )

    rows = sweep_main.unassigned_rows(results).set_index("series")

    assert set(rows.index) == set(sweep_main.GROUP_COLOURS)
    # Every unplaced student falls in exactly one group of their scenario.
    counts = rows["unassigned"]
    assert counts["disadvantaged, with routes"] == 40
    assert counts["other, with routes"] == 60
    assert counts["disadvantaged, without routes"] == 150
    assert counts["other, without routes"] == 50


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


# --------------------------------------------------------------------------
# fsm_term
# --------------------------------------------------------------------------


def test_fsm_term_takes_the_register_counts_over_the_schools_holding_them(capsys):
    # Alpha takes its 50% over 40 of its 60 pupils, as a school with a sixth
    # form does, so 20 are eligible and 20 are not; Gamma has 10 of 40.
    schools = pd.DataFrame(
        {
            "EstablishmentName": ["Alpha School", "Beta School", "Gamma School"],
            "FSM": [20, np.nan, 10],
            "PercentageFSM": [50.0, np.nan, 25.0],
        }
    )

    terms = sweep_main.fsm_term(schools)

    # 30 eligible and 50 not over Alpha and Gamma alone: Beta holds no figures.
    assert terms["Alpha School"] == pytest.approx(20 / 30 - 20 / 50)
    assert terms["Gamma School"] == pytest.approx(10 / 30 - 30 / 50)
    assert np.isnan(terms["Beta School"])
    assert "left out of the FSM totals: Beta School" in capsys.readouterr().out


def test_fsm_term_rejects_a_school_recording_no_pupil_eligible():
    schools = pd.DataFrame(
        {
            "EstablishmentName": ["Alpha School", "Beta School"],
            "FSM": [20, 0],
            "PercentageFSM": [50.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="cannot be recovered: Beta School"):
        sweep_main.fsm_term(schools)
