import numpy as np
import pandas as pd
import pytest
from frames import synthetic_frames
from matplotlib.figure import Figure

from geo_model.analysis.sweep import GRID, sweep_tests
from geo_model.plots.style import AXIS_LABELS, INTAKE_MEASURES, SIGNIFICANCE_KEY
from geo_model.plots.sweep import draw_columns, plot_sweep, positions, stars

# Two of GRID's parameters, enough to show how the columns are laid side by
# side; every parameter's positions are checked in the `positions` test.
DRAWN = ["decile", "capacity_scale"]


def draw(parameters: list[str]) -> np.ndarray:
    """Draw `parameters` on a bare figure and return its panels, row by row."""
    results, intake, modes, fsm = synthetic_frames()
    fig = Figure(figsize=(4.5 * len(parameters), 20), layout="constrained")
    draw_columns(fig, results, intake, modes, fsm, GRID, parameters)
    # The subplots come first and in row order; the colorbar follows them.
    grid = np.array(fig.axes[: 5 * (len(parameters) + 1)]).reshape(5, -1)
    return grid[:, :-1]  # the FSM strips take the last column


# --------------------------------------------------------------------------
# plot_sweep
# --------------------------------------------------------------------------


def test_plot_sweep_writes_a_png_per_parameter_and_the_matrix(tmp_path):
    out_dir = tmp_path / "out"

    plot_sweep(*synthetic_frames(), GRID, DRAWN, out_dir / "sweep.png")

    for name in [*(f"{p}.png" for p in DRAWN), "sweep.png"]:
        png = out_dir / name
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name


# --------------------------------------------------------------------------
# draw_columns
# --------------------------------------------------------------------------


def test_draw_columns_shares_one_x_axis_down_every_column():
    parameters = DRAWN

    panels = draw(parameters)

    for column, parameter in zip(panels.T, parameters):
        at = positions(GRID, parameter)
        shared = column[0].get_shared_x_axes()
        assert all(shared.joined(column[0], ax) for ax in column[1:])
        # One axis, so one set of limits and one set of ticks for the column.
        for ax in column:
            assert ax.get_xlim() == (0, len(at))
            assert list(ax.get_xticks()) == list(at)
        # Drawn once, at the foot of the column.
        assert column[-1].get_xlabel() == AXIS_LABELS[parameter]
        assert all(ax.get_xlabel() == "" for ax in column[:-1])
        assert column[-1].xaxis.get_tick_params()["labelbottom"]
        assert not any(ax.xaxis.get_tick_params()["labelbottom"] for ax in column[:-1])


@pytest.mark.parametrize("measure", list(INTAKE_MEASURES))
def test_draw_columns_puts_the_intake_panels_and_fsm_strip_on_one_scale(measure):
    results, intake, modes, fsm = synthetic_frames()
    fig = Figure(figsize=(9, 20), layout="constrained")

    draw_columns(fig, results, intake, modes, fsm, GRID, ["decile"], measure)

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
    assert label == INTAKE_MEASURES[measure]["label"]


def test_draw_columns_rejects_a_value_the_grid_does_not_sweep():
    results, intake, modes, fsm = synthetic_frames()
    stray = results[results["parameter"] == "decile"].head(1).assign(value=-1)

    with pytest.raises(
        ValueError, match="decile carries values the grid does not sweep"
    ):
        draw_columns(
            Figure(), pd.concat([results, stray]), intake, modes, fsm, GRID, ["decile"]
        )


# --------------------------------------------------------------------------
# stars and the index panels' stars
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "mark"),
    [
        (0.0005, "***"),
        (0.001, "**"),
        (0.005, "**"),
        (0.01, "*"),
        (0.049, "*"),
        (0.05, "ns"),
        (0.9, "ns"),
        (np.nan, "–"),
    ],
)
def test_stars_mark_the_level_a_p_value_falls_below(p, mark):
    assert stars(p) == mark


def test_draw_columns_stars_every_value_by_its_test_over_the_whole_frame():
    results, intake, modes, fsm = synthetic_frames()
    # Routes lower every decile cell's index by 0.1 in seed 0 and 0.1001 in
    # seed 1, so t = -2001 over 1 degree of freedom and p = 3.2e-4: "***"
    # alone, "**" corrected over the decile's 9 cells and "*" over every
    # cell of GRID.
    decile = results["parameter"] == "decile"
    unrouted = results[decile & (results["scenario"] == "without routes")]
    routed = decile & (results["scenario"] == "with routes")
    results.loc[routed, "dissimilarity"] = (
        unrouted["dissimilarity"].to_numpy() - 0.1 - 1e-4 * unrouted["seed"].to_numpy()
    )
    fig = Figure(figsize=(9, 20), layout="constrained")

    draw_columns(fig, results, intake, modes, fsm, GRID, ["decile"])

    marks = fig.axes[0].texts
    assert [mark.get_text() for mark in marks] == ["*"] * 9
    corrected = sweep_tests(results)["p_holm"].xs("decile")
    at = positions(GRID, "decile")
    assert [mark.get_position()[0] for mark in marks] == list(at[corrected.index])
    assert fig.legends[0].get_title().get_text() == SIGNIFICANCE_KEY


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------


def test_positions_centre_every_grid_value_on_its_heatmap_cell():
    for parameter, cells in GRID.items():
        at = positions(GRID, parameter)

        assert list(at.index) == [getattr(cell, parameter) for cell in cells]
        assert list(at) == [i + 0.5 for i in range(len(cells))]
