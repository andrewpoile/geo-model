from dataclasses import replace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from frames import default_intake, synthetic_frames
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from shapely import box

from geo_model import sweep as sw
from geo_model.analysis.sweep import GRID, sweep_tests
from geo_model.data import sources
from geo_model.measures.segregation import gini, lorenz_curve
from geo_model.measures.travel import MODES

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
                sw.DEFAULTS,
                replace(sw.DEFAULTS, capacity_scale=25.0),
            ],
            "decile": [sw.DEFAULTS],
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


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


# Two of GRID's parameters, enough to show how the columns are laid side by
# side; every parameter's positions are checked in the `positions` test.
DRAWN = ["decile", "capacity_scale"]


def draw(parameters: list[str]) -> np.ndarray:
    """Draw `parameters` on a bare figure and return its panels, row by row."""
    results, intake, modes, fsm = synthetic_frames()
    fig = Figure(figsize=(4.5 * len(parameters), 20), layout="constrained")
    sw.draw_columns(fig, results, intake, modes, fsm, parameters)
    # The subplots come first and in row order; the colorbar follows them.
    grid = np.array(fig.axes[: 5 * (len(parameters) + 1)]).reshape(5, -1)
    return grid[:, :-1]  # the FSM strips take the last column


def test_plot_writes_a_png_per_parameter_and_the_matrix(tmp_path):
    out_dir = tmp_path / "out"

    sw.plot(*synthetic_frames(), DRAWN, out_dir)

    for name in [*(f"{p}.png" for p in DRAWN), sw.PLOT_PNG]:
        png = out_dir / name
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name


def test_draw_columns_shares_one_x_axis_down_every_column():
    parameters = DRAWN

    panels = draw(parameters)

    for column, parameter in zip(panels.T, parameters):
        at = sw.positions(parameter)
        shared = column[0].get_shared_x_axes()
        assert all(shared.joined(column[0], ax) for ax in column[1:])
        # One axis, so one set of limits and one set of ticks for the column.
        for ax in column:
            assert ax.get_xlim() == (0, len(at))
            assert list(ax.get_xticks()) == list(at)
        # Drawn once, at the foot of the column.
        assert column[-1].get_xlabel() == sw.AXIS_LABELS[parameter]
        assert all(ax.get_xlabel() == "" for ax in column[:-1])
        assert column[-1].xaxis.get_tick_params()["labelbottom"]
        assert not any(ax.xaxis.get_tick_params()["labelbottom"] for ax in column[:-1])


@pytest.mark.parametrize("measure", list(sw.INTAKE_MEASURES))
def test_draw_columns_puts_the_intake_panels_and_fsm_strip_on_one_scale(measure):
    results, intake, modes, fsm = synthetic_frames()
    fig = Figure(figsize=(9, 20), layout="constrained")

    sw.draw_columns(fig, results, intake, modes, fsm, ["decile"], measure)

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
    assert label == sw.INTAKE_MEASURES[measure]["label"]


def test_draw_columns_rejects_a_value_the_grid_does_not_sweep():
    results, intake, modes, fsm = synthetic_frames()
    stray = results[results["parameter"] == "decile"].head(1).assign(value=-1)

    with pytest.raises(ValueError, match="decile carries values GRID does not sweep"):
        sw.draw_columns(
            Figure(), pd.concat([results, stray]), intake, modes, fsm, ["decile"]
        )


# --------------------------------------------------------------------------
# sweep_tests, stars and the index panels' stars
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
    assert sw.stars(p) == mark


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

    sw.draw_columns(fig, results, intake, modes, fsm, ["decile"])

    marks = fig.axes[0].texts
    assert [mark.get_text() for mark in marks] == ["*"] * 9
    corrected = sweep_tests(results)["p_holm"].xs("decile")
    at = sw.positions("decile")
    assert [mark.get_position()[0] for mark in marks] == list(at[corrected.index])
    assert fig.legends[0].get_title().get_text() == sw.SIGNIFICANCE_KEY


# --------------------------------------------------------------------------
# draw_map
# --------------------------------------------------------------------------


def synthetic_city():
    """Two 10km districts, two schools and four students, matched both ways.

    With routes, student 0 is seated at school 0, student 1 at school 1 on
    route 0 and student 2 at school 1, while student 3 is left unassigned.
    Without routes, student 1 is seated at school 0 instead.
    """
    areas = gpd.GeoDataFrame(
        {"IDACI Decile": [1, 10]},
        geometry=[box(0, 0, 10_000, 10_000), box(10_000, 0, 20_000, 10_000)],
        crs=sources.CRS,
    )
    schools = pd.DataFrame(
        {
            "Easting": [5_000.0, 15_000.0],
            "Northing": [5_000.0, 5_000.0],
            "P8MEA": [-0.5, 0.25],
        }
    )
    student_xy = np.array(
        [
            [1_000.0, 1_000.0],
            [2_000.0, 8_000.0],
            [12_000.0, 3_000.0],
            [18_000.0, 8_000.0],
        ]
    )
    matchings = {
        "with routes": np.array([[0, -1], [1, 0], [1, -1], [-1, -1]]),
        "without routes": np.array([[0, -1], [0, -1], [1, -1], [-1, -1]]),
    }
    return areas, schools, student_xy, matchings


def test_draw_map_links_every_seated_student_to_their_school():
    areas, schools, student_xy, matchings = synthetic_city()
    fig = Figure(figsize=(16, 8), layout="constrained")

    sw.draw_map(fig, areas, schools, student_xy, matchings)

    school_xy = schools[["Easting", "Northing"]].to_numpy()
    linked = {
        "with routes": {"without a route": [0, 2], "on a route": [1]},
        "without routes": {"without a route": [0, 1, 2], "on a route": []},
    }
    # The panels come first and in scenario order; the colorbars follow them.
    for ax, (scenario, matching) in zip(fig.axes, matchings.items()):
        assert ax.get_title() == scenario
        drawn = {collection.get_label(): collection for collection in ax.collections}
        deciles = drawn["districts"].get_array()
        assert deciles is not None and list(deciles) == [1, 10]
        assert drawn["districts"].get_clim() == (0.5, 10.5)
        for label, students in linked[scenario].items():
            links = drawn[label]
            assert isinstance(links, LineCollection)
            assert [s.tolist() for s in links.get_segments()] == [
                [student_xy[i].tolist(), school_xy[matching[i, 0]].tolist()]
                for i in students
            ]
        np.testing.assert_array_equal(drawn["seated"].get_offsets(), student_xy[:3])
        np.testing.assert_array_equal(drawn["unassigned"].get_offsets(), student_xy[3:])
        scores = drawn["schools"].get_array()
        assert scores is not None and list(scores) == [-0.5, 0.25]
        # Centred on the England average, reaching the largest score either way.
        assert drawn["schools"].get_clim() == (-0.5, 0.5)
        # A metre is as long north as east, so the scale holds in every direction.
        assert ax.get_aspect() == 1.0
        # Kilometres ticked above the line and miles below, from one origin.
        origins = []
        for (unit, (metres, length)), side in zip(sw.SCALE_BAR.items(), (1, -1)):
            scale = drawn[unit]
            assert isinstance(scale, LineCollection)
            (start, end), *ticks = scale.get_segments()
            assert start[1] == end[1]
            assert end[0] - start[0] == pytest.approx(length * metres)
            assert [tick[0, 0] - start[0] for tick in ticks] == pytest.approx(
                [i * metres for i in range(length + 1)]
            )
            assert all(np.sign(tick[1, 1] - tick[0, 1]) == side for tick in ticks)
            origins.append(start.tolist())
        assert origins[0] == origins[1]
        labels = ["0", "1", "2", "3 km", "0", "1", "2 miles"]
        assert [text.get_text() for text in ax.texts] == labels


def test_plot_map_writes_a_png(tmp_path):
    png = tmp_path / "out" / "map.png"

    sw.plot_map(*synthetic_city(), png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# draw_lorenz
# --------------------------------------------------------------------------


def test_draw_lorenz_draws_every_seed_and_its_gini_per_scenario():
    _, intake, _, _ = synthetic_frames()
    fig = Figure(figsize=(12, 6.5), layout="constrained")

    default = default_intake(intake)
    sw.draw_lorenz(fig, default)
    for ax, scenario in zip(fig.axes, sw.COLOURS):
        assert ax.get_title() == scenario
        seeds = default[default["scenario"] == scenario].groupby("seed")
        lines = {line.get_label(): line for line in ax.get_lines()}
        assert set(lines) == {"equality", *(f"seed {seed}" for seed, _ in seeds)}
        coefficients = []
        for seed, rows in seeds:
            x, y = lorenz_curve(rows["disadvantaged"], rows["other"])
            np.testing.assert_allclose(
                np.asarray(lines[f"seed {seed}"].get_xydata()), np.column_stack((x, y))
            )
            coefficients.append(gini(x, y))
        (text,) = ax.texts
        assert text.get_text().startswith(f"Gini {np.mean(coefficients):.3f} (mean)")


@pytest.mark.parametrize(
    ("disadvantage", "group"),
    [
        ("score", "drawn from LSOA IDACI scores"),
        ("score-open", "drawn from LSOA IDACI scores"),
        ("decile", f"IDACI decile ≤ {sw.DEFAULTS.decile}"),
    ],
)
def test_draw_lorenz_names_the_group_the_rows_were_scored_on(disadvantage, group):
    fig = Figure()

    sw.draw_lorenz(fig, default_intake(synthetic_frames()[1]), disadvantage)

    assert fig.axes[0].get_ylabel() == (
        f"Cumulative share of disadvantaged students ({group})"
    )


def test_plot_lorenz_writes_a_png(tmp_path):
    png = tmp_path / "out" / "lorenz.png"

    sw.plot_lorenz(default_intake(synthetic_frames()[1]), png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------


def test_positions_centre_every_grid_value_on_its_heatmap_cell():
    for parameter, cells in GRID.items():
        at = sw.positions(parameter)

        assert list(at.index) == [getattr(cell, parameter) for cell in cells]
        assert list(at) == [i + 0.5 for i in range(len(cells))]
