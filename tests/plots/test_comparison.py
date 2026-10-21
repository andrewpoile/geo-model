import numpy as np
import pandas as pd
import pytest
from frames import default_intake, local_rows, route_rows, synthetic_frames
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from geo_model.measures.segregation import gini, lorenz_curve
from geo_model.measures.summaries import (
    local_acceptance,
    local_difference,
    route_utilisation,
)
from geo_model.measures.travel import MODES
from geo_model.model.settings import DEFAULTS
from geo_model.plots.comparison import (
    MAX_LABELLED_ROUTES,
    draw_local_acceptance,
    draw_lorenz,
    draw_route_utilisation,
    draw_school_intake,
    draw_unassigned,
    plot_dissimilarity,
    plot_lorenz,
    plot_modes,
)
from geo_model.plots.style import COLOURS, GROUP_COLOURS, INTAKE_MEASURES

# --------------------------------------------------------------------------
# plot_dissimilarity
# --------------------------------------------------------------------------


def test_plot_dissimilarity_writes_a_png_of_the_results(tmp_path):
    png = tmp_path / "out" / "dissimilarity.png"
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "dissimilarity": [0.30, 0.35, 0.28, 0.36],
        }
    )

    plot_dissimilarity(results, png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_dissimilarity_fits_the_y_axis_to_the_indices(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(
        Figure, "savefig", lambda fig, *args, **kwargs: saved.append(fig)
    )
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "dissimilarity": [0.30, 0.35, 0.28, 0.36],
        }
    )

    plot_dissimilarity(results, tmp_path / "dissimilarity.png")

    low, high = saved[0].axes[0].get_ylim()
    # Every index is in view, on an axis far narrower than the index's 0 to 1.
    assert 0 < low <= 0.28 and 0.36 <= high
    assert high - low < 0.2


def test_plot_dissimilarity_draws_one_seed_with_its_test_undefined(tmp_path):
    png = tmp_path / "dissimilarity.png"
    results = pd.DataFrame(
        {
            "seed": [0, 0],
            "scenario": ["with routes", "without routes"],
            "dissimilarity": [0.30, 0.35],
        }
    )

    plot_dissimilarity(results, png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# plot_modes
# --------------------------------------------------------------------------


def test_plot_modes_writes_a_png_of_the_results(tmp_path):
    png = tmp_path / "out" / "car_displacement.png"
    results = pd.DataFrame(
        [
            {"seed": seed, "scenario": scenario, "mode": mode, "students": 10.0}
            for seed in (0, 1)
            for scenario in ("with routes", "without routes")
            for mode in MODES
        ]
    )

    plot_modes(results, png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# draw_unassigned
# --------------------------------------------------------------------------


def test_draw_unassigned_stacks_each_groups_mean_over_seeds_per_scenario():
    # With routes 3 disadvantaged and 8 others go unplaced on average, without
    # them 7 and 4.
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "n_unmatched": [10, 10, 12, 12],
            "unassigned_disadvantaged": [2, 8, 4, 6],
        }
    )
    ax = Figure().subplots()

    draw_unassigned(ax, results)

    bars = [
        (patch.get_x() + patch.get_width() / 2, patch.get_y(), patch.get_height())
        for patch in ax.patches
        if isinstance(patch, Rectangle)
    ]
    # Disadvantaged at the foot of both bars, then the others on top.
    assert bars == [(0, 0, 3), (1, 0, 7), (0, 3, 8), (1, 7, 4)]
    assert [t.get_text() for t in ax.get_xticklabels()] == list(COLOURS)
    colours = [to_hex(patch.get_facecolor()) for patch in ax.patches]
    assert colours == [
        GROUP_COLOURS[f"{group}, {scenario}"]
        for group in ("disadvantaged", "other")
        for scenario in COLOURS
    ]


# --------------------------------------------------------------------------
# draw_school_intake
# --------------------------------------------------------------------------


def school_rows() -> pd.DataFrame:
    """Two seeds of three schools, each school's disadvantaged share of its
    intake averaging 0.4, 0.25 and 0 with routes, 0.15, 0.75 and 0.4
    without."""
    counts = {
        # (seed, scenario): disadvantaged and other at Alpha, Beta and Gamma.
        (0, "with routes"): ([10, 5, 0], [10, 15, 20]),
        (1, "with routes"): ([6, 5, 0], [14, 15, 20]),
        (0, "without routes"): ([2, 16, 8], [18, 4, 12]),
        (1, "without routes"): ([4, 14, 8], [16, 6, 12]),
    }
    rows = pd.DataFrame(
        {
            "seed": seed,
            "scenario": scenario,
            "school": school,
            "disadvantaged": a,
            "other": b,
        }
        for (seed, scenario), (group_a, group_b) in counts.items()
        for school, a, b in zip(("Alpha", "Beta", "Gamma"), group_a, group_b)
    )
    totals = rows.groupby(["seed", "scenario"])[["disadvantaged", "other"]]
    totals = totals.transform("sum")
    return rows.assign(
        dissimilarity_term=rows["disadvantaged"] / totals["disadvantaged"]
        - rows["other"] / totals["other"]
    )


def drawn_table(fig: Figure) -> tuple[np.ndarray, list[str], list[str]]:
    """The heatmap's cells, row and column labels, as drawn."""
    ax = fig.axes[0]
    cells = ax.collections[0].get_array()
    assert isinstance(cells, np.ma.MaskedArray)
    return (
        cells.filled(np.nan).reshape(3, 3),
        [t.get_text() for t in ax.get_yticklabels()],
        [t.get_text() for t in ax.get_xticklabels()],
    )


def test_draw_school_intake_orders_schools_by_their_share_without_routes():
    fsm = pd.Series({"Alpha": 0.3, "Beta": np.nan, "Gamma": 0.5})
    fig = Figure(layout="constrained")

    draw_school_intake(fig, school_rows(), fsm, "share")

    cells, schools, columns = drawn_table(fig)
    assert schools == ["Beta", "Gamma", "Alpha"]
    assert columns == ["with\nroutes", "without\nroutes", "FSM"]
    np.testing.assert_allclose(
        cells, [[0.25, 0.75, np.nan], [0.0, 0.4, 0.5], [0.4, 0.15, 0.3]]
    )
    norm = fig.axes[0].collections[0].norm
    assert (norm.vmin, norm.vmax) == (0, 1)
    assert fig.axes[-1].get_ylabel() == INTAKE_MEASURES["share"]["label"]


def test_draw_school_intake_centres_the_terms_on_the_largest_drawn():
    rows = school_rows()
    # Larger than any school's term, so the register's figure sets the scale.
    fsm = pd.Series({"Alpha": -0.9, "Beta": 0.4, "Gamma": 0.5})
    fig = Figure(layout="constrained")

    draw_school_intake(fig, rows, fsm, "dissimilarity")

    cells, schools, _ = drawn_table(fig)
    means = rows.pivot_table(
        index="school", columns="scenario", values="dissimilarity_term"
    )
    np.testing.assert_allclose(
        cells[:, :2], means.loc[schools, list(COLOURS)].to_numpy()
    )
    np.testing.assert_allclose(cells[:, 2], fsm[schools])
    norm = fig.axes[0].collections[0].norm
    assert (norm.vmin, norm.vmax) == pytest.approx((-0.9, 0.9))


# --------------------------------------------------------------------------
# draw_route_utilisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_routes", "labelled"), [(2, True), (MAX_LABELLED_ROUTES + 1, False)]
)
def test_draw_route_utilisation_stacks_the_riders_below_the_unused_seats(
    n_routes, labelled
):
    fig = Figure()
    draw_route_utilisation(fig, route_rows(n_routes))
    ax = fig.axes[0]

    per_route = route_utilisation(route_rows(n_routes))
    bars = ax.containers[:3]
    heights = np.array([[bar.get_height() for bar in stack] for stack in bars])
    # Disadvantaged riders, then the others, then the unused seats, so every
    # bar stands at its route's seats.
    np.testing.assert_allclose(heights[0], per_route["disadvantaged"])
    np.testing.assert_allclose(heights[1], per_route["other"])
    np.testing.assert_allclose(heights.sum(axis=0), per_route["capacity"])
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    if labelled:
        assert labels == ["E00000001 → School 1", "E00000000 → School 0"]
    else:
        assert labels == []
        assert ax.get_xlabel() == "Routes, by seats provided"


# --------------------------------------------------------------------------
# draw_local_acceptance
# --------------------------------------------------------------------------


def test_draw_local_acceptance_stacks_the_accepted_below_the_rejected():
    fig = Figure()
    draw_local_acceptance(fig, local_rows())
    ax, change = fig.axes

    per_school = local_acceptance(local_rows())
    bars = ax.containers[:3]
    heights = np.array([[bar.get_height() for bar in stack] for stack in bars])
    # Disadvantaged accepted, then the others, then those rejected, so every
    # bar stands at its school's local applicants.
    np.testing.assert_allclose(heights[0], per_school["local_disadvantaged"])
    np.testing.assert_allclose(heights[1], per_school["local_other"])
    np.testing.assert_allclose(heights.sum(axis=0), per_school["local_applicants"])
    # Each school's bars with routes then without, in the scenario's colours.
    assert [to_hex(bar.get_facecolor()) for bar in bars[0][:2]] == [
        GROUP_COLOURS[f"disadvantaged, {scenario}"] for scenario in COLOURS
    ]
    # Below, a point per seed and school at its paired difference, under the
    # same schools.
    assert [tick.get_text() for tick in change.get_xticklabels()] == [
        "Beta",
        "Alpha",
        "Gamma",
    ]
    points = np.concatenate(
        [np.asarray(c.get_offsets())[:, 1] for c in change.collections]
    )
    np.testing.assert_array_equal(
        np.sort(points), np.sort(local_difference(local_rows())["difference"])
    )


# --------------------------------------------------------------------------
# draw_lorenz and plot_lorenz
# --------------------------------------------------------------------------


def test_draw_lorenz_draws_every_seed_and_its_gini_per_scenario():
    _, intake, _, _ = synthetic_frames()
    fig = Figure(figsize=(12, 6.5), layout="constrained")

    default = default_intake(intake)
    draw_lorenz(fig, default)
    for ax, scenario in zip(fig.axes, COLOURS):
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
        ("decile", f"IDACI decile ≤ {DEFAULTS.decile}"),
    ],
)
def test_draw_lorenz_names_the_group_the_rows_were_scored_on(disadvantage, group):
    fig = Figure()

    draw_lorenz(fig, default_intake(synthetic_frames()[1]), disadvantage)

    assert fig.axes[0].get_ylabel() == (
        f"Cumulative share of disadvantaged students ({group})"
    )


def test_plot_lorenz_writes_a_png(tmp_path):
    png = tmp_path / "out" / "lorenz.png"

    plot_lorenz(default_intake(synthetic_frames()[1]), png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
