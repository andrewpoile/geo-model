import pandas as pd
import pytest
from matplotlib.figure import Figure

from geo_model import dissimilarity as ds
from geo_model.analysis.scoring import settings_matchings
from geo_model.measures.segregation import dissimilarity_index
from geo_model.model.settings import DEFAULTS, DISADVANTAGE_CHOICES
from geo_model.model.students import disadvantaged_group

# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def test_plot_writes_a_png_of_the_results(tmp_path):
    png = tmp_path / "out" / "dissimilarity.png"
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "dissimilarity": [0.30, 0.35, 0.28, 0.36],
        }
    )

    ds.plot(results, png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_fits_the_y_axis_to_the_indices(tmp_path, monkeypatch):
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

    ds.plot(results, tmp_path / "dissimilarity.png")

    low, high = saved[0].axes[0].get_ylim()
    # Every index is in view, on an axis far narrower than the index's 0 to 1.
    assert 0 < low <= 0.28 and 0.36 <= high
    assert high - low < 0.2


def test_plot_draws_one_seed_with_its_test_undefined(tmp_path):
    png = tmp_path / "dissimilarity.png"
    results = pd.DataFrame(
        {
            "seed": [0, 0],
            "scenario": ["with routes", "without routes"],
            "dissimilarity": [0.30, 0.35],
        }
    )

    ds.plot(results, png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


def test_run_scores_both_scenarios_of_every_seed(areas, secondary, tmp_path):
    results = ds.run(1, areas, secondary, tmp_path)

    assert list(results["scenario"]) == ["with routes", "without routes"]
    assert results["dissimilarity"].between(0, 1).all()
    # Both scenarios match the same student sample.
    assert results["n_matched"].add(results["n_unmatched"]).nunique() == 1
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / ds.RESULTS_CSV), results)


# --------------------------------------------------------------------------
# settings_matchings: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("disadvantage", DISADVANTAGE_CHOICES)
def test_settings_matchings_are_the_ones_the_sweep_scores_at_the_defaults(
    samples, areas, secondary, shares, disadvantage
):
    matchings = settings_matchings(
        DEFAULTS, samples[0], areas, secondary, disadvantage=disadvantage
    )
    rows, _, _, route_rows = ds.score_settings(
        DEFAULTS,
        samples,
        areas,
        secondary,
        shares,
        disadvantage=disadvantage,
    )
    # The route rows are kept only when asked for.
    assert route_rows == []

    _, student_lsoa, drawn, _ = samples[0]
    disadvantaged = disadvantaged_group(
        student_lsoa, drawn, areas, DEFAULTS.decile, disadvantage
    )
    assert [row["scenario"] for row in rows] == list(matchings)
    for row in rows:
        matched_school = matchings[row["scenario"]][:, 0]
        assert (
            dissimilarity_index(matched_school, disadvantaged, len(secondary))
            == row["dissimilarity"]
        )
        assert (matched_school >= 0).sum() == row["n_matched"]
    assert (matchings["with routes"][:, 1] >= 0).any()
    assert (matchings["without routes"][:, 1] < 0).all()
