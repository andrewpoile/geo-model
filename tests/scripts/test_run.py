import sys
from dataclasses import replace

import geopandas as gpd
import pandas as pd
import pytest

from geo_model.analysis.scoring import settings_matchings
from geo_model.measures.segregation import dissimilarity_index
from geo_model.measures.travel import MODES
from geo_model.model.settings import DEFAULTS, DISADVANTAGE_CHOICES
from geo_model.model.students import disadvantaged_group
from geo_model.output import ROUTES_CSV, ROUTES_NPZ
from geo_model.scripts import run as rn

# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


def test_run_scores_writes_and_plots_the_setting_against_no_routes(
    samples, areas, secondary, shares, tmp_path
):
    settings = replace(DEFAULTS, capacity_scale=2.0, progressivity=0.5)

    results, intake, modes, routes = rn.run(
        settings, samples, areas, secondary, shares, tmp_path
    )

    rows, school_rows, mode_rows, route_rows = rn.score_settings(
        settings, samples, areas, secondary, shares, utilisation=True
    )
    pd.testing.assert_frame_equal(results, pd.DataFrame(rows))
    pd.testing.assert_frame_equal(intake, pd.DataFrame(school_rows))
    pd.testing.assert_frame_equal(modes, pd.DataFrame(mode_rows))
    pd.testing.assert_frame_equal(routes, pd.DataFrame(route_rows))
    assert len(routes) == results["n_routes"].iloc[0] * len(samples)
    assert list(results["scenario"]) == ["with routes", "without routes"]
    assert results["dissimilarity"].between(0, 1).all()
    # Both scenarios match the same students, and seat them by every mode.
    assert results["n_matched"].add(results["n_unmatched"]).nunique() == 1
    assert set(modes["mode"]) == set(MODES)
    assert modes.groupby("scenario")["students"].sum().round(6).nunique() == 1
    for frame, name in (
        (results, rn.RESULTS_CSV),
        (intake, rn.SCHOOLS_CSV),
        (modes, rn.MODES_CSV),
        (routes, rn.ROUTE_UTILISATION_CSV),
    ):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), frame)
    assert set(pd.read_csv(tmp_path / rn.DISTRICTS_CSV)["seed"]) == {0}
    assert (tmp_path / ROUTES_CSV).exists() and (tmp_path / ROUTES_NPZ).exists()
    for name in (
        rn.DISSIMILARITY_PNG,
        rn.UNASSIGNED_PNG,
        rn.INTAKE_PNG,
        rn.MODES_PNG,
        rn.LORENZ_PNG,
        rn.MAP_PNG,
        rn.ROUTES_PNG,
        rn.LOCALS_PNG,
    ):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name


def test_run_rejects_no_samples(tmp_path):
    with pytest.raises(ValueError, match="No student sample"):
        rn.run(
            DEFAULTS,
            [],
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            pd.DataFrame(),
            tmp_path,
        )


def test_main_rejects_two_route_rules_before_loading_a_region(monkeypatch):
    def load(*args):
        pytest.fail("loaded a region for settings it cannot run")

    monkeypatch.setattr(rn, "regions", load)
    monkeypatch.setattr(sys, "argv", ["run", "--shortest-only", "--bottleneck"])

    with pytest.raises(SystemExit):
        rn.main()


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
    rows, _, _, route_rows = rn.score_settings(
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
