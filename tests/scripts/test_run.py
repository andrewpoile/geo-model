import math
import sys
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geo_model.analysis.scoring import settings_matchings
from geo_model.data import sources
from geo_model.measures.segregation import dissimilarity_index
from geo_model.measures.travel import MODES
from geo_model.model.routes import centroid_xy
from geo_model.model.settings import (
    DEFAULTS,
    DISADVANTAGE_CHOICES,
    PRIORITY_CHOICES,
)
from geo_model.model.students import disadvantaged_group
from geo_model.output import ROUTES_CSV, ROUTES_NPZ, region_dir
from geo_model.scripts import run as rn

# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


def test_run_scores_writes_and_plots_the_setting_against_no_routes(
    samples, areas, secondary, district_distance, district_minutes, shares, tmp_path
):
    settings = replace(DEFAULTS, capacity_scale=2.0, progressivity=0.5)

    results, intake, modes, routes = rn.run(
        settings,
        samples,
        areas,
        secondary,
        district_distance,
        district_minutes,
        shares,
        tmp_path,
        distance="euclidean",
    )

    rows, school_rows, mode_rows, route_rows = rn.score_settings(
        settings,
        samples,
        areas,
        secondary,
        district_distance,
        district_minutes,
        shares,
        utilisation=True,
        distance="euclidean",
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


def test_load_region_measures_along_the_walking_network_by_default(
    areas, to_school, route_data
):
    _, _, walk, district_distance, district_minutes, folder = rn.load_region(
        sources.LAS, areas, False, "distance", "network", math.inf
    )
    straight = to_school(centroid_xy(areas))
    assert walk.rule == "network"
    assert district_distance.shape == straight.shape
    assert (district_distance >= straight).all()
    assert district_minutes is not None
    assert district_minutes.shape == straight.shape
    assert np.isfinite(district_minutes).all()
    assert folder == region_dir(sources.LAS)


def test_load_region_reads_no_network_in_a_straight_line_run_with_no_cap(
    areas, monkeypatch
):
    def unread(*args):
        raise AssertionError("read the network for a straight-line run")

    monkeypatch.setattr(rn, "region_links", unread)
    _, _, straight, _, district_minutes, _ = rn.load_region(
        sources.LAS, areas, False, "distance", "euclidean", math.inf
    )
    assert straight.rule == "euclidean"
    assert district_minutes is None


def test_run_rejects_no_samples(tmp_path):
    with pytest.raises(ValueError, match="No student sample"):
        rn.run(
            DEFAULTS,
            [],
            gpd.GeoDataFrame(),
            gpd.GeoDataFrame(),
            np.empty((0, 0)),
            np.empty((0, 0)),
            pd.DataFrame(),
            tmp_path,
            distance="euclidean",
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["--shortest-only", "--bottleneck"],
        ["--priority-discount", "1.5"],
        ["--priority", "bracket", "--priority-discount", "0.8"],
    ],
)
def test_main_rejects_settings_it_cannot_run_before_loading_a_region(monkeypatch, argv):
    def load(*args):
        pytest.fail("loaded a region for settings it cannot run")

    monkeypatch.setattr(rn, "regions", load)
    monkeypatch.setattr(sys, "argv", ["run", *argv])

    with pytest.raises(SystemExit):
        rn.main()


# --------------------------------------------------------------------------
# settings_matchings: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("disadvantage", DISADVANTAGE_CHOICES)
@pytest.mark.parametrize("priority", PRIORITY_CHOICES)
def test_settings_matchings_are_the_ones_the_sweep_scores(
    samples,
    areas,
    secondary,
    district_distance,
    district_minutes,
    shares,
    disadvantage,
    priority,
):
    # Off the default discount, so a discount settings_matchings or
    # score_settings leaves behind shows; below them, score_sample's own
    # test covers the way down.
    settings = replace(DEFAULTS, priority_discount=0.9)
    matchings = settings_matchings(
        settings,
        samples[0],
        areas,
        secondary,
        district_distance,
        district_minutes,
        disadvantage=disadvantage,
        priority=priority,
    )
    rows, _, _, route_rows = rn.score_settings(
        settings,
        samples,
        areas,
        secondary,
        district_distance,
        district_minutes,
        shares,
        disadvantage=disadvantage,
        priority=priority,
        distance="euclidean",
    )
    # The route rows are kept only when asked for.
    assert route_rows == []

    _, student_lsoa, drawn, _, _ = samples[0]
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
