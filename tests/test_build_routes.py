import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geo_model import build_routes as br

CRS = "EPSG:27700"

# Three districts on a line, 10km apart, and two schools sitting on the first
# and the middle one. Only the middle district is out of range of nothing.
SCHOOL_XY = np.array([[0.0, 0.0], [10_000.0, 0.0]])

# Both schools score below MAX_LOCAL_P8, so the local performance condition
# passes every district unless a test raises a score itself.
SCHOOL_P8 = np.array([-0.5, -0.5])


def make_areas(deciles, spacing=10_000.0, index=None):
    """Districts on a line, one every `spacing` metres, in the given deciles."""
    n = len(deciles)
    frame = gpd.GeoDataFrame(
        {
            "LSOA21CD": [f"E{i:08d}" for i in range(n)],
            "IMD Decile": deciles,
        },
        geometry=gpd.points_from_xy(np.arange(n) * spacing, np.zeros(n), crs=CRS),
        index=index,
    )
    return frame.rename_geometry("Centroids")


# --------------------------------------------------------------------------
# centroid_xy
# --------------------------------------------------------------------------


def test_centroid_xy_reads_the_population_centroid_of_every_district():
    coordinates = br.centroid_xy(make_areas([1, 1, 1]))

    np.testing.assert_array_equal(
        coordinates, [[0.0, 0.0], [10_000.0, 0.0], [20_000.0, 0.0]]
    )


# --------------------------------------------------------------------------
# build_routes
# --------------------------------------------------------------------------


def test_build_routes_joins_every_district_to_the_schools_beyond_the_threshold():
    # Distances are district 0: [0, 10000] and district 1: [10000, 0].
    districts = make_areas([1, 1])
    routes = br.build_routes(districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, 30)

    pd.testing.assert_frame_equal(
        routes,
        pd.DataFrame(
            {
                "route_id": np.array([0, 1], dtype=np.int32),
                "district_idx": np.array([0, 1], dtype=np.int32),
                "LSOA21CD": ["E00000000", "E00000001"],
                "school_idx": np.array([1, 0], dtype=np.int32),
                "capacity": np.array([30, 30], dtype=np.int32),
            }
        ),
    )


def test_build_routes_keeps_the_district_frame_index_as_district_idx():
    # The frame handed over is a disadvantaged subset, so its index labels are
    # positions in the full area frame and must survive unrenumbered.
    districts = make_areas([1, 1], index=[4, 7])
    routes = br.build_routes(districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, 30)

    np.testing.assert_array_equal(routes["district_idx"], [4, 7])
    np.testing.assert_array_equal(routes["route_id"], [0, 1])  # contiguous from 0


def test_build_routes_returns_nothing_when_every_school_is_within_reach():
    districts = make_areas([1, 1])
    routes = br.build_routes(districts, SCHOOL_XY, 50_000, 30)
    assert routes.empty


def test_build_routes_measures_from_the_population_centroid():
    # A district 4km out is routed to the school at the origin, one at 3km is
    # not, so the cut is made on the centroid distance and nothing else.
    districts = make_areas([1, 1], spacing=1.0)
    districts["Centroids"] = gpd.points_from_xy([3_000.0, 4_000.0], [0.0, 0.0])
    routes = br.build_routes(districts, np.array([[0.0, 0.0]]), 3218, 30)

    np.testing.assert_array_equal(routes["district_idx"], [1])


# --------------------------------------------------------------------------
# route_network
# --------------------------------------------------------------------------


def test_route_network_routes_only_the_disadvantaged_districts(capsys):
    areas = make_areas([5, 2, 1])
    routes = br.route_network(areas, SCHOOL_XY, SCHOOL_P8, decile=3, capacity=30)

    # District 0 is not disadvantaged. District 1 sits on the second school and
    # is routed to the first; district 2 is beyond both.
    np.testing.assert_array_equal(routes["district_idx"], [1, 2, 2])
    np.testing.assert_array_equal(routes["school_idx"], [0, 0, 1])
    assert (routes["capacity"] == 30).all()
    assert "2 disadvantaged districts at IMD decile 3" in capsys.readouterr().out


def test_route_network_names_a_district_left_without_any_route(capsys):
    # District 0 sits on top of the only school, so nothing is far enough away.
    areas = make_areas([1, 1], spacing=1.0)
    areas["Centroids"] = gpd.points_from_xy([0.0, 100_000.0], [0.0, 0.0])
    routes = br.route_network(areas, np.array([[0.0, 0.0]]), SCHOOL_P8[:1], decile=3)

    assert 0 not in set(routes["district_idx"])
    assert "E00000000" in capsys.readouterr().out


def test_route_network_uses_the_module_defaults():
    areas = make_areas([1, 1, 1])
    routes = br.route_network(areas, SCHOOL_XY, SCHOOL_P8)

    assert (routes["capacity"] == br.ROUTE_CAPACITY).all()
    # 3218m is the threshold, so the school a district sits on is never routed.
    assert not ((routes["district_idx"] == 1) & (routes["school_idx"] == 1)).any()
    # Neither school scores above the default P8, so every district is eligible.
    assert set(routes["district_idx"]) == {0, 1, 2}


@pytest.mark.parametrize("decile", [0, 11, -1])
def test_route_network_rejects_a_decile_outside_one_to_ten(decile):
    with pytest.raises(ValueError, match="must be an IMD decile"):
        br.route_network(make_areas([1, 2]), SCHOOL_XY, SCHOOL_P8, decile=decile)


def test_route_network_rejects_an_area_frame_that_is_not_positionally_indexed():
    # A student carries the position of the area it was sampled in, so a
    # relabelled frame would silently misalign district_idx.
    areas = make_areas([1, 1], index=[3, 9])
    with pytest.raises(ValueError, match="not positionally indexed"):
        br.route_network(areas, SCHOOL_XY, SCHOOL_P8)


def test_route_network_rejects_an_instance_with_no_disadvantaged_district():
    with pytest.raises(ValueError, match="No LSOA sits at or below"):
        br.route_network(make_areas([8, 9, 10]), SCHOOL_XY, SCHOOL_P8, decile=3)


def test_route_network_rejects_an_empty_route_set():
    with pytest.raises(ValueError, match="so the route set is"):
        br.route_network(make_areas([1, 1]), SCHOOL_XY, SCHOOL_P8, min_distance=50_000)


def test_route_network_drops_a_district_a_high_scoring_school_already_serves():
    # District 0 sits on the first school, which scores above the threshold, so
    # it is served already. The others are 10km and 20km off and stay eligible.
    areas = make_areas([1, 1, 1])
    routes = br.route_network(areas, SCHOOL_XY, np.array([1.0, -0.5]))

    assert set(routes["district_idx"]) == {1, 2}


def test_route_network_keeps_a_district_whose_local_school_is_not_above_the_threshold():
    # A school scoring exactly the threshold is not above it, so it serves
    # nobody and district 0 keeps its routes.
    areas = make_areas([1, 1, 1])
    routes = br.route_network(areas, SCHOOL_XY, np.array([0.0, -0.5]), max_local_p8=0.0)

    assert 0 in set(routes["district_idx"])


def test_route_network_takes_a_school_on_the_radius_as_local():
    # A district exactly on the radius is served, one a metre beyond it is not,
    # so the cut is made on the centroid distance and nothing else.
    areas = make_areas([1, 1], spacing=1.0)
    areas["Centroids"] = gpd.points_from_xy([3_218.0, 3_219.0], [0.0, 0.0])
    routes = br.route_network(
        areas, np.array([[0.0, 0.0]]), np.array([1.0]), local_radius=3218
    )

    assert set(routes["district_idx"]) == {1}


def test_route_network_rejects_an_instance_with_no_route_eligible_district():
    with pytest.raises(ValueError, match="no district is route-eligible"):
        br.route_network(
            make_areas([1, 1]), SCHOOL_XY, np.array([1.0, 1.0]), local_radius=50_000
        )


def test_route_network_rejects_school_scores_that_do_not_align_with_the_coordinates():
    # Misaligned scores would gate on the wrong schools, plausibly and silently.
    with pytest.raises(ValueError, match="school_scores must align"):
        br.route_network(make_areas([1, 1]), SCHOOL_XY, np.array([0.0]))


def test_route_network_does_not_name_a_district_excluded_on_performance_as_unrouted(
    capsys,
):
    # District 0 sits on the only school and is served by its score, so it is
    # excluded on performance. Reporting it as too close to every school would
    # confuse the two conditions.
    areas = make_areas([1, 1], spacing=1.0)
    areas["Centroids"] = gpd.points_from_xy([0.0, 100_000.0], [0.0, 0.0])
    routes = br.route_network(areas, np.array([[0.0, 0.0]]), np.array([1.0]))

    assert set(routes["district_idx"]) == {1}
    out = capsys.readouterr().out
    assert "Every secondary school is within" not in out
    assert "2 disadvantaged districts at IMD decile 3 or below, 1 of them" in out


# --------------------------------------------------------------------------
# save_routes
# --------------------------------------------------------------------------


def test_save_routes_writes_the_set_whole_and_by_axis(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "ROUTES_CSV", tmp_path / "out" / "routes.csv")
    monkeypatch.setattr(br, "ROUTES_NPZ", tmp_path / "out" / "routes.npz")
    routes = br.build_routes(make_areas([1, 1]), SCHOOL_XY, br.MIN_ROUTE_DISTANCE, 30)

    br.save_routes(routes)

    pd.testing.assert_frame_equal(pd.read_csv(br.ROUTES_CSV), routes, check_dtype=False)
    with np.load(br.ROUTES_NPZ) as axes:
        np.testing.assert_array_equal(axes["route_capacities"], routes["capacity"])
        np.testing.assert_array_equal(axes["route_school_idx"], routes["school_idx"])
        np.testing.assert_array_equal(
            axes["route_district_idx"], routes["district_idx"]
        )
        assert all(axes[name].dtype == np.int32 for name in axes.files)
