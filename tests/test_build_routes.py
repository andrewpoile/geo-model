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

# Seats a route would hold between every pair, for `build_routes` directly.
SEATS = np.full((2, 2), 30, dtype=np.int32)


def make_areas(deciles, spacing=10_000.0, index=None):
    """Districts on a line, one every `spacing` metres, in the given deciles."""
    n = len(deciles)
    frame = gpd.GeoDataFrame(
        {
            "LSOA21CD": [f"E{i:08d}" for i in range(n)],
            "IDACI Decile": deciles,
        },
        geometry=gpd.points_from_xy(np.arange(n) * spacing, np.zeros(n), crs=CRS),
        index=index,
    )
    return frame.rename_geometry("Centroids")


def network(areas, school_xy=SCHOOL_XY, school_scores=SCHOOL_P8, **kwargs):
    """`route_network` with every school admitting 100 and every district
    holding 10 students, unless a test passes its own."""
    school_pan = kwargs.pop("school_pan", np.full(len(school_xy), 100))
    district_cohort = kwargs.pop("district_cohort", np.full(len(areas), 10))
    return br.route_network(
        areas, school_xy, school_scores, school_pan, district_cohort, **kwargs
    )


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
    routes = br.build_routes(districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, SEATS)

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
    routes = br.build_routes(districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, SEATS)

    np.testing.assert_array_equal(routes["district_idx"], [4, 7])
    np.testing.assert_array_equal(routes["route_id"], [0, 1])  # contiguous from 0


def test_build_routes_returns_nothing_when_every_school_is_within_reach():
    districts = make_areas([1, 1])
    routes = br.build_routes(districts, SCHOOL_XY, 50_000, SEATS)
    assert routes.empty


def test_build_routes_measures_from_the_population_centroid():
    # A district 4km out is routed to the school at the origin, one at 3km is
    # not, so the cut is made on the centroid distance and nothing else.
    districts = make_areas([1, 1], spacing=1.0)
    districts["Centroids"] = gpd.points_from_xy([3_000.0, 4_000.0], [0.0, 0.0])
    routes = br.build_routes(districts, np.array([[0.0, 0.0]]), 3218, SEATS[:, :1])

    np.testing.assert_array_equal(routes["district_idx"], [1])


def test_build_routes_leaves_a_pair_holding_no_seat_unrouted(capsys):
    # Both pairs lie beyond the threshold, but the first holds no seat, so only
    # the second is routed and route_id still runs contiguously from 0.
    districts = make_areas([1, 1])
    seats = np.array([[30, 0], [30, 30]], dtype=np.int32)
    routes = br.build_routes(districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, seats)

    np.testing.assert_array_equal(routes["district_idx"], [1])
    np.testing.assert_array_equal(routes["school_idx"], [0])
    np.testing.assert_array_equal(routes["route_id"], [0])
    assert (
        "1 of the 2 routes beyond 3218m would hold no seat" in capsys.readouterr().out
    )


# --------------------------------------------------------------------------
# route_network
# --------------------------------------------------------------------------


def test_route_network_routes_only_the_disadvantaged_districts(capsys):
    areas = make_areas([5, 2, 1])
    routes = network(areas, decile=3)

    # District 0 is not disadvantaged. District 1 sits on the second school and
    # is routed to the first; district 2 is beyond both.
    np.testing.assert_array_equal(routes["district_idx"], [1, 2, 2])
    np.testing.assert_array_equal(routes["school_idx"], [0, 0, 1])
    assert "2 disadvantaged districts at IDACI decile 3" in capsys.readouterr().out


def test_route_network_names_a_district_left_without_any_route(capsys):
    # District 0 sits on top of the only school, so nothing is far enough away.
    areas = make_areas([1, 1], spacing=1.0)
    areas["Centroids"] = gpd.points_from_xy([0.0, 100_000.0], [0.0, 0.0])
    routes = network(areas, np.array([[0.0, 0.0]]), SCHOOL_P8[:1], decile=3)

    assert 0 not in set(routes["district_idx"])
    assert "E00000000" in capsys.readouterr().out


def test_route_network_uses_the_module_defaults():
    areas = make_areas([1, 1, 1])
    routes = network(areas)

    # At the default scale a route holds that multiple of its fair share,
    # rounded to the nearest seat.
    assert (routes["capacity"] == 167).all()  # rint(5 * 100 * 10 / 30)
    # 3218m is the threshold, so the school a district sits on is never routed.
    assert not ((routes["district_idx"] == 1) & (routes["school_idx"] == 1)).any()
    # Neither school scores above the default P8, so every district is eligible.
    assert set(routes["district_idx"]) == {0, 1, 2}


@pytest.mark.parametrize("decile", [0, 11, -1])
def test_route_network_rejects_a_decile_outside_one_to_ten(decile):
    with pytest.raises(ValueError, match="must be an IDACI decile"):
        network(make_areas([1, 2]), decile=decile)


def test_route_network_rejects_an_area_frame_that_is_not_positionally_indexed():
    # A student carries the position of the area it was sampled in, so a
    # relabelled frame would silently misalign district_idx.
    areas = make_areas([1, 1], index=[3, 9])
    with pytest.raises(ValueError, match="not positionally indexed"):
        network(areas)


def test_route_network_rejects_an_instance_with_no_disadvantaged_district():
    with pytest.raises(ValueError, match="No LSOA sits at or below"):
        network(make_areas([8, 9, 10]), decile=3)


def test_route_network_rejects_an_empty_route_set():
    with pytest.raises(ValueError, match="so the route set is"):
        network(make_areas([1, 1]), min_distance=50_000)


def test_route_network_drops_a_district_a_high_scoring_school_already_serves():
    # District 0 sits on the first school, which scores above the threshold, so
    # it is served already. The others are 10km and 20km off and stay eligible.
    areas = make_areas([1, 1, 1])
    routes = network(areas, school_scores=np.array([1.0, -0.5]))

    assert set(routes["district_idx"]) == {1, 2}


def test_route_network_keeps_a_district_whose_local_school_is_not_above_the_threshold():
    # A school scoring exactly the threshold is not above it, so it serves
    # nobody and district 0 keeps its routes.
    areas = make_areas([1, 1, 1])
    routes = network(areas, school_scores=np.array([0.0, -0.5]), max_local_p8=0.0)

    assert 0 in set(routes["district_idx"])


def test_route_network_takes_a_school_on_the_radius_as_local():
    # A district exactly on the radius is served, one a metre beyond it is not,
    # so the cut is made on the centroid distance and nothing else.
    areas = make_areas([1, 1], spacing=1.0)
    areas["Centroids"] = gpd.points_from_xy([3_218.0, 3_219.0], [0.0, 0.0])
    routes = network(areas, np.array([[0.0, 0.0]]), np.array([1.0]), local_radius=3218)

    assert set(routes["district_idx"]) == {1}


def test_route_network_rejects_an_instance_with_no_route_eligible_district():
    with pytest.raises(ValueError, match="no district is route-eligible"):
        network(
            make_areas([1, 1]), school_scores=np.array([1.0, 1.0]), local_radius=50_000
        )


def test_route_network_rejects_school_scores_that_do_not_align_with_the_coordinates():
    # Misaligned scores would gate on the wrong schools, plausibly and silently.
    with pytest.raises(ValueError, match="school_scores must align"):
        network(make_areas([1, 1]), school_scores=np.array([0.0]))


def test_route_network_rejects_admission_numbers_that_do_not_align_with_the_schools():
    with pytest.raises(ValueError, match="school_pan must align"):
        network(make_areas([1, 1]), school_pan=np.array([100]))


def test_route_network_rejects_cohorts_that_do_not_align_with_the_areas():
    with pytest.raises(ValueError, match="district_cohort must align"):
        network(make_areas([1, 1]), district_cohort=np.array([10]))


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_route_network_rejects_a_capacity_scale_that_is_not_positive(scale):
    with pytest.raises(ValueError, match="capacity_scale must be positive"):
        network(make_areas([1, 1]), capacity_scale=scale)


def test_route_network_seats_every_route_on_its_fair_share_of_the_pan():
    # 100 students in the city, district 2 is not disadvantaged, and the
    # schools admit 100 and 200: a route holds k * PAN * n_d / 100 seats.
    areas = make_areas([1, 1, 5])
    routes = network(
        areas,
        school_pan=np.array([100, 200]),
        district_cohort=np.array([20, 30, 50]),
        capacity_scale=1.5,
    )

    # District 0 sits on school 0 and district 1 on school 1.
    np.testing.assert_array_equal(routes["district_idx"], [0, 1])
    np.testing.assert_array_equal(routes["school_idx"], [1, 0])
    np.testing.assert_array_equal(routes["capacity"], [60, 45])  # 1.5*200*20/100


def test_route_network_drops_a_route_rounding_to_no_seat_unless_rounding_up(capsys):
    # District 0 takes 0.4 of a seat at school 1 and district 1 takes 1.2 at
    # school 0, so rounding to the nearest seat leaves district 0 unrouted.
    areas = make_areas([1, 1, 5])
    kwargs = {
        "school_pan": np.array([12, 4]),
        "district_cohort": np.array([10, 10, 80]),
        "capacity_scale": 1.0,
    }

    nearest = network(areas, **kwargs)
    assert "No secondary school beyond 3218m holds a seat" in capsys.readouterr().out
    np.testing.assert_array_equal(nearest["district_idx"], [1])
    np.testing.assert_array_equal(nearest["capacity"], [1])

    up = network(areas, round_up=True, **kwargs)
    np.testing.assert_array_equal(up["district_idx"], [0, 1])
    np.testing.assert_array_equal(up["capacity"], [1, 2])


def test_route_network_rejects_a_route_set_holding_no_seat():
    with pytest.raises(ValueError, match="so the route set is"):
        network(make_areas([1, 1, 9]), district_cohort=np.array([1, 1, 1_000_000]))


@pytest.mark.parametrize(
    ("progressivity", "seats"), [(0.0, [10, 10]), (0.5, [12, 8]), (1.0, [15, 5])]
)
def test_route_network_leans_seats_towards_the_deprived_districts(progressivity, seats):
    # Districts at deciles 1 and 3 of threshold 3, 10 students each out of 100,
    # routed to one far school admitting 100. At p = 1 the raw weights are 1
    # and 1/3, rescaled to 1.5 and 0.5; at p = 0.5 they are 1 and 2/3, rescaled
    # to 1.2 and 0.8. The 20 seats the two share never change.
    routes = network(
        make_areas([1, 3, 5]),
        np.array([[100_000.0, 0.0]]),
        SCHOOL_P8[:1],
        school_pan=np.array([100]),
        district_cohort=np.array([10, 10, 80]),
        decile=3,
        capacity_scale=1.0,
        progressivity=progressivity,
    )

    np.testing.assert_array_equal(routes["district_idx"], [0, 1])
    np.testing.assert_array_equal(routes["capacity"], seats)


@pytest.mark.parametrize(("linear", "seats"), [(False, [13, 7]), (True, [12, 8])])
def test_route_network_weights_the_middle_deciles_by_the_profile_chosen(linear, seats):
    # Districts at deciles 1 and 2 of threshold 3, 10 students each out of 100,
    # routed to one far school admitting 100, fully progressive. By 1 / D the
    # raw weights are 1 and 1/2, rescaled to 4/3 and 2/3; by the linear profile
    # they are 1 and 2/3, rescaled to 1.2 and 0.8.
    routes = network(
        make_areas([1, 2, 5]),
        np.array([[100_000.0, 0.0]]),
        SCHOOL_P8[:1],
        school_pan=np.array([100]),
        district_cohort=np.array([10, 10, 80]),
        decile=3,
        capacity_scale=1.0,
        progressivity=1.0,
        linear=linear,
    )

    np.testing.assert_array_equal(routes["district_idx"], [0, 1])
    np.testing.assert_array_equal(routes["capacity"], seats)


@pytest.mark.parametrize("progressivity", [-0.1, 1.1])
def test_route_network_rejects_a_progressivity_outside_zero_to_one(progressivity):
    with pytest.raises(ValueError, match="progressivity must lie in"):
        network(make_areas([1, 1]), progressivity=progressivity)


def test_route_network_rejects_route_eligible_districts_holding_no_students():
    with pytest.raises(ValueError, match="hold no students"):
        network(make_areas([1, 1, 9]), district_cohort=np.array([0, 0, 100]))


def test_route_network_does_not_name_a_district_excluded_on_performance_as_unrouted(
    capsys,
):
    # District 0 sits on the only school and is served by its score, so it is
    # excluded on performance. Reporting it as too close to every school would
    # confuse the two conditions.
    areas = make_areas([1, 1], spacing=1.0)
    areas["Centroids"] = gpd.points_from_xy([0.0, 100_000.0], [0.0, 0.0])
    routes = network(areas, np.array([[0.0, 0.0]]), np.array([1.0]))

    assert set(routes["district_idx"]) == {1}
    out = capsys.readouterr().out
    assert "No secondary school beyond" not in out
    assert "2 disadvantaged districts at IDACI decile 3 or below, 1 of them" in out


# --------------------------------------------------------------------------
# save_routes
# --------------------------------------------------------------------------


def test_save_routes_writes_the_set_whole_and_by_axis(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "ROUTES_CSV", tmp_path / "out" / "routes.csv")
    monkeypatch.setattr(br, "ROUTES_NPZ", tmp_path / "out" / "routes.npz")
    routes = br.build_routes(
        make_areas([1, 1]), SCHOOL_XY, br.MIN_ROUTE_DISTANCE, SEATS
    )

    br.save_routes(routes)

    pd.testing.assert_frame_equal(pd.read_csv(br.ROUTES_CSV), routes, check_dtype=False)
    with np.load(br.ROUTES_NPZ) as axes:
        np.testing.assert_array_equal(axes["route_capacities"], routes["capacity"])
        np.testing.assert_array_equal(axes["route_school_idx"], routes["school_idx"])
        np.testing.assert_array_equal(
            axes["route_district_idx"], routes["district_idx"]
        )
        assert all(axes[name].dtype == np.int32 for name in axes.files)
