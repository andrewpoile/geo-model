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

# District weights and the seats on the routes into each school, for
# `build_routes` directly.
WEIGHT = np.ones(2)
SCHOOL_SEATS = np.full(2, 30)


def make_areas(deciles, spacing=10_000.0, index=None, scores=None, ranks=None):
    """Districts on a line, one every `spacing` metres, in the given deciles.

    Every district scores 1 on IDACI unless a test passes its own scores, so
    under every choice of disadvantage every student of a district may ride.
    The districts are ranked on IDACI in order unless a test passes its own
    ranks.
    """
    n = len(deciles)
    frame = gpd.GeoDataFrame(
        {
            "LSOA21CD": [f"E{i:08d}" for i in range(n)],
            "IDACI": np.arange(1, n + 1) if ranks is None else ranks,
            "IDACI Decile": deciles,
            "IDACI Score": np.ones(n) if scores is None else scores,
        },
        geometry=gpd.points_from_xy(np.arange(n) * spacing, np.zeros(n), crs=CRS),
        index=index,
    )
    return frame.rename_geometry("Centroids")


def network(areas, school_xy=SCHOOL_XY, school_scores=SCHOOL_P8, **kwargs):
    """`route_network` with every school offering 100 places and every
    district holding 10 students, unless a test passes its own."""
    school_places = kwargs.pop("school_places", np.full(len(school_xy), 100))
    district_cohort = kwargs.pop("district_cohort", np.full(len(areas), 10))
    return br.route_network(
        areas, school_xy, school_scores, school_places, district_cohort, **kwargs
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
# disadvantaged_cohort
# --------------------------------------------------------------------------


def test_disadvantaged_cohort_rounds_each_share_to_the_nearest_student():
    # 0.134 * 19 = 2.5, 0.767 * 21 = 16.1 and 0 * 30 = 0.
    areas = make_areas([1, 1, 1], scores=[0.134, 0.767, 0.0])

    cohort = br.disadvantaged_cohort(areas, np.array([19, 21, 30]))

    np.testing.assert_array_equal(cohort, [3, 16, 0])


# --------------------------------------------------------------------------
# apportion
# --------------------------------------------------------------------------


def test_apportion_hands_the_seats_left_over_to_the_largest_remainders():
    # 10 seats on weights 1, 2 and 3 are quotas 1.67, 3.33 and 5: the floors
    # take 9 and the largest remainder, 0.67, the tenth. On equal weights the
    # quotas tie at 3.33 and the first row takes the tenth.
    quotas = 10 * np.array([[1, 1], [2, 1], [3, 1]]) / np.array([6, 3])

    seats = br.apportion(quotas, np.array([10, 10]))

    np.testing.assert_array_equal(seats, [[2, 4], [3, 3], [5, 3]])


# --------------------------------------------------------------------------
# bottleneck_pairs
# --------------------------------------------------------------------------


def test_bottleneck_pairs_makes_the_longest_pair_as_short_as_it_can_be():
    # The diagonal is the shorter in total, 11 against 12, but its longest
    # pair is 10 against the other pairing's 6.
    pairs = br.bottleneck_pairs(np.array([[1.0, 6.0], [6.0, 10.0]]))

    np.testing.assert_array_equal(pairs, [[False, True], [True, False]])


def test_bottleneck_pairs_takes_the_least_total_of_the_pairings_within_the_longest():
    # School 0 is allowed district 0 alone, 10 away, so every pairing's
    # longest pair is 10. Within it, districts 1 and 2 cross over to schools
    # 2 and 1 for 4 in total rather than 6 straight.
    inf = np.inf
    pairs = br.bottleneck_pairs(
        np.array([[10.0, inf, inf], [inf, 1.0, 2.0], [inf, 2.0, 5.0]])
    )

    np.testing.assert_array_equal(
        pairs, [[True, False, False], [False, False, True], [False, True, False]]
    )


def test_bottleneck_pairs_rejects_schools_that_cannot_each_have_a_district():
    # Both schools are allowed district 0 alone.
    inf = np.inf
    with pytest.raises(ValueError, match="No pairing gives each of the 2 schools"):
        br.bottleneck_pairs(np.array([[1.0, 2.0], [inf, inf]]))


# --------------------------------------------------------------------------
# build_routes
# --------------------------------------------------------------------------


def test_build_routes_joins_every_district_to_the_schools_beyond_the_threshold():
    # Distances are district 0: [0, 10000] and district 1: [10000, 0].
    districts = make_areas([1, 1])
    routes = br.build_routes(
        districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, WEIGHT, SCHOOL_SEATS
    )

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
    routes = br.build_routes(
        districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, WEIGHT, SCHOOL_SEATS
    )

    np.testing.assert_array_equal(routes["district_idx"], [4, 7])
    np.testing.assert_array_equal(routes["route_id"], [0, 1])  # contiguous from 0


def test_build_routes_returns_nothing_when_every_school_is_within_reach():
    districts = make_areas([1, 1])
    routes = br.build_routes(districts, SCHOOL_XY, 50_000, WEIGHT, SCHOOL_SEATS)
    assert routes.empty


def test_build_routes_measures_from_the_population_centroid():
    # A district 4km out is routed to the school at the origin, one at 3km is
    # not, so the cut is made on the centroid distance and nothing else.
    districts = make_areas([1, 1], spacing=1.0)
    districts["Centroids"] = gpd.points_from_xy([3_000.0, 4_000.0], [0.0, 0.0])
    routes = br.build_routes(
        districts, np.array([[0.0, 0.0]]), 3218, WEIGHT, SCHOOL_SEATS[:1]
    )

    np.testing.assert_array_equal(routes["district_idx"], [1])


def test_build_routes_splits_a_schools_seats_between_the_routes_feeding_it():
    # Districts at 0, 10km and 20km and one school at 0. District 0 is within
    # the threshold, so the school's 30 seats go to the other two alone, by
    # their weights of 1 and 2.
    districts = make_areas([1, 1, 1])
    routes = br.build_routes(
        districts,
        np.array([[0.0, 0.0]]),
        br.MIN_ROUTE_DISTANCE,
        np.array([5.0, 1.0, 2.0]),
        np.array([30]),
    )

    np.testing.assert_array_equal(routes["district_idx"], [1, 2])
    np.testing.assert_array_equal(routes["capacity"], [10, 20])


def test_build_routes_leaves_a_district_of_no_weight_unrouted():
    districts = make_areas([1, 1])
    routes = br.build_routes(
        districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, np.array([0.0, 1.0]), SCHOOL_SEATS
    )

    np.testing.assert_array_equal(routes["district_idx"], [1])
    np.testing.assert_array_equal(routes["school_idx"], [0])


def test_build_routes_leaves_a_route_apportioned_no_seat_unrouted(capsys):
    # Both pairs lie beyond the threshold, but the second school holds no seat,
    # so only the route into the first is built and route_id still runs
    # contiguously from 0.
    districts = make_areas([1, 1])
    routes = br.build_routes(
        districts, SCHOOL_XY, br.MIN_ROUTE_DISTANCE, WEIGHT, np.array([30, 0])
    )

    np.testing.assert_array_equal(routes["district_idx"], [1])
    np.testing.assert_array_equal(routes["school_idx"], [0])
    np.testing.assert_array_equal(routes["route_id"], [0])
    assert (
        "1 of the 2 routes beyond 3218m would hold no seat" in capsys.readouterr().out
    )


def test_build_routes_keeps_only_the_shortest_route_holding_a_seat_of_each_school():
    # Districts at 0, 10km and 20km, schools at 0, 10km and 50km. School 0 is
    # nearest district 1, which carries no weight, so it keeps district 2.
    # School 1 lies 10km from districts 0 and 2 and keeps the first. School 2
    # holds no seat, so it keeps no route. Each route holds its school's seats.
    districts = make_areas([1, 1, 1])
    school_xy = np.array([[0.0, 0.0], [10_000.0, 0.0], [50_000.0, 0.0]])

    routes = br.build_routes(
        districts,
        school_xy,
        0.0,
        np.array([1.0, 0.0, 1.0]),
        np.array([30, 40, 0]),
        shortest_only=True,
    )

    np.testing.assert_array_equal(routes["district_idx"], [0, 2])
    np.testing.assert_array_equal(routes["school_idx"], [1, 0])
    np.testing.assert_array_equal(routes["capacity"], [40, 30])
    np.testing.assert_array_equal(routes["route_id"], [0, 1])


# --------------------------------------------------------------------------
# route_network
# --------------------------------------------------------------------------


def test_route_network_routes_only_the_districts_at_or_below_the_decile(capsys):
    areas = make_areas([5, 2, 1])
    routes = network(areas, decile=3)

    # District 0 sits above the decile. District 1 sits on the second school
    # and is routed to the first; district 2 is beyond both.
    np.testing.assert_array_equal(routes["district_idx"], [1, 2, 2])
    np.testing.assert_array_equal(routes["school_idx"], [0, 0, 1])
    assert "2 districts at IDACI decile 3" in capsys.readouterr().out


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

    # Every student is disadvantaged, so at the default scale each school's
    # two routes share 1 * 100 * 30 / 30 seats.
    assert (routes["capacity"] == 50).all()
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


def test_route_network_rejects_an_unknown_choice_of_disadvantage():
    with pytest.raises(ValueError, match="disadvantage must be one of"):
        network(make_areas([1, 1]), disadvantage="district")


def test_route_network_rejects_an_instance_with_no_district_at_or_below_the_decile():
    with pytest.raises(ValueError, match="No LSOA sits at or below"):
        network(make_areas([8, 9, 10]), decile=3)


def test_route_network_rejects_an_empty_route_set():
    with pytest.raises(br.EmptyRouteSet, match="so the route set is"):
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
    with pytest.raises(br.EmptyRouteSet, match="no district is route-eligible"):
        network(
            make_areas([1, 1]), school_scores=np.array([1.0, 1.0]), local_radius=50_000
        )


def test_route_network_rejects_school_scores_that_do_not_align_with_the_coordinates():
    # Misaligned scores would gate on the wrong schools, plausibly and silently.
    with pytest.raises(ValueError, match="school_scores must align"):
        network(make_areas([1, 1]), school_scores=np.array([0.0]))


def test_route_network_rejects_places_that_do_not_align_with_the_schools():
    with pytest.raises(ValueError, match="school_places must align"):
        network(make_areas([1, 1]), school_places=np.array([100]))


def test_route_network_rejects_cohorts_that_do_not_align_with_the_areas():
    with pytest.raises(ValueError, match="district_cohort must align"):
        network(make_areas([1, 1]), district_cohort=np.array([10]))


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_route_network_rejects_a_capacity_scale_that_is_not_positive(scale):
    with pytest.raises(ValueError, match="capacity_scale must be positive"):
        network(make_areas([1, 1]), capacity_scale=scale)


def test_route_network_seats_each_school_on_the_regions_disadvantaged_share():
    # 100 students in the region, 26 of them disadvantaged, counting the 10 of
    # district 2 though it sits above the decile, and the schools offer 100
    # and 200 places: the routes into a school hold k * places * 26 / 100
    # seats between them.
    areas = make_areas([1, 1, 5], scores=[0.5, 0.2, 0.2])
    routes = network(
        areas,
        school_places=np.array([100, 200]),
        district_cohort=np.array([20, 30, 50]),
        capacity_scale=1.5,
    )

    # District 0 sits on school 0 and district 1 on school 1, so each school
    # has one route.
    np.testing.assert_array_equal(routes["district_idx"], [0, 1])
    np.testing.assert_array_equal(routes["school_idx"], [1, 0])
    np.testing.assert_array_equal(routes["capacity"], [78, 39])  # 1.5*200*0.26


def test_route_network_rounds_a_schools_seats_to_the_nearest_unless_rounding_up(
    capsys,
):
    # 20 of the 100 students are disadvantaged, so the one far school's 7
    # places give its routes 1.4 seats: 1 to the nearest seat, which the tie
    # hands to district 0, leaving district 1 unrouted, and 2 rounded up.
    areas = make_areas([1, 1, 5], scores=[1.0, 1.0, 0.0])
    kwargs = {
        "school_places": np.array([7]),
        "district_cohort": np.array([10, 10, 80]),
        "capacity_scale": 1.0,
    }
    school = np.array([[100_000.0, 0.0]])

    nearest = network(areas, school, SCHOOL_P8[:1], **kwargs)
    out = capsys.readouterr().out
    assert "1 of the 2 routes beyond 3218m would hold no seat" in out
    assert "No secondary school beyond 3218m holds a seat" in out
    np.testing.assert_array_equal(nearest["district_idx"], [0])
    np.testing.assert_array_equal(nearest["capacity"], [1])

    up = network(areas, school, SCHOOL_P8[:1], round_up=True, **kwargs)
    np.testing.assert_array_equal(up["district_idx"], [0, 1])
    np.testing.assert_array_equal(up["capacity"], [1, 1])


def test_route_network_rejects_a_route_set_holding_no_seat():
    with pytest.raises(br.EmptyRouteSet, match="so the route set is"):
        network(
            make_areas([1, 1, 9], scores=[1.0, 1.0, 0.0]),
            district_cohort=np.array([1, 1, 1_000_000]),
        )


@pytest.mark.parametrize(
    ("progressivity", "seats"), [(0.0, [10, 10]), (1.0, [15, 5]), (2.0, [18, 2])]
)
def test_route_network_leans_seats_towards_the_deprived_districts(progressivity, seats):
    # Districts at deciles 1 and 3 of threshold 3 hold the region's 20
    # disadvantaged students out of 100, 10 each, routed to one far school
    # admitting 100, so its routes hold 20 seats at every p. At p = 1 the
    # weights are 1 and 1/3, splitting them 15 and 5; at p = 2 they are 1 and
    # 1/9, splitting them 18 and 2.
    routes = network(
        make_areas([1, 3, 5], scores=[1.0, 1.0, 0.0]),
        np.array([[100_000.0, 0.0]]),
        SCHOOL_P8[:1],
        school_places=np.array([100]),
        district_cohort=np.array([10, 10, 80]),
        decile=3,
        capacity_scale=1.0,
        progressivity=progressivity,
    )

    np.testing.assert_array_equal(routes["district_idx"], [0, 1])
    np.testing.assert_array_equal(routes["capacity"], seats)


@pytest.mark.parametrize(("linear", "seats"), [(False, [13, 7]), (True, [12, 8])])
def test_route_network_weights_the_middle_deciles_by_the_profile_chosen(linear, seats):
    # Districts at deciles 1 and 2 of threshold 3 hold the region's 20
    # disadvantaged students out of 100, 10 each, routed to one far school
    # admitting 100, fully progressive, so its routes hold 20 seats. By 1 / D
    # the weights are 1 and 1/2, splitting them 13.3 and 6.7; by the linear
    # profile they are 1 and 2/3, splitting them 12 and 8.
    routes = network(
        make_areas([1, 2, 5], scores=[1.0, 1.0, 0.0]),
        np.array([[100_000.0, 0.0]]),
        SCHOOL_P8[:1],
        school_places=np.array([100]),
        district_cohort=np.array([10, 10, 80]),
        decile=3,
        capacity_scale=1.0,
        progressivity=1.0,
        linear=linear,
    )

    np.testing.assert_array_equal(routes["district_idx"], [0, 1])
    np.testing.assert_array_equal(routes["capacity"], seats)


def test_route_network_rejects_a_negative_progressivity():
    with pytest.raises(ValueError, match="progressivity must be non-negative"):
        network(make_areas([1, 1]), progressivity=-0.1)


@pytest.mark.parametrize(
    ("disadvantage", "seats"),
    [("score", [10, 40]), ("score-open", [10, 40]), ("decile", [10, 20])],
)
def test_route_network_seats_the_disadvantaged_students(disadvantage, seats):
    # Two districts of 10 and 20 students out of 100, scoring 0.3 and 0.6, are
    # routed to one far school admitting 100, the weights flat; a third of 70
    # scoring 0.5 sits above the decile. Drawn from the scores, 3, 12 and 35
    # are disadvantaged, so the school's routes hold 50 seats split 3 to 12,
    # whoever rides. Under "decile" the 30 students of the first two are.
    routes = network(
        make_areas([1, 1, 5], scores=[0.3, 0.6, 0.5]),
        np.array([[100_000.0, 0.0]]),
        SCHOOL_P8[:1],
        school_places=np.array([100]),
        district_cohort=np.array([10, 20, 70]),
        capacity_scale=1.0,
        progressivity=0.0,
        disadvantage=disadvantage,
    )

    np.testing.assert_array_equal(routes["district_idx"], [0, 1])
    np.testing.assert_array_equal(routes["capacity"], seats)


def test_route_network_rejects_route_eligible_districts_holding_no_students():
    with pytest.raises(br.EmptyRouteSet, match="hold no disadvantaged student"):
        network(make_areas([1, 1, 9]), district_cohort=np.array([0, 0, 100]))


def test_route_network_rejects_route_eligible_districts_holding_no_disadvantaged():
    # Every student is there, but none of them is disadvantaged.
    with pytest.raises(br.EmptyRouteSet, match="hold no disadvantaged student"):
        network(make_areas([1, 1, 9], scores=[0.0, 0.0, 1.0]))


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
    assert "2 districts at IDACI decile 3 or below, 1 of them" in out


def test_route_network_keeps_each_schools_shortest_route_on_all_its_seats(capsys):
    # School 0 is nearest district 1 and school 1 ties districts 0 and 2,
    # keeping the first. District 2 is nearest no school, which is not the
    # distance condition leaving it unrouted, so it is not named as such. Each
    # route holds all 1 * 100 * 30 / 30 seats of its school, the seats the
    # school's two routes share in the whole network.
    areas = make_areas([1, 1, 1])

    routes = network(areas, shortest_only=True)

    np.testing.assert_array_equal(routes["district_idx"], [0, 1])
    np.testing.assert_array_equal(routes["school_idx"], [1, 0])
    np.testing.assert_array_equal(routes["capacity"], [100, 100])
    out = capsys.readouterr().out
    assert "No secondary school beyond" not in out
    assert "each school keeping only its route to the nearest district" in out
    pd.testing.assert_series_equal(
        routes.groupby("school_idx")["capacity"].sum(),
        network(areas).groupby("school_idx")["capacity"].sum(),
    )


def test_route_network_pairs_each_school_with_one_of_the_lowest_ranked_districts(
    capsys,
):
    # Districts at 0, 10, 20 and 30km, schools at 0 and 5km. The two of lowest
    # rank, districts 1 and 3, are routed though they sit above the decile.
    # Both schools are nearest district 1, but each takes a district of its
    # own: school 0 district 1 and school 1 district 3, the longest route
    # 25km, rather than the other way round, the longest 30km. Each route
    # holds all 1 * 100 * 40 / 40 seats of its school.
    areas = make_areas([1, 5, 1, 5], ranks=[4, 1, 3, 2])

    routes = network(areas, np.array([[0.0, 0.0], [5_000.0, 0.0]]), bottleneck=True)

    np.testing.assert_array_equal(routes["district_idx"], [1, 3])
    np.testing.assert_array_equal(routes["school_idx"], [0, 1])
    np.testing.assert_array_equal(routes["capacity"], [100, 100])
    out = capsys.readouterr().out
    assert "2 districts of lowest IDACI rank" in out
    assert "each school paired with a district of its own" in out
    assert "the longest route 25000m" in out


def test_route_network_rejects_shortest_only_and_bottleneck_together():
    with pytest.raises(ValueError, match="only one may be set"):
        network(make_areas([1, 1]), shortest_only=True, bottleneck=True)


# --------------------------------------------------------------------------
# save_routes
# --------------------------------------------------------------------------


def test_save_routes_writes_the_set_whole_and_by_axis(tmp_path):
    out_dir = tmp_path / "out"
    routes = br.build_routes(
        make_areas([1, 1]), SCHOOL_XY, br.MIN_ROUTE_DISTANCE, WEIGHT, SCHOOL_SEATS
    )

    br.save_routes(routes, out_dir)

    pd.testing.assert_frame_equal(
        pd.read_csv(out_dir / br.ROUTES_CSV), routes, check_dtype=False
    )
    with np.load(out_dir / br.ROUTES_NPZ) as axes:
        np.testing.assert_array_equal(axes["route_capacities"], routes["capacity"])
        np.testing.assert_array_equal(axes["route_school_idx"], routes["school_idx"])
        np.testing.assert_array_equal(
            axes["route_district_idx"], routes["district_idx"]
        )
        assert all(axes[name].dtype == np.int32 for name in axes.files)
