from dataclasses import replace

from geo_model.analysis.scoring import settings_routes
from geo_model.model import routes as rt
from geo_model.model.settings import MIN_ROUTE_DISTANCE
from geo_model.model.students import cohort_sizes, disadvantaged_cohort
from geo_model.scripts import scenario_2 as s2


def test_scenario_pairs_every_school_with_one_of_the_most_deprived_districts(
    areas, secondary, district_distance, district_minutes
):
    # Every district in IDACI rank order may be routed to the schools beyond
    # the minimum distance if it holds a disadvantaged student, no school
    # scoring above the scenario's Progress 8 threshold.
    ranked = areas.sort_values("IDACI", kind="stable")
    disadvantaged = disadvantaged_cohort(areas, cohort_sizes(areas, "secondary"))
    allowed = (district_distance[ranked.index] > MIN_ROUTE_DISTANCE) & (
        disadvantaged[ranked.index] > 0
    )[:, None]
    targets = ranked[rt.first_districts(allowed)]

    routes = settings_routes(
        s2.SCENARIO, areas, secondary, district_distance, district_minutes
    )

    # One route per school, each from a district of its own, every school's
    # seats covering any district's disadvantaged students at the scenario's
    # capacity scale, the districts the most deprived that can each have a
    # school of their own.
    assert sorted(routes["school_idx"]) == list(range(len(secondary)))
    assert routes["district_idx"].is_unique
    assert set(routes["district_idx"]) == set(targets.index)
    distances = district_distance[routes["district_idx"]]
    lengths = district_distance[routes["district_idx"], routes["school_idx"]]
    assert (lengths > MIN_ROUTE_DISTANCE).all()
    # No pairing of the same districts and schools has a shorter longest
    # route: the pairs shorter than this one's longest leave some school
    # without a district of its own.
    shorter = (distances > MIN_ROUTE_DISTANCE) & (distances < lengths.max())
    assert rt.matching_size(shorter) < len(secondary)


def test_scenario_shares_a_district_between_schools_at_a_small_capacity_scale(
    areas, secondary, district_distance, district_minutes
):
    routes = settings_routes(
        replace(s2.SCENARIO, capacity_scale=0.1),
        areas,
        secondary,
        district_distance,
        district_minutes,
    )

    # Still one route per school, some district now taking several, each
    # short of its disadvantaged students without the smallest of its routes.
    assert sorted(routes["school_idx"]) == list(range(len(secondary)))
    assert not routes["district_idx"].is_unique
    seats = routes.groupby("district_idx")["capacity"]
    students = disadvantaged_cohort(areas, cohort_sizes(areas, "secondary"))
    assert (seats.sum() - seats.min() < students[seats.sum().index]).all()
