import numpy as np

from geo_model.analysis.scoring import settings_routes
from geo_model.model.settings import MIN_ROUTE_DISTANCE
from geo_model.model.students import cohort_sizes, disadvantaged_cohort
from geo_model.scripts import scenario_3 as s3


def test_scenario_pairs_the_most_disadvantaged_districts_largest_first(
    areas, secondary, district_distance, district_minutes
):
    places = secondary["PlacesOffered"].to_numpy()
    targets = (
        areas.assign(n=disadvantaged_cohort(areas, cohort_sizes(areas, "secondary")))
        .sort_values(["n", "IDACI"], ascending=[False, True])
        .head(len(secondary))
    )

    routes = settings_routes(
        s3.SCENARIO, areas, secondary, district_distance, district_minutes, s3.ROUND_UP
    )

    # One route per school, each from a district of its own, the districts
    # being the region's holding the most disadvantaged students, as many as
    # there are schools.
    assert sorted(routes["school_idx"]) == list(range(len(secondary)))
    assert routes["district_idx"].is_unique
    assert set(routes["district_idx"]) == set(targets.index)
    distances = district_distance[targets.index]
    school_of = routes.set_index("district_idx")["school_idx"]
    # Each district, in turn from the most disadvantaged, took the school
    # offering the most places of those beyond the minimum distance that no
    # district before it took.
    taken = []
    for district, row in zip(targets.index, distances, strict=True):
        school = school_of[district]
        assert row[school] > MIN_ROUTE_DISTANCE
        free = np.setdiff1d(np.flatnonzero(row > MIN_ROUTE_DISTANCE), taken)
        assert places[school] == places[free].max()
        taken.append(school)
