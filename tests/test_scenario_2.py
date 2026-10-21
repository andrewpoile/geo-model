import numpy as np
from scipy.spatial import distance as spdist

from geo_model import build_routes as br
from geo_model import scenario_1 as s1
from geo_model import scenario_2 as s2
from geo_model.build_prefs import cohort_sizes
from geo_model.dissimilarity import settings_routes


def test_scenario_pairs_every_school_with_one_of_the_most_deprived_districts(
    areas, secondary
):
    school_xy = secondary[["Easting", "Northing"]].to_numpy()
    # Every district in IDACI rank order may be routed to the schools beyond
    # the minimum distance if it holds a disadvantaged student, no school
    # scoring above the scenario's Progress 8 threshold.
    ranked = areas.sort_values("IDACI", kind="stable")
    disadvantaged = br.disadvantaged_cohort(areas, cohort_sizes(areas, "secondary"))
    allowed = (
        spdist.cdist(br.centroid_xy(ranked), school_xy) > br.MIN_ROUTE_DISTANCE
    ) & (disadvantaged[ranked.index] > 0)[:, None]
    targets = ranked[br.first_districts(allowed)]

    routes = settings_routes(s2.SCENARIO, areas, secondary, s1.ROUND_UP)

    # One route per school, each from a district of its own, the districts the
    # most deprived that can each have a school of their own.
    assert sorted(routes["school_idx"]) == list(range(len(secondary)))
    assert routes["district_idx"].is_unique
    assert set(routes["district_idx"]) == set(targets.index)
    routed = br.centroid_xy(areas.iloc[routes["district_idx"]])
    lengths = np.linalg.norm(routed - school_xy[routes["school_idx"]], axis=1)
    assert (lengths > br.MIN_ROUTE_DISTANCE).all()
    # No pairing of the same districts and schools has a shorter longest
    # route: the pairs shorter than this one's longest leave some school
    # without a district of its own.
    distances = spdist.cdist(routed, school_xy)
    shorter = (distances > br.MIN_ROUTE_DISTANCE) & (distances < lengths.max())
    assert br.matching_size(shorter) < len(secondary)
