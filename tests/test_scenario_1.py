import numpy as np
import pandas as pd
import pytest
from scipy.spatial import distance as spdist

from geo_model import build_prefs as bp
from geo_model import build_routes as br
from geo_model import load_data as ld
from geo_model import scenario_1 as s1
from geo_model.dissimilarity import settings_routes, student_samples
from geo_model.utils import disadvantaged_group, district_groups

DATA = ld.POPULATION_XLSX.parent.parent


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_every_school_keeps_one_route_from_its_nearest_district_holding_a_rider():
    areas = ld.load_areas()
    _, schools = ld.load_schools()
    school_xy = schools[["Easting", "Northing"]].to_numpy()

    routes = settings_routes(s1.SCENARIO, areas, schools, s1.ROUND_UP).sort_values(
        "school_idx"
    )

    # No condition on nearby schools, so every district at or below the decile
    # is eligible, and under "score" one holding a disadvantaged student rides.
    # A school's route leaves the nearest of them beyond the minimum distance.
    assert s1.SCENARIO.min_distance == br.MIN_ROUTE_DISTANCE
    riders = br.disadvantaged_cohort(areas, bp.cohort_sizes(areas, "secondary"))
    eligible = areas[(areas["IDACI Decile"] <= s1.SCENARIO.decile) & (riders > 0)]
    distances = spdist.cdist(br.centroid_xy(eligible), school_xy)
    distances[distances <= s1.SCENARIO.min_distance] = np.inf
    nearest = eligible.index[distances.argmin(axis=0)]
    np.testing.assert_array_equal(routes["school_idx"], np.arange(len(schools)))
    np.testing.assert_array_equal(routes["district_idx"], nearest)
    assert (routes["capacity"] >= 1).all()


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_every_seed_draws_each_districts_groups_alike():
    areas = ld.load_areas()
    sizes = bp.cohort_sizes(areas, "secondary")

    seeds = [
        district_groups(
            areas,
            student_lsoa,
            disadvantaged_group(
                student_lsoa, drawn, areas, s1.SCENARIO.decile, br.DISADVANTAGE
            ),
        )
        for _, student_lsoa, drawn in student_samples(areas, sizes, 2)
    ]

    pd.testing.assert_frame_equal(seeds[0], seeds[1])
    np.testing.assert_array_equal(seeds[0]["cohort"], sizes)
    np.testing.assert_array_equal(
        seeds[0]["disadvantaged"], br.disadvantaged_cohort(areas, sizes)
    )
