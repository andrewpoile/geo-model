import numpy as np
import pytest
from scipy.spatial import distance as spdist

from geo_model import build_prefs as bp
from geo_model import build_routes as br
from geo_model import load_data as ld
from geo_model import scenario_1 as s1
from geo_model.dissimilarity import settings_routes

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
