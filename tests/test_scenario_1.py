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
    # is eligible, and one holding a disadvantaged student is routed.
    cohort = bp.cohort_sizes(areas, "secondary")
    disadvantaged = br.disadvantaged_cohort(areas, cohort)
    eligible = areas[
        (areas["IDACI Decile"] <= s1.SCENARIO.decile) & (disadvantaged > 0)
    ]
    nearest = eligible.index[
        spdist.cdist(br.centroid_xy(eligible), school_xy).argmin(axis=0)
    ]
    np.testing.assert_array_equal(routes["school_idx"], np.arange(len(schools)))
    np.testing.assert_array_equal(routes["district_idx"], nearest)
    # The one route holds all of its school's seats, rounded up.
    np.testing.assert_array_equal(
        routes["capacity"],
        np.ceil(
            s1.SCENARIO.capacity_scale
            * schools["PlacesOffered"].to_numpy()
            * disadvantaged.sum()
            / cohort.sum()
        ),
    )
    assert (routes["capacity"] >= 1).all()
