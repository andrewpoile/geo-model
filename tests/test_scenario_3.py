import numpy as np
import pytest
from scipy.spatial import distance as spdist

from geo_model import build_routes as br
from geo_model import load_data as ld
from geo_model import scenario_1 as s1
from geo_model import scenario_3 as s3
from geo_model.build_prefs import cohort_sizes
from geo_model.dissimilarity import settings_routes

DATA = ld.POPULATION_DIR.parent.parent


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_scenario_pairs_the_most_disadvantaged_districts_largest_first():
    areas = ld.load_areas()
    _, schools = ld.load_schools()
    places = schools["PlacesOffered"].to_numpy()
    targets = (
        areas.assign(n=br.disadvantaged_cohort(areas, cohort_sizes(areas, "secondary")))
        .sort_values(["n", "IDACI"], ascending=[False, True])
        .head(len(schools))
    )

    routes = settings_routes(s3.SCENARIO, areas, schools, s1.ROUND_UP)

    # One route per school, each from a district of its own, the districts
    # being the region's holding the most disadvantaged students, as many as
    # there are schools.
    assert sorted(routes["school_idx"]) == list(range(len(schools)))
    assert routes["district_idx"].is_unique
    assert set(routes["district_idx"]) == set(targets.index)
    distances = spdist.cdist(
        br.centroid_xy(targets), schools[["Easting", "Northing"]].to_numpy()
    )
    school_of = routes.set_index("district_idx")["school_idx"]
    # Each district, in turn from the most disadvantaged, took the school
    # offering the most places of those beyond the minimum distance that no
    # district before it took.
    taken = []
    for district, row in zip(targets.index, distances, strict=True):
        school = school_of[district]
        assert row[school] > br.MIN_ROUTE_DISTANCE
        free = np.setdiff1d(np.flatnonzero(row > br.MIN_ROUTE_DISTANCE), taken)
        assert places[school] == places[free].max()
        taken.append(school)
