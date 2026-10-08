import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import distance as spdist

from geo_model import build_routes as br
from geo_model import load_data as ld
from geo_model import scenario_1 as s1
from geo_model import scenario_2 as s2
from geo_model.dissimilarity import settings_routes

DATA = ld.POPULATION_DIR.parent.parent


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_scenario_pairs_every_school_with_one_of_the_most_deprived_districts():
    areas = ld.load_areas()
    _, schools = ld.load_schools()
    school_xy = schools[["Easting", "Northing"]].to_numpy()
    targets = areas.nsmallest(len(schools), "IDACI")

    routes = settings_routes(s2.SCENARIO, areas, schools, s1.ROUND_UP)

    # One route per school, each from a district of its own, the districts
    # being the region's of lowest IDACI rank, as many as there are schools.
    assert sorted(routes["school_idx"]) == list(range(len(schools)))
    assert routes["district_idx"].is_unique
    assert set(routes["district_idx"]) == set(targets.index)
    lengths = np.linalg.norm(
        br.centroid_xy(areas.iloc[routes["district_idx"]])
        - school_xy[routes["school_idx"]],
        axis=1,
    )
    assert (lengths > br.MIN_ROUTE_DISTANCE).all()
    # No pairing of the same districts and schools has a shorter longest
    # route: the pairs shorter than this one's longest leave some school
    # without a district of its own.
    distances = spdist.cdist(br.centroid_xy(targets), school_xy)
    shorter = (distances > br.MIN_ROUTE_DISTANCE) & (distances < lengths.max())
    assert (maximum_bipartite_matching(csr_matrix(shorter), perm_type="row") < 0).any()
