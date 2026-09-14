import numpy as np
import pytest

from geo_model import build_prefs as bp
from geo_model import build_routes as br
from geo_model import load_data as ld

DATA = ld.POPULATION_XLSX.parent.parent


# --------------------------------------------------------------------------
# main: the whole pipeline, against the real data folder
# --------------------------------------------------------------------------

pytestmark_data = pytest.mark.skipif(
    not DATA.is_dir(), reason=f"the {DATA} folder is not present"
)


@pytest.fixture
def redirected_outputs(tmp_path, monkeypatch):
    """Send every file the pipeline writes to a temporary folder.

    The real temp/ folder holds the artefacts the rest of the model is run
    from, so a test must not overwrite them.
    """
    monkeypatch.setattr(bp, "OUTPUT_NPZ", tmp_path / "prefprio.npz")
    monkeypatch.setattr(ld, "POPULATION_CACHE", tmp_path / "population_lsoa.pkl")
    monkeypatch.setattr(br, "ROUTES_CSV", tmp_path / "secondary_routes.csv")
    monkeypatch.setattr(br, "ROUTES_NPZ", tmp_path / "secondary_routes.npz")
    return tmp_path


@pytestmark_data
def test_main_writes_a_consistent_instance_for_both_phases(redirected_outputs):
    bp.main()

    assert bp.OUTPUT_NPZ.exists()
    assert br.ROUTES_CSV.exists() and br.ROUTES_NPZ.exists()

    with np.load(bp.OUTPUT_NPZ) as built:
        routes = np.load(br.ROUTES_NPZ)
        n_routes = len(routes["route_capacities"])

        for phase, phase_routes in [("primary", 0), ("secondary", n_routes)]:
            preferences = built[f"{phase}_student_preferences"]
            priorities = built[f"{phase}_school_priorities"]
            capacities = built[f"{phase}_school_capacities"]

            n_schools, route_axis, n_students = priorities.shape
            assert preferences.shape == (n_students, preferences.shape[1], 2)
            # The route axis carries every route plus the routeless bundle.
            assert route_axis == phase_routes + 1
            assert capacities.shape == (n_schools,)
            assert n_students > 0 and n_schools > 0
            assert (capacities >= 1).all()

            # Every ranked bundle is a real school, or the padding a short list
            # carries, and every route index falls on the route axis.
            schools, bundle_routes = preferences[..., 0], preferences[..., 1]
            assert ((schools == -1) | (schools < n_schools)).all()
            assert ((bundle_routes == -1) | (bundle_routes < phase_routes)).all()
            assert (preferences[schools == -1] == -1).all()

            # Every student carries the district it was drawn in, and every
            # district a decile, so the matching can be scored for segregation.
            district = built[f"{phase}_student_district"]
            assert district.shape == (n_students,)
            assert ((0 <= district) & (district < len(built["district_decile"]))).all()

        decile = built["district_decile"]
        assert ((1 <= decile) & (decile <= 10)).all()


@pytestmark_data
def test_main_only_routes_the_secondary_phase(redirected_outputs):
    bp.main()

    with np.load(bp.OUTPUT_NPZ) as built:
        # Routes serve the schools the disadvantaged districts cannot reach
        # unaided, which the model builds for the secondary phase alone.
        assert built["primary_school_priorities"].shape[1] == 1
        assert built["secondary_school_priorities"].shape[1] > 1
        assert (built["primary_student_preferences"][..., 1] == -1).all()
        assert (built["secondary_student_preferences"][..., 1] > -1).any()


@pytestmark_data
def test_main_is_reproducible_from_the_seed(redirected_outputs):
    bp.main()
    first = dict(np.load(bp.OUTPUT_NPZ))

    bp.main()
    with np.load(bp.OUTPUT_NPZ) as second:
        for name, array in first.items():
            np.testing.assert_array_equal(array, second[name], err_msg=name)
