import numpy as np

from geo_model.output import ROUTES_CSV, ROUTES_NPZ
from geo_model.scripts import build_prefs as bp

# --------------------------------------------------------------------------
# build: the whole pipeline, against the real data folder
# --------------------------------------------------------------------------


def build(areas, schools, out_dir):
    """Build the default region's instance into `out_dir`, returned.

    The real output folder holds the artefacts the rest of the model is run
    from, so a test must not overwrite them.
    """
    primary, secondary = schools
    bp.build(areas, primary, secondary, out_dir)
    return out_dir


def test_build_writes_a_consistent_instance_for_both_phases(areas, schools, tmp_path):
    out_dir = build(areas, schools, tmp_path)

    assert (out_dir / bp.OUTPUT_NPZ).exists()
    assert (out_dir / ROUTES_CSV).exists() and (out_dir / ROUTES_NPZ).exists()

    with (
        np.load(out_dir / bp.OUTPUT_NPZ) as built,
        np.load(out_dir / ROUTES_NPZ) as routes,
    ):
        n_routes = len(routes["route_capacities"])

        for phase, phase_routes in [("primary", 0), ("secondary", n_routes)]:
            preferences = built[f"{phase}_student_preferences"]
            ranks = built[f"{phase}_preference_ranks"]
            capacities = built[f"{phase}_school_capacities"]

            n_students, n_schools = len(preferences), len(capacities)
            assert preferences.shape == (n_students, preferences.shape[1], 2)
            assert ranks.shape == preferences.shape[:2]
            assert n_students > 0 and n_schools > 0
            assert (capacities >= 1).all()

            # Every ranked bundle is a real school, or the padding a short list
            # carries, and every route index is a route that was built.
            bundle_schools, bundle_routes = preferences[..., 0], preferences[..., 1]
            assert ((bundle_schools == -1) | (bundle_schools < n_schools)).all()
            assert ((bundle_routes == -1) | (bundle_routes < phase_routes)).all()
            assert (preferences[bundle_schools == -1] == -1).all()
            # Every bundle on a list carries its school's rank, padding none.
            np.testing.assert_array_equal(ranks == -1, bundle_schools == -1)

            # Every student carries the district it was drawn in and whether
            # it is disadvantaged, so the matching can be scored for
            # segregation.
            district = built[f"{phase}_student_district"]
            disadvantaged = built[f"{phase}_student_disadvantaged"]
            assert district.shape == disadvantaged.shape == (n_students,)
            assert (district >= 0).all()
            assert disadvantaged.dtype == bool
            assert disadvantaged.any() and not disadvantaged.all()

        # Routes serve the schools the disadvantaged districts cannot reach
        # unaided, which the model builds for the secondary phase alone.
        assert (built["primary_student_preferences"][..., 1] == -1).all()
        assert (built["secondary_student_preferences"][..., 1] > -1).any()


def test_build_is_reproducible_from_the_seed(areas, schools, tmp_path):
    with (
        np.load(build(areas, schools, tmp_path / "first") / bp.OUTPUT_NPZ) as first,
        np.load(build(areas, schools, tmp_path / "second") / bp.OUTPUT_NPZ) as second,
    ):
        assert first.files == second.files
        for name in first.files:
            np.testing.assert_array_equal(first[name], second[name], err_msg=name)
