import numpy as np
import pandas as pd
import pytest

from geo_model import build_prefs as bp
from geo_model import build_routes as br
from geo_model import load_data as ld

DATA = ld.POPULATION_DIR.parent.parent


# --------------------------------------------------------------------------
# build: the whole pipeline, against the real data folder
# --------------------------------------------------------------------------

pytestmark_data = pytest.mark.skipif(
    not DATA.is_dir(), reason=f"the {DATA} folder is not present"
)


@pytest.fixture
def build(tmp_path, monkeypatch):
    """Build Southampton's instance into a temporary folder, returned.

    The real output folder holds the artefacts the rest of the model is run
    from, so a test must not overwrite them.
    """
    monkeypatch.setattr(ld, "POPULATION_CACHE", tmp_path / "population_lsoa.pkl")

    def built():
        primary, secondary = ld.load_schools()
        bp.build(ld.load_areas(), primary, secondary, tmp_path)
        return tmp_path

    return built


@pytestmark_data
def test_build_writes_a_consistent_instance_for_both_phases(build):
    out_dir = build()

    assert (out_dir / bp.OUTPUT_NPZ).exists()
    assert (out_dir / br.ROUTES_CSV).exists() and (out_dir / br.ROUTES_NPZ).exists()

    with np.load(out_dir / bp.OUTPUT_NPZ) as built:
        routes = np.load(out_dir / br.ROUTES_NPZ)
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
            schools, bundle_routes = preferences[..., 0], preferences[..., 1]
            assert ((schools == -1) | (schools < n_schools)).all()
            assert ((bundle_routes == -1) | (bundle_routes < phase_routes)).all()
            assert (preferences[schools == -1] == -1).all()
            # Every bundle on a list carries its school's rank, padding none.
            np.testing.assert_array_equal(ranks == -1, schools == -1)

            # Every student carries the district it was drawn in and whether
            # it is disadvantaged, so the matching can be scored for
            # segregation.
            district = built[f"{phase}_student_district"]
            disadvantaged = built[f"{phase}_student_disadvantaged"]
            assert district.shape == disadvantaged.shape == (n_students,)
            assert (district >= 0).all()
            assert disadvantaged.dtype == bool
            assert disadvantaged.any() and not disadvantaged.all()


@pytestmark_data
def test_build_only_routes_the_secondary_phase(build):
    with np.load(build() / bp.OUTPUT_NPZ) as built:
        # Routes serve the schools the disadvantaged districts cannot reach
        # unaided, which the model builds for the secondary phase alone.
        assert (built["primary_student_preferences"][..., 1] == -1).all()
        assert (built["secondary_student_preferences"][..., 1] > -1).any()


# --------------------------------------------------------------------------
# cohort_sizes and draw_cohort_sizes
# --------------------------------------------------------------------------


def cohorts(mean, sd):
    """Areas carrying the eleven-year-olds' mean and SD alone."""
    return pd.DataFrame({"Age 11 Mean": mean, "Age 11 SD": sd})


def test_cohort_sizes_is_the_mean_cohort():
    np.testing.assert_array_equal(
        bp.cohort_sizes(cohorts([12.5, 3.25], [2.0, 1.0]), "secondary"), [12.5, 3.25]
    )


def test_draw_cohort_sizes_rounds_a_cohort_that_never_varies_to_its_mean():
    drawn = bp.draw_cohort_sizes(
        cohorts([12.4, 3.6], [0.0, 0.0]), "secondary", np.random.default_rng(0)
    )

    np.testing.assert_array_equal(drawn, [12, 4])


def test_draw_cohort_sizes_places_no_students_for_a_negative_draw():
    drawn = bp.draw_cohort_sizes(
        cohorts([-50.0] * 100, [1.0] * 100), "secondary", np.random.default_rng(0)
    )

    np.testing.assert_array_equal(drawn, 0)


def test_draw_cohort_sizes_draws_around_the_mean_reproducibly():
    areas = cohorts([20.0] * 10000, [4.0] * 10000)

    drawn = bp.draw_cohort_sizes(areas, "secondary", np.random.default_rng(0))

    np.testing.assert_array_equal(
        drawn, bp.draw_cohort_sizes(areas, "secondary", np.random.default_rng(0))
    )
    assert drawn.dtype == np.int64
    # Rounding adds a variance of 1/12 to the normal's.
    assert drawn.mean() == pytest.approx(20, abs=0.2)
    assert drawn.std() == pytest.approx(np.sqrt(16 + 1 / 12), abs=0.2)


# --------------------------------------------------------------------------
# secondary_instance
# --------------------------------------------------------------------------


def by_bundle(preferences, ranks):
    """Each student's ranks ordered by bundle rather than by preference."""
    key = preferences[..., 0] * (preferences[..., 1].max() + 2) + preferences[..., 1]
    return np.take_along_axis(ranks, np.argsort(key, axis=1), axis=1)


@pytestmark_data
def test_secondary_instance_ranks_nearest_first_without_performance_weight():
    areas = ld.load_areas()
    _, schools = ld.load_schools()
    rng = np.random.default_rng(0)
    student_xy, student_lsoa = bp.sample_students(
        areas["Borders"],
        areas["Centroids"],
        bp.draw_cohort_sizes(areas, "secondary", rng),
        rng,
    )

    preferences, _, _, _ = bp.secondary_instance(
        student_xy,
        student_lsoa,
        areas,
        schools,
        None,
        bp.disadvantaged_students(student_lsoa, areas, np.random.default_rng(1)),
        performance_weight=0.0,
        disadvantaged_performance_weight=0.0,
        noise_scale=0.0,
    )

    school_xy = schools[["Easting", "Northing"]].to_numpy()
    nearest = np.argmin(
        np.linalg.norm(student_xy[:, None] - school_xy[None], axis=2), axis=1
    )
    np.testing.assert_array_equal(preferences[:, 0, 0], nearest)


@pytestmark_data
def test_secondary_instance_ranks_each_group_on_its_own_weights():
    areas = ld.load_areas()
    _, schools = ld.load_schools()
    rng = np.random.default_rng(0)
    student_xy, student_lsoa = bp.sample_students(
        areas["Borders"],
        areas["Centroids"],
        bp.draw_cohort_sizes(areas, "secondary", rng),
        rng,
    )
    routes = br.route_network(
        areas,
        schools[["Easting", "Northing"]].to_numpy(),
        schools["P8MEA"].to_numpy(),
        schools["PlacesOffered"].to_numpy(),
        bp.cohort_sizes(areas, "secondary"),
    )
    disadvantaged = bp.disadvantaged_students(
        student_lsoa, areas, np.random.default_rng(1)
    )
    assert disadvantaged.any() and not disadvantaged.all()

    def instance(other, own):
        return bp.secondary_instance(
            student_xy,
            student_lsoa,
            areas,
            schools,
            routes,
            disadvantaged,
            *other,
            *own,
            noise_seed=0,
        )

    mixed = instance((0.1, 0.2), (0.7, 0.9))
    other = instance((0.1, 0.2), (0.1, 0.2))
    own = instance((0.7, 0.9), (0.7, 0.9))

    # The two settings rank the disadvantaged differently, so their rows show
    # whose weights they were ranked on.
    assert not np.array_equal(other[0][disadvantaged], own[0][disadvantaged])
    np.testing.assert_array_equal(mixed[0][~disadvantaged], other[0][~disadvantaged])
    np.testing.assert_array_equal(mixed[0][disadvantaged], own[0][disadvantaged])
    # School priorities do not depend on how students weigh schools, so every
    # bundle keeps its rank wherever the weights move it on the list.
    np.testing.assert_array_equal(by_bundle(*mixed[:2]), by_bundle(*other[:2]))


@pytestmark_data
def test_secondary_instance_offers_other_students_routes_only_when_open():
    areas = ld.load_areas()
    _, schools = ld.load_schools()
    rng = np.random.default_rng(0)
    student_xy, student_lsoa = bp.sample_students(
        areas["Borders"],
        areas["Centroids"],
        bp.draw_cohort_sizes(areas, "secondary", rng),
        rng,
    )
    disadvantaged = bp.disadvantaged_students(student_lsoa, areas, rng)
    routes = br.route_network(
        areas,
        schools[["Easting", "Northing"]].to_numpy(),
        schools["P8MEA"].to_numpy(),
        schools["PlacesOffered"].to_numpy(),
        bp.cohort_sizes(areas, "secondary"),
    )
    offered = {}
    for disadvantage in ("score", "score-open"):
        preferences, _, _, _ = bp.secondary_instance(
            student_xy,
            student_lsoa,
            areas,
            schools,
            routes,
            disadvantaged,
            disadvantage=disadvantage,
            noise_seed=0,
        )
        offered[disadvantage] = np.any(preferences[..., 1] >= 0, axis=1)

    # Under "score" the routes carry the disadvantaged alone; open, they carry
    # the other students of the same districts as well.
    assert offered["score"][disadvantaged].any()
    assert not offered["score"][~disadvantaged].any()
    assert offered["score-open"][~disadvantaged].any()
    np.testing.assert_array_equal(
        offered["score-open"][disadvantaged], offered["score"][disadvantaged]
    )


@pytestmark_data
def test_build_is_reproducible_from_the_seed(build):
    first = dict(np.load(build() / bp.OUTPUT_NPZ))

    with np.load(build() / bp.OUTPUT_NPZ) as second:
        for name, array in first.items():
            np.testing.assert_array_equal(array, second[name], err_msg=name)
