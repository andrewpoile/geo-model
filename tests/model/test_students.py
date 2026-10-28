import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from geo_model.model.settings import SEED
from geo_model.model.students import (
    cohort_sizes,
    disadvantaged_cohort,
    disadvantaged_group,
    disadvantaged_students,
    draw_cohort_sizes,
    sample_in_polygon,
    sample_students,
    student_samples,
)

UNIT_SQUARE = shapely.box(0.0, 0.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# sample_in_polygon
# --------------------------------------------------------------------------


def test_sample_in_polygon_returns_requested_count_inside_geometry():
    points = sample_in_polygon(UNIT_SQUARE, (0.5, 0.5), 50, np.random.default_rng(0))
    assert points.shape == (50, 2)
    assert shapely.contains_xy(UNIT_SQUARE, points[:, 0], points[:, 1]).all()


def test_sample_in_polygon_is_reproducible_from_the_seed():
    a = sample_in_polygon(UNIT_SQUARE, (0.5, 0.5), 20, np.random.default_rng(7))
    b = sample_in_polygon(UNIT_SQUARE, (0.5, 0.5), 20, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "geom, size",
    [(UNIT_SQUARE, 0), (shapely.Polygon(), 10)],
    ids=["no-points-wanted", "empty-polygon"],
)
def test_sample_in_polygon_degenerate_cases_return_no_points(geom, size):
    points = sample_in_polygon(geom, (0.5, 0.5), size, np.random.default_rng(0))
    assert points.shape == (0, 2)


def test_sample_in_polygon_raises_when_centre_lies_far_outside():
    # The spread is set by the polygon's own bounds, so a centre 1000 widths
    # away never lands a candidate inside and rejection sampling cannot finish.
    with pytest.raises(RuntimeError, match="Rejection sampling failed"):
        sample_in_polygon(UNIT_SQUARE, (1000.0, 1000.0), 5, np.random.default_rng(0))


# --------------------------------------------------------------------------
# sample_students
# --------------------------------------------------------------------------


@pytest.fixture
def two_areas():
    borders = gpd.GeoSeries([shapely.box(0, 0, 1, 1), shapely.box(10, 0, 11, 1)])
    centroids = gpd.GeoSeries([shapely.Point(0.5, 0.5), shapely.Point(10.5, 0.5)])
    return borders, centroids


def test_sample_students_places_each_area_its_own_students(two_areas):
    borders, centroids = two_areas
    xy, area = sample_students(
        borders, centroids, np.array([3, 5]), np.random.default_rng(1)
    )

    assert xy.shape == (8, 2)
    np.testing.assert_array_equal(area, [0, 0, 0, 1, 1, 1, 1, 1])
    for i, geom in enumerate(borders):
        assert shapely.contains_xy(geom, xy[area == i, 0], xy[area == i, 1]).all()


def test_sample_students_with_no_students_returns_empty_arrays(two_areas):
    borders, centroids = two_areas
    xy, area = sample_students(
        borders, centroids, np.array([0, 0]), np.random.default_rng(1)
    )
    assert xy.shape == (0, 2)
    assert area.shape == (0,)


def test_sample_students_rejects_misaligned_inputs(two_areas):
    borders, centroids = two_areas
    with pytest.raises(ValueError, match="must align"):
        sample_students(
            borders, centroids, np.array([1, 2, 3]), np.random.default_rng(1)
        )


# --------------------------------------------------------------------------
# cohort_sizes and draw_cohort_sizes
# --------------------------------------------------------------------------


def cohorts(mean, sd):
    """Areas carrying the eleven-year-olds' mean and SD alone."""
    return pd.DataFrame({"Age 11 Mean": mean, "Age 11 SD": sd})


def test_cohort_sizes_is_the_mean_cohort():
    np.testing.assert_array_equal(
        cohort_sizes(cohorts([12.5, 3.25], [2.0, 1.0]), "secondary"), [12.5, 3.25]
    )


def test_draw_cohort_sizes_rounds_a_cohort_that_never_varies_to_its_mean():
    drawn = draw_cohort_sizes(
        cohorts([12.4, 3.6], [0.0, 0.0]), "secondary", np.random.default_rng(0)
    )

    np.testing.assert_array_equal(drawn, [12, 4])


def test_draw_cohort_sizes_places_no_students_for_a_negative_draw():
    drawn = draw_cohort_sizes(
        cohorts([-50.0] * 100, [1.0] * 100), "secondary", np.random.default_rng(0)
    )

    np.testing.assert_array_equal(drawn, 0)


def test_draw_cohort_sizes_draws_around_the_mean_reproducibly():
    areas = cohorts([20.0] * 10000, [4.0] * 10000)

    drawn = draw_cohort_sizes(areas, "secondary", np.random.default_rng(0))

    np.testing.assert_array_equal(
        drawn, draw_cohort_sizes(areas, "secondary", np.random.default_rng(0))
    )
    assert drawn.dtype == np.int64
    # Rounding adds a variance of 1/12 to the normal's.
    assert drawn.mean() == pytest.approx(20, abs=0.2)
    assert drawn.std() == pytest.approx(np.sqrt(16 + 1 / 12), abs=0.2)


# --------------------------------------------------------------------------
# disadvantaged_students
# --------------------------------------------------------------------------


def test_disadvantaged_students_holds_each_area_to_its_rounded_share():
    # Scores of 0, 0.25, 0.6 and 1 over 4, 4, 5 and 3 students: 0, 1, 3 and 3.
    areas = pd.DataFrame({"IDACI Score": [0.0, 0.25, 0.6, 1.0]})
    student_lsoa = np.repeat(np.arange(4), [4, 4, 5, 3])

    drawn = disadvantaged_students(student_lsoa, areas, np.random.default_rng(0))

    assert drawn.dtype == bool
    np.testing.assert_array_equal(
        np.bincount(student_lsoa[drawn], minlength=4), [0, 1, 3, 3]
    )


def test_disadvantaged_students_is_reproducible_from_the_seed():
    areas = pd.DataFrame({"IDACI Score": [0.5]})
    student_lsoa = np.zeros(40, dtype=np.int64)

    first, again = (
        disadvantaged_students(student_lsoa, areas, np.random.default_rng(3))
        for _ in range(2)
    )
    other = disadvantaged_students(student_lsoa, areas, np.random.default_rng(4))

    np.testing.assert_array_equal(first, again)
    # Another seed picks other students of the same area, as many of them.
    assert not np.array_equal(first, other)
    assert first.sum() == other.sum() == 20


# --------------------------------------------------------------------------
# disadvantaged_group
# --------------------------------------------------------------------------


@pytest.mark.parametrize("disadvantage", ["score", "score-open"])
def test_disadvantaged_group_under_the_scores_is_the_group_drawn(disadvantage):
    areas = pd.DataFrame({"IDACI Decile": [1, 5]})
    drawn = np.array([False, True, True])

    group = disadvantaged_group(np.array([0, 0, 1]), drawn, areas, 3, disadvantage)

    np.testing.assert_array_equal(group, drawn)


def test_disadvantaged_group_under_decile_labels_each_student_by_its_district():
    areas = pd.DataFrame({"IDACI Decile": [1, 5, 3]})
    student_lsoa = np.array([0, 1, 1, 2, 0])

    group = disadvantaged_group(
        student_lsoa, np.zeros(5, dtype=bool), areas, 3, "decile"
    )

    np.testing.assert_array_equal(group, [True, False, False, True, True])


def test_disadvantaged_group_rejects_an_unknown_choice():
    with pytest.raises(ValueError, match="must be one of"):
        disadvantaged_group(
            np.array([0]),
            np.array([True]),
            pd.DataFrame({"IDACI Decile": [1]}),
            3,
            "district",
        )


# --------------------------------------------------------------------------
# student_samples: against the real data folder
# --------------------------------------------------------------------------


def test_student_samples_draws_one_reproducible_sample_per_seed(areas, to_school):
    first = list(student_samples(areas, 2, to_school))
    second = list(student_samples(areas, 2, to_school))

    assert len(first) == 2
    # Each seed draws its own cohort, from the head of its own stream.
    streams = np.random.SeedSequence(SEED).spawn(2)
    cohorts = [
        draw_cohort_sizes(areas, "secondary", np.random.default_rng(stream))
        for stream in streams
    ]
    for (_, student_lsoa, _, _, _), cohort in zip(first, cohorts):
        np.testing.assert_array_equal(
            np.bincount(student_lsoa, minlength=len(areas)), cohort
        )
    assert not np.array_equal(cohorts[0], cohorts[1])
    # Different seeds draw different students and tastes, the same seed the
    # same ones.
    assert not np.array_equal(first[0][0][:100], first[1][0][:100])
    tastes = [
        [sample[3].generate_state(4) for sample in run] for run in (first, second)
    ]
    assert not np.array_equal(tastes[0][0], tastes[0][1])
    for sample_a, sample_b, taste_a, taste_b in zip(first, second, *tastes):
        for a, b in zip(sample_a[:3], sample_b[:3]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(taste_a, taste_b)
        # Each carries its students' distance to every school, at single
        # precision.
        assert sample_a[4].dtype == np.float32
        np.testing.assert_array_equal(
            sample_a[4], to_school(sample_a[0]).astype(np.float32)
        )


def test_student_samples_draw_each_district_its_idaci_share_after_the_locations(
    areas, to_school
):
    student_xy, student_lsoa, drawn, _, _ = next(student_samples(areas, 1, to_school))

    np.testing.assert_array_equal(
        np.bincount(student_lsoa[drawn], minlength=len(areas)),
        disadvantaged_cohort(areas, np.bincount(student_lsoa, minlength=len(areas))),
    )
    # The locations are the ones a draw of the cohort and locations alone
    # takes from the seed's stream, so drawing the group after them leaves
    # them as they were.
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(1)[0])
    alone, _ = sample_students(
        areas["Borders"],
        areas["Centroids"],
        draw_cohort_sizes(areas, "secondary", rng),
        rng,
    )
    np.testing.assert_array_equal(student_xy, alone)
