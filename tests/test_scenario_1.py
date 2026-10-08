import numpy as np
import pytest

from geo_model import build_routes as br
from geo_model import load_data as ld
from geo_model import scenario_1 as s1
from geo_model.dissimilarity import student_samples

DATA = ld.POPULATION_DIR.parent.parent


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_district_rows_count_every_seeds_own_students_by_district():
    areas = ld.load_areas()
    samples = list(student_samples(areas, 2))

    rows = s1.district_rows(samples, areas, s1.SCENARIO.decile, br.DISADVANTAGE)

    assert list(rows.columns[:2]) == ["seed", "LSOA21CD"]
    for seed, (_, student_lsoa, drawn, _) in enumerate(samples):
        seed_rows = rows[rows["seed"] == seed]
        np.testing.assert_array_equal(seed_rows["LSOA21CD"], areas["LSOA21CD"])
        np.testing.assert_array_equal(
            seed_rows["cohort"], np.bincount(student_lsoa, minlength=len(areas))
        )
        np.testing.assert_array_equal(
            seed_rows["disadvantaged"],
            np.bincount(student_lsoa[drawn], minlength=len(areas)),
        )
        assert seed_rows["share_of_disadvantaged"].sum() == pytest.approx(1)
        assert seed_rows["share_of_advantaged"].sum() == pytest.approx(1)
    # Each seed draws its cohorts afresh, so the seeds' rows differ.
    assert not np.array_equal(
        rows.loc[rows["seed"] == 0, "cohort"], rows.loc[rows["seed"] == 1, "cohort"]
    )
