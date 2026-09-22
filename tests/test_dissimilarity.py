import numpy as np
import pandas as pd
import pytest

from geo_model import build_prefs as bp
from geo_model import build_routes as br
from geo_model import dissimilarity as ds
from geo_model import load_data as ld
from geo_model.utils import MODES

DATA = ld.POPULATION_XLSX.parent.parent


# --------------------------------------------------------------------------
# disadvantaged_students
# --------------------------------------------------------------------------


def test_disadvantaged_students_labels_each_student_by_its_own_district():
    areas = pd.DataFrame({"IMD Decile": [1, 5, 3]})
    student_lsoa = np.array([0, 1, 1, 2, 0])

    labels = ds.disadvantaged_students(student_lsoa, areas, decile=3)

    np.testing.assert_array_equal(labels, [True, False, False, True, True])


def test_disadvantaged_students_defaults_to_the_route_decile():
    areas = pd.DataFrame(
        {"IMD Decile": [br.DISADVANTAGED_DECILE, br.DISADVANTAGED_DECILE + 1]}
    )

    np.testing.assert_array_equal(
        ds.disadvantaged_students(np.array([0, 1]), areas), [True, False]
    )


# --------------------------------------------------------------------------
# student_samples and score_sample: against the real data folder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def secondary():
    """The area and secondary school frames, loaded once for the module."""
    return ld.load_areas(), ld.load_schools()[1]


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_student_samples_draws_one_reproducible_sample_per_seed(secondary):
    areas, _ = secondary
    sizes = bp.cohort_sizes(areas, "secondary")

    first = list(ds.student_samples(areas, sizes, 2))
    second = list(ds.student_samples(areas, sizes, 2))

    assert len(first) == 2
    assert all(len(xy) == sizes.sum() for xy, _ in first)
    # Different seeds draw different students, the same seed the same ones.
    assert not np.array_equal(first[0][0], first[1][0])
    for (xy_a, lsoa_a), (xy_b, lsoa_b) in zip(first, second):
        np.testing.assert_array_equal(xy_a, xy_b)
        np.testing.assert_array_equal(lsoa_a, lsoa_b)


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_score_sample_scores_both_scenarios_of_one_sample(secondary):
    areas, schools = secondary
    routes = br.route_network(
        areas,
        schools[["Easting", "Northing"]].to_numpy(),
        schools["P8MEA"].to_numpy(),
    )
    student_xy, student_lsoa = next(
        ds.student_samples(areas, bp.cohort_sizes(areas, "secondary"), 1)
    )

    rows, school_rows, mode_rows = ds.score_sample(
        student_xy, student_lsoa, areas, schools, routes, ld.load_nts_mode_shares()
    )
    rows, intake, modes = map(pd.DataFrame, (rows, school_rows, mode_rows))

    assert list(rows["scenario"]) == ["with routes", "without routes"]
    assert rows["dissimilarity"].between(0, 1).all()
    assert (rows["n_matched"] + rows["n_unmatched"] == len(student_xy)).all()

    # The disadvantaged group is the sample's, not the matching's, so it is
    # the same in both scenarios, and those of it without a place are a
    # subset of both the group and the unmatched.
    assert rows["n_disadvantaged"].nunique() == 1
    assert 0 < rows["n_disadvantaged"].iloc[0] < len(student_xy)
    assert (rows["unassigned_disadvantaged"] <= rows["n_unmatched"]).all()
    assert (rows["unassigned_disadvantaged"] <= rows["n_disadvantaged"]).all()

    # Every school of both scenarios has its intake counted, and the counts
    # add up to the students seated in that scenario.
    assert len(intake) == 2 * len(schools)
    assert set(intake["school"]) == set(schools["EstablishmentName"])
    seated = intake.groupby("scenario")[["disadvantaged", "other"]].sum().sum(axis=1)
    pd.testing.assert_series_equal(
        seated, rows.set_index("scenario")["n_matched"], check_names=False
    )

    # Every disadvantaged student is either seated somewhere or unassigned.
    placed = intake.groupby("scenario")["disadvantaged"].sum()
    indexed = rows.set_index("scenario")
    pd.testing.assert_series_equal(
        placed + indexed["unassigned_disadvantaged"],
        indexed["n_disadvantaged"],
        check_names=False,
    )

    # Every seated student travels by exactly one expected mode, and only
    # the routed scenario seats anyone on a route.
    assert len(modes) == 2 * len(MODES)
    pd.testing.assert_series_equal(
        modes.groupby("scenario")["students"].sum(),
        rows.set_index("scenario")["n_matched"].astype(float),
        check_names=False,
    )
    on_route = modes[modes["mode"] == "route"].set_index("scenario")["students"]
    assert on_route["with routes"] > 0
    assert on_route["without routes"] == 0


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def test_plot_writes_a_png_of_the_results(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "PLOT_PNG", tmp_path / "out" / "dissimilarity.png")
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "dissimilarity": [0.30, 0.35, 0.28, 0.36],
        }
    )

    ds.plot(results)

    assert ds.PLOT_PNG.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_run_scores_both_scenarios_of_every_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "RESULTS_CSV", tmp_path / "dissimilarity.csv")
    monkeypatch.setattr(ld, "POPULATION_CACHE", tmp_path / "population_lsoa.pkl")

    results = ds.run(n_seeds=1)

    assert list(results["scenario"]) == ["with routes", "without routes"]
    assert results["dissimilarity"].between(0, 1).all()
    # Both scenarios match the same student sample.
    assert results["n_matched"].add(results["n_unmatched"]).nunique() == 1
    pd.testing.assert_frame_equal(pd.read_csv(ds.RESULTS_CSV), results)
