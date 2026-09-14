import numpy as np
import pandas as pd
import pytest

from geo_model import build_routes as br
from geo_model import dissimilarity as ds
from geo_model import load_data as ld

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
