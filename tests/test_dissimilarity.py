import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from geo_model import build_prefs as bp
from geo_model import build_routes as br
from geo_model import dissimilarity as ds
from geo_model import load_data as ld
from geo_model.utils import MODES, sample_students

DATA = ld.POPULATION_XLSX.parent.parent


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
    assert all(len(xy) == sizes.sum() for xy, _, _ in first)
    # Different seeds draw different students, the same seed the same ones.
    assert not np.array_equal(first[0][0], first[1][0])
    for sample_a, sample_b in zip(first, second):
        for a, b in zip(sample_a, sample_b):
            np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_student_samples_draw_each_district_its_idaci_share_after_the_locations(
    secondary,
):
    areas, _ = secondary
    sizes = bp.cohort_sizes(areas, "secondary")

    student_xy, student_lsoa, drawn = next(ds.student_samples(areas, sizes, 1))

    np.testing.assert_array_equal(
        np.bincount(student_lsoa[drawn], minlength=len(areas)),
        br.disadvantaged_cohort(areas, sizes),
    )
    # The locations are the ones a draw of locations alone takes from the
    # seed's stream, so drawing the group after them leaves them as they were.
    stream = np.random.SeedSequence(ds.SEED).spawn(1)[0]
    alone, _ = sample_students(
        areas["Borders"], areas["Centroids"], sizes, np.random.default_rng(stream)
    )
    np.testing.assert_array_equal(student_xy, alone)


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_score_sample_scores_both_scenarios_of_one_sample(secondary):
    areas, schools = secondary
    sizes = bp.cohort_sizes(areas, "secondary")
    routes = br.route_network(
        areas,
        schools[["Easting", "Northing"]].to_numpy(),
        schools["P8MEA"].to_numpy(),
        schools["PlacesOffered"].to_numpy(),
        sizes,
    )
    student_xy, student_lsoa, disadvantaged = next(ds.student_samples(areas, sizes, 1))

    rows, school_rows, mode_rows, route_rows = ds.score_sample(
        student_xy,
        student_lsoa,
        areas,
        schools,
        routes,
        disadvantaged,
        ld.load_nts_mode_shares(),
    )
    rows, intake, modes, riders = map(
        pd.DataFrame, (rows, school_rows, mode_rows, route_rows)
    )

    assert list(rows["scenario"]) == ["with routes", "without routes"]
    assert rows["dissimilarity"].between(0, 1).all()
    assert (rows["n_matched"] + rows["n_unmatched"] == len(student_xy)).all()

    # The disadvantaged group is the sample's, not the matching's, so it is
    # the same in both scenarios, and those of it without a place are a
    # subset of both the group and the unmatched.
    assert (rows["n_disadvantaged"] == disadvantaged.sum()).all()
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

    # The schools' terms are the index taken apart: they cancel out, and
    # half their absolute sum is the scenario's index.
    terms = intake.groupby("scenario")["dissimilarity_term"]
    assert terms.sum().abs().max() < 1e-12
    pd.testing.assert_series_equal(
        0.5 * terms.apply(lambda t: t.abs().sum()),
        indexed["dissimilarity"],
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

    # Every route of the routed matching has its riders counted, none more
    # than its seats, and together they are the students seated on a route.
    pd.testing.assert_series_equal(
        riders["route"], routes["route_id"], check_names=False, check_dtype=False
    )
    np.testing.assert_array_equal(riders["capacity"], routes["capacity"])
    used = riders["disadvantaged"] + riders["other"]
    assert (used <= riders["capacity"]).all()
    assert used.sum() == on_route["with routes"]
    assert (
        riders["disadvantaged"].sum()
        <= intake.groupby("scenario")["disadvantaged"].sum()["with routes"]
    )


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def test_plot_writes_a_png_of_the_results(tmp_path):
    png = tmp_path / "out" / "dissimilarity.png"
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "dissimilarity": [0.30, 0.35, 0.28, 0.36],
        }
    )

    ds.plot(results, png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_fits_the_y_axis_to_the_indices(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(
        Figure, "savefig", lambda fig, *args, **kwargs: saved.append(fig)
    )
    results = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "scenario": ["with routes", "without routes"] * 2,
            "dissimilarity": [0.30, 0.35, 0.28, 0.36],
        }
    )

    ds.plot(results, tmp_path / "dissimilarity.png")

    low, high = saved[0].axes[0].get_ylim()
    # Every index is in view, on an axis far narrower than the index's 0 to 1.
    assert 0 < low <= 0.28 and 0.36 <= high
    assert high - low < 0.2


def test_plot_draws_one_seed_with_its_test_undefined(tmp_path):
    png = tmp_path / "dissimilarity.png"
    results = pd.DataFrame(
        {
            "seed": [0, 0],
            "scenario": ["with routes", "without routes"],
            "dissimilarity": [0.30, 0.35],
        }
    )

    ds.plot(results, png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# t_test_label
# --------------------------------------------------------------------------


def test_t_test_label_gives_the_seeds_difference_t_and_p():
    test = pd.Series({"seeds": 30, "difference": -0.0123, "t": -4.214, "p": 0.00023})

    assert ds.t_test_label(test) == (
        "Paired t-test over 30 seeds\nmean difference -0.0123, t = -4.21, p = 0.00023"
    )


def test_t_test_label_says_why_a_test_is_undefined():
    one_seed = pd.Series({"seeds": 1, "difference": -0.05, "t": np.nan, "p": np.nan})
    unchanged = pd.Series({"seeds": 5, "difference": 0.0, "t": np.nan, "p": np.nan})

    assert ds.t_test_label(one_seed) == (
        "Paired t-test over 1 seed\nundefined: one seed"
    )
    assert ds.t_test_label(unchanged).endswith(
        "undefined: routes change no seed's index"
    )


# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_run_scores_both_scenarios_of_every_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "POPULATION_CACHE", tmp_path / "population_lsoa.pkl")
    _, schools = ld.load_schools()

    results = ds.run(1, ld.load_areas(), schools, tmp_path)

    assert list(results["scenario"]) == ["with routes", "without routes"]
    assert results["dissimilarity"].between(0, 1).all()
    # Both scenarios match the same student sample.
    assert results["n_matched"].add(results["n_unmatched"]).nunique() == 1
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / ds.RESULTS_CSV), results)
