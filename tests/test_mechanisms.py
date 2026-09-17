from dataclasses import asdict

import pandas as pd
import pytest

from geo_model import load_data as ld
from geo_model import mechanisms as mc
from geo_model.utils import MODES

DATA = ld.NTS_MODE_BY_LENGTH_ODS.parent.parent.parent

VARIANTS = [mc.BASELINE, *mc.variants()]


def synthetic(students: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frames of the shape `run` returns, two seeds of every variant."""
    results = pd.DataFrame(
        [
            {
                "seed": seed,
                "variant": variant,
                "dissimilarity": 0.3,
                "n_matched": 60,
                "n_unmatched": 0,
            }
            for seed in (0, 1)
            for variant in VARIANTS
        ]
    )
    modes = pd.DataFrame(
        [
            {"seed": seed, "variant": variant, "mode": mode, "students": students}
            for seed in (0, 1)
            for variant in VARIANTS
            for mode in MODES
        ]
    )
    return results, modes


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------


def test_variants_start_from_the_paper_mechanism_and_pull_each_lever():
    variants = mc.variants(walk_distance=500.0)

    assert next(iter(variants)) == "current"
    assert variants["current"] == mc.DEFAULTS
    for name, settings in variants.items():
        # A lever is pulled by name alone; everything else holds the default.
        assert settings.reserved == ("reserved" in name)
        assert (settings.walk_distance > 0) == ("gated" in name)
        assert settings.walk_distance in (0.0, 500.0)
        assert settings.walk_rule == ("nearest" if "nearest" in name else "routeless")
        untouched = {
            k: v
            for k, v in asdict(settings).items()
            if k not in ("walk_distance", "walk_rule", "reserved")
        }
        assert untouched == {k: asdict(mc.DEFAULTS)[k] for k in untouched}


# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_run_scores_every_variant_and_the_baseline_once(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "RESULTS_CSV", tmp_path / "mechanisms.csv")
    monkeypatch.setattr(mc, "MODES_CSV", tmp_path / "mechanisms_modes.csv")
    monkeypatch.setattr(ld, "POPULATION_CACHE", tmp_path / "population_lsoa.pkl")

    results, modes = mc.run(n_seeds=1)

    assert list(results.columns) == [
        "seed",
        "variant",
        "dissimilarity",
        "n_matched",
        "n_unmatched",
    ]
    assert list(modes.columns) == ["seed", "variant", "mode", "students"]
    assert list(results["variant"]) == VARIANTS
    assert list(modes["variant"].unique()) == VARIANTS
    assert len(modes) == len(VARIANTS) * len(MODES)
    # Every seated student travels by one expected mode, and only the
    # baseline seats nobody on a route.
    pd.testing.assert_series_equal(
        modes.groupby("variant", sort=False)["students"].sum(),
        results.set_index("variant")["n_matched"].astype(float),
        check_names=False,
    )
    on_route = modes[modes["mode"] == "route"].set_index("variant")["students"]
    assert on_route[mc.BASELINE] == 0
    assert (on_route.drop(mc.BASELINE) > 0).all()
    pd.testing.assert_frame_equal(pd.read_csv(mc.RESULTS_CSV), results)
    pd.testing.assert_frame_equal(pd.read_csv(mc.MODES_CSV), modes)


# --------------------------------------------------------------------------
# summary and plot
# --------------------------------------------------------------------------


def test_summary_tabulates_the_means_and_shares_per_variant():
    results, modes = synthetic(students=10.0)

    table = mc.summary(results, modes)

    assert list(table.index) == VARIANTS
    assert list(table.columns) == [*MODES, "car share", "active share", "dissimilarity"]
    assert (table[MODES] == 10.0).all().all()
    assert table["car share"].to_list() == pytest.approx([1 / len(MODES)] * 7)
    assert table["active share"].to_list() == pytest.approx([2 / len(MODES)] * 7)
    assert (table["dissimilarity"] == 0.3).all()


def test_plot_writes_a_png_of_the_results(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "PLOT_PNG", tmp_path / "out" / "mechanisms.png")

    mc.plot(*synthetic())

    assert mc.PLOT_PNG.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
