import pandas as pd
import pytest

from geo_model import car_displacement as cd
from geo_model import load_data as ld
from geo_model.utils import MODES

DATA = ld.NTS_MODE_BY_LENGTH_ODS.parent.parent.parent


# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_run_counts_every_mode_of_both_scenarios_of_every_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "RESULTS_CSV", tmp_path / "car_displacement.csv")
    monkeypatch.setattr(ld, "POPULATION_CACHE", tmp_path / "population_lsoa.pkl")

    results = cd.run(n_seeds=1)

    assert list(results.columns) == ["seed", "scenario", "mode", "students"]
    assert len(results) == 2 * len(MODES)
    # Both scenarios seat the same students, each travelling by one mode.
    assert results.groupby("scenario")["students"].sum().round(6).nunique() == 1
    pd.testing.assert_frame_equal(pd.read_csv(cd.RESULTS_CSV), results)


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def test_plot_writes_a_png_of_the_results(tmp_path, monkeypatch):
    monkeypatch.setattr(cd, "PLOT_PNG", tmp_path / "out" / "car_displacement.png")
    results = pd.DataFrame(
        [
            {"seed": seed, "scenario": scenario, "mode": mode, "students": 10.0}
            for seed in (0, 1)
            for scenario in ("with routes", "without routes")
            for mode in MODES
        ]
    )

    cd.plot(results)

    assert cd.PLOT_PNG.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
