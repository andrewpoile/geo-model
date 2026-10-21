import pandas as pd

from geo_model.measures.travel import MODES
from geo_model.scripts import car_displacement as cd

# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


def test_run_counts_every_mode_of_both_scenarios_of_every_seed(
    areas, secondary, tmp_path
):
    results = cd.run(1, areas, secondary, tmp_path)

    assert list(results.columns) == ["seed", "scenario", "mode", "students"]
    assert len(results) == 2 * len(MODES)
    # Both scenarios seat the same students, each travelling by one mode.
    assert results.groupby("scenario")["students"].sum().round(6).nunique() == 1
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / cd.RESULTS_CSV), results)
