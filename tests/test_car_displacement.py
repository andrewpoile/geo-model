import pandas as pd

from geo_model import car_displacement as cd
from geo_model.utils import MODES

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


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------


def test_plot_writes_a_png_of_the_results(tmp_path):
    png = tmp_path / "out" / "car_displacement.png"
    results = pd.DataFrame(
        [
            {"seed": seed, "scenario": scenario, "mode": mode, "students": 10.0}
            for seed in (0, 1)
            for scenario in ("with routes", "without routes")
            for mode in MODES
        ]
    )

    cd.plot(results, png)

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
