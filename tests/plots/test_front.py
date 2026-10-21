import numpy as np
import pandas as pd
import pytest

from geo_model.analysis.search import DEFAULT_OBJECTIVES
from geo_model.plots.front import plot_front

# --------------------------------------------------------------------------
# plot_front
# --------------------------------------------------------------------------


def test_plot_front_writes_a_png_of_the_front(tmp_path):
    png = tmp_path / "out" / "front.png"
    evaluations = pd.DataFrame(
        {
            "generation": [1, 1, 1, 2],
            "min_distance": [1000.0, 2000.0, 9000.0, 3000.0],
            "dissimilarity": [0.30, 0.35, np.nan, 0.40],
            "car_displacement": [5.0, 12.0, np.nan, 8.0],
            "feasible": [True, True, False, True],
        }
    )
    front = evaluations.iloc[[0, 1]].drop(columns="feasible")

    plot_front(
        evaluations,
        front,
        front.iloc[1],
        {"dissimilarity": 0.33, "car_displacement": 9.0},
        DEFAULT_OBJECTIVES,
        png,
    )

    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_front_rejects_anything_but_two_objectives(tmp_path):
    with pytest.raises(ValueError, match="plotted in two objectives"):
        plot_front(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(),
            {},
            ["dissimilarity"],
            tmp_path / "front.png",
        )
