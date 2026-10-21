import numpy as np
import pandas as pd
import pytest

from geo_model.data.travel import NTS_BANDS
from geo_model.measures.travel import (
    MILE,
    MODES,
    expected_modes,
    mode_change,
    trip_band,
)

# --------------------------------------------------------------------------
# trip_band
# --------------------------------------------------------------------------


def test_trip_band_bins_on_the_nts_edges():
    miles = np.array([0.0, 0.99, 1.0, 1.99, 2.0, 4.99, 5.0, 40.0])

    bands = trip_band(miles * MILE)

    np.testing.assert_array_equal(
        bands, [NTS_BANDS[i] for i in (0, 0, 1, 1, 2, 2, 3, 3)]
    )


def test_trip_band_scales_by_circuity_before_binning():
    # 0.8 straight-line miles is a 1.04 mile road trip at a circuity of 1.3.
    assert trip_band(np.array([0.8 * MILE]), circuity=1.3)[0] == NTS_BANDS[1]


def test_trip_band_rejects_a_non_positive_circuity():
    with pytest.raises(ValueError, match="circuity must be positive"):
        trip_band(np.array([100.0]), circuity=0)


# --------------------------------------------------------------------------
# expected_modes and mode_change
# --------------------------------------------------------------------------


SHARES = pd.DataFrame(
    {
        "walk": [0.8, 0.5, 0.1, 0.0],
        "cycle": [0.1, 0.1, 0.1, 0.0],
        "car": [0.1, 0.3, 0.5, 0.4],
        "bus": [0.0, 0.1, 0.2, 0.5],
        "other": [0.0, 0.0, 0.1, 0.1],
    },
    index=NTS_BANDS,
)


def test_expected_modes_counts_routed_routeless_and_drops_unseated():
    school_xy = np.array([[0.0, 0.0], [10_000.0, 0.0]])
    # Student 0 walks distance 0 to school 0, student 1 rides a route to
    # school 1, student 2 is 3 miles from school 1 without a route, student 3
    # holds no seat.
    student_xy = np.array(
        [[0.0, 0.0], [0.0, 0.0], [10_000 - 3 * MILE, 0.0], [5.0, 5.0]]
    )
    matching = np.array([[0, -1], [1, 0], [1, -1], [-1, -1]], dtype=np.int32)

    modes = expected_modes(matching, student_xy, school_xy, SHARES)

    expected = SHARES.loc[[NTS_BANDS[0], NTS_BANDS[2]]].sum()
    expected["route"] = 1
    assert list(modes.index) == MODES
    assert modes.to_dict() == pytest.approx(expected.to_dict())
    assert modes.sum() == pytest.approx(3)


def test_mode_change_subtracts_the_unrouted_count_within_every_key():
    results = pd.DataFrame(
        {
            "parameter": ["capacity_scale"] * 4,
            "value": [1, 1, 1, 1],
            "seed": [0, 0, 0, 0],
            "scenario": ["with routes", "without routes"] * 2,
            "mode": ["walk", "walk", "route", "route"],
            "students": [30.0, 50.0, 20.0, 0.0],
        }
    )

    change = mode_change(results)

    assert list(change.columns) == ["parameter", "value", "seed", "mode", "change"]
    assert change.set_index("mode")["change"].to_dict() == {
        "walk": -20.0,
        "route": 20.0,
    }
