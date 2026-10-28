"""Estimate the change in travel mode that routed matching induces.

The routeless matching is taken to reproduce the existing statistics: a
student seated without a route travels by the National Travel Survey's mode
split for school trips of their length (NTS0614a, aged 11 to 16), so the
count of every mode is an expected value and the only randomness is the
student sample per seed. A student seated with a route takes it with
certainty, counted as a mode of its own, so the change in each existing mode
shows where routed students are drawn from. Trip length is the distance to
the seated school, as the sample measures it, scaled by a circuity factor for
the road distance the survey records, and an unseated student makes no trip. The
counting itself is `expected_modes`, which `analysis.scoring.score_sample`
applies to every matching it scores, so every run, sweep cell and search
candidate carries the same estimate.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from geo_model.data.travel import NTS_BANDS, NTS_MODES

MILE = 1609.344  # metres
# Miles, the lower edge of every NTS trip-length band but the first.
BAND_EDGES = np.array([1.0, 2.0, 5.0])
# Road distance per metre of trip length, applied before a trip is banded. 1 is
# no uplift, so a trip is banded on its length as measured: along the walking
# network, or in a straight line under --distance euclidean.
CIRCUITY = 1.0
# The NTS main modes and the route, which a routed student takes with certainty.
MODES = [*NTS_MODES.values(), "route"]


def trip_band(distance_m: ArrayLike, circuity: float = CIRCUITY) -> np.ndarray:
    """The NTS trip-length band of each trip.

    Args:
        distance_m (ArrayLike): Length of each trip, in metres.

        circuity (float, optional): Road distance per metre of trip length.
        Defaults to CIRCUITY.

    Returns:
        np.ndarray: The band of each trip, one of NTS_BANDS.
    """
    if circuity <= 0:
        raise ValueError(f"circuity must be positive, got {circuity}.")
    miles = np.asarray(distance_m, dtype=float) * circuity / MILE
    return np.array(NTS_BANDS)[np.digitize(miles, BAND_EDGES)]


def expected_modes(
    matching: np.ndarray,
    student_distance: np.ndarray,
    shares: pd.DataFrame,
    circuity: float = CIRCUITY,
) -> pd.Series:
    """Expected students per travel mode under a matching.

    A student seated without a route travels by the NTS mode split for school
    trips of their length, held as a probability vector rather than sampled,
    so the count is an expected value. A student seated with a route takes it
    with certainty. An unseated student makes no trip and is left out.

    Args:
        matching (np.ndarray): (school, route) per student, -1 when absent, as
        returned by `fast_DAT`.

        student_distance (np.ndarray): Distance from every student to every
        school, shape (n_students, n_schools), in metres.

        shares (pd.DataFrame): Mode share within each band, as returned by
        `load_nts_mode_shares`.

        circuity (float, optional): Passed to `trip_band`. Defaults to
        CIRCUITY.

    Returns:
        pd.Series: Expected students per mode, indexed by MODES, summing to
        the number seated.
    """
    seated = matching[:, 0] >= 0
    school, route = matching[seated, 0], matching[seated, 1]
    distance = student_distance[seated, school]
    band = trip_band(distance, circuity)

    modes = shares.loc[band[route < 0]].sum()
    modes["route"] = int((route >= 0).sum())
    return modes.reindex(MODES)


def mode_change(results: pd.DataFrame) -> pd.DataFrame:
    """The change in expected students per mode the routes induce, per seed.

    Args:
        results (pd.DataFrame): As returned by `run`, or any frame carrying
        "seed", "scenario", "mode" and "students" plus other key columns.

    Returns:
        pd.DataFrame: One row per (seed, mode) and any other key columns,
        with "change" as the students with routes minus those without.
    """
    keys = [c for c in results.columns if c not in ("scenario", "students")]
    change = results.pivot(index=keys, columns="scenario", values="students")
    change = change["with routes"] - change["without routes"]
    return change.rename("change").reset_index()
