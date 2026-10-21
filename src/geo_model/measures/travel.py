import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from geo_model.data.travel import NTS_BANDS, NTS_MODES

MILE = 1609.344  # metres


# Miles, the lower edge of every NTS trip-length band but the first.
BAND_EDGES = np.array([1.0, 2.0, 5.0])


# Road distance per straight-line metre, applied before a trip is banded. 1 is
# no uplift, so a trip is banded on its straight-line length.
CIRCUITY = 1.0


# The NTS main modes and the route, which a routed student takes with certainty.
MODES = [*NTS_MODES.values(), "route"]


def trip_band(distance_m: ArrayLike, circuity: float = CIRCUITY) -> np.ndarray:
    """The NTS trip-length band of each trip.

    Args:
        distance_m (ArrayLike): Straight-line length of each trip, in metres.

        circuity (float, optional): Road distance per straight-line metre.
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
    student_xy: np.ndarray,
    school_xy: np.ndarray,
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

        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2), in
        the same CRS as `student_xy`.

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
    distance = np.linalg.norm(student_xy[seated] - school_xy[school], axis=1)
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
