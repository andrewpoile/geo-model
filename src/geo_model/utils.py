import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import ttest_rel

from geo_model.data.travel import NTS_BANDS, NTS_MODES

MILE = 1609.344  # metres
# Miles, the lower edge of every NTS trip-length band but the first.
BAND_EDGES = np.array([1.0, 2.0, 5.0])
# Road distance per straight-line metre, applied before a trip is banded. 1 is
# no uplift, so a trip is banded on its straight-line length.
CIRCUITY = 1.0
# The NTS main modes and the route, which a routed student takes with certainty.
MODES = [*NTS_MODES.values(), "route"]


def school_intake(
    matched_school: ArrayLike,
    disadvantaged: ArrayLike,
    n_schools: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count the disadvantaged and the other students seated at each school.

    A student without a seat belongs to no school's intake, so unmatched
    students are left out of both counts.

    Args:
        matched_school (ArrayLike): School each student is matched to, shape
        (n_students,), -1 for an unmatched student. The first column of the
        matching `fast_DAT` returns.

        disadvantaged (ArrayLike): Whether each student is disadvantaged,
        shape (n_students,).

        n_schools (int): Number of schools the matching indexes.

    Returns:
        tuple[np.ndarray, np.ndarray]: Disadvantaged and other students
        seated at each school, each of shape (n_schools,).
    """
    matched_school = np.asarray(matched_school, dtype=np.int64)
    disadvantaged = np.asarray(disadvantaged, dtype=bool)
    if matched_school.shape != disadvantaged.shape:
        raise ValueError(
            f"matched_school and disadvantaged must align: got "
            f"{matched_school.shape} and {disadvantaged.shape}."
        )
    if (matched_school >= n_schools).any():
        raise ValueError(
            f"matched_school holds indices beyond the {n_schools} schools."
        )

    seated = matched_school >= 0
    return (
        np.bincount(matched_school[seated & disadvantaged], minlength=n_schools),
        np.bincount(matched_school[seated & ~disadvantaged], minlength=n_schools),
    )


def local_intake(
    preferences: np.ndarray,
    matching: np.ndarray,
    student_district: ArrayLike,
    school_district: ArrayLike,
    disadvantaged: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count each school's local applicants, and the disadvantaged and the
    other students among them it seats.

    A student is local to a school sitting in their own district, the
    students the top bracket of its priorities holds without a route. A
    student applied to a school if they proposed any bundle of it, and
    Deferred Acceptance has every student propose down their list: to every
    bundle up to the one they hold, or to every bundle on it if they hold
    none.

    Args:
        preferences (np.ndarray): Student preferences, as `rank_bundles`
        returns them, shape (n_students, max_options, 2).

        matching (np.ndarray): The matching `fast_DAT` returns of those
        preferences, shape (n_students, 2).

        student_district (ArrayLike): Positional index of the district each
        student lives in, shape (n_students,).

        school_district (ArrayLike): Positional index of the district each
        school sits in, as `district_index` gives it, shape (n_schools,).

        disadvantaged (ArrayLike): Whether each student is disadvantaged,
        shape (n_students,).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Local students who applied
        to each school, and the disadvantaged and the other of them it seats,
        each of shape (n_schools,).
    """
    student_district = np.asarray(student_district, dtype=np.int64)
    school_district = np.asarray(school_district, dtype=np.int64)
    disadvantaged = np.asarray(disadvantaged, dtype=bool)
    n_students, max_options, _ = preferences.shape
    n_schools = len(school_district)
    if not (
        matching.shape == (n_students, 2)
        and student_district.shape == disadvantaged.shape == (n_students,)
    ):
        raise ValueError(
            f"preferences, matching, student_district and disadvantaged must "
            f"hold one row per student: got {preferences.shape}, "
            f"{matching.shape}, {student_district.shape} and "
            f"{disadvantaged.shape}."
        )

    matched = matching[:, 0] >= 0
    held = (preferences == matching[:, None, :]).all(axis=2)
    if not held[matched].any(axis=1).all():
        raise ValueError("A student holds a bundle missing from their own list.")
    last = np.where(matched, held.argmax(axis=1), max_options - 1)
    proposed = (np.arange(max_options) <= last[:, None]) & (preferences[:, :, 0] >= 0)

    applied = np.zeros((n_students, n_schools), dtype=bool)
    student, option = np.nonzero(proposed)
    applied[student, preferences[student, option, 0]] = True
    applied &= student_district[:, None] == school_district
    seated = applied & (matching[:, [0]] == np.arange(n_schools))
    return (
        applied.sum(axis=0),
        (seated & disadvantaged[:, None]).sum(axis=0),
        (seated & ~disadvantaged[:, None]).sum(axis=0),
    )


def dissimilarity_terms(group_a: ArrayLike, group_b: ArrayLike) -> np.ndarray:
    """Each school's term of the dissimilarity index.

    With a_c and b_c students of each group at school c, out of A and B in
    all, the term is a_c / A - b_c / B: positive where the school holds more
    than its share of group a, negative where it holds less. The terms sum to
    0, and half the sum of their absolute values is the index.

    Args:
        group_a (ArrayLike): Students of the first group at each school,
        shape (n_schools,).

        group_b (ArrayLike): Students of the second group at each school,
        shape (n_schools,).

    Returns:
        np.ndarray: The term of each school, shape (n_schools,).
    """
    group_a = np.asarray(group_a, dtype=np.float64)
    group_b = np.asarray(group_b, dtype=np.float64)
    if group_a.shape != group_b.shape:
        raise ValueError(
            f"group_a and group_b must align: got {group_a.shape} and {group_b.shape}."
        )
    if group_a.sum() == 0 or group_b.sum() == 0:
        raise ValueError(
            f"{group_a.sum():g} and {group_b.sum():g} students of the two groups "
            "hold a seat, so the index is undefined."
        )
    return group_a / group_a.sum() - group_b / group_b.sum()


def district_groups(
    areas: pd.DataFrame, student_lsoa: np.ndarray, disadvantaged: np.ndarray
) -> pd.DataFrame:
    """Each district's disadvantaged and advantaged students, and how the two
    groups compare there and against the region.

    Args:
        areas (pd.DataFrame): Districts carrying "LSOA21CD", "LA (name)",
        "IDACI Decile" and "IDACI Score", in the order students were sampled
        from.

        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        disadvantaged (np.ndarray): Whether each student is disadvantaged, as
        `disadvantaged_group` gives it.

    Returns:
        pd.DataFrame: One row per district, in the order of `areas`, with its
        "LSOA21CD", "LA (name)", "IDACI Decile" and "IDACI Score", its
        "cohort" of students, its "disadvantaged" and "advantaged" students,
        the "disadvantaged_share" of its cohort, its "share_of_disadvantaged"
        and "share_of_advantaged" of the region's, and the "difference" of
        those two, the district's term of the dissimilarity index as
        `dissimilarity_terms` gives it. A district with no student has no
        disadvantaged share, so it is named and left NaN.
    """
    # Students are counted by district as a school's intake is by school.
    group_a, group_b = school_intake(student_lsoa, disadvantaged, len(areas))
    cohort = group_a + group_b
    if (cohort == 0).any():
        print(
            "No student sampled, so no disadvantaged share: "
            + ", ".join(areas["LSOA21CD"].to_numpy()[cohort == 0])
        )
    return areas[["LSOA21CD", "LA (name)", "IDACI Decile", "IDACI Score"]].assign(
        cohort=cohort,
        disadvantaged=group_a,
        advantaged=group_b,
        disadvantaged_share=np.divide(
            group_a, cohort, out=np.full(len(cohort), np.nan), where=cohort > 0
        ),
        share_of_disadvantaged=group_a / group_a.sum(),
        share_of_advantaged=group_b / group_b.sum(),
        difference=dissimilarity_terms(group_a, group_b),
    )


def dissimilarity_index(
    matched_school: ArrayLike,
    disadvantaged: ArrayLike,
    n_schools: int,
) -> float:
    """Dissimilarity index of a matching, disadvantaged students against the rest.

    With a_c disadvantaged and b_c other students seated at school c, out of
    A and B seated in all, the index is

        D = 1/2 * sum_c | a_c / A - b_c / B |

    the share of either group that would have to change school for every
    school to hold the region-wide mix. 0 is that mix everywhere, 1 is complete
    segregation. The counts are those of `school_intake`, so unmatched
    students are left out of both groups.

    Args:
        matched_school (ArrayLike): School each student is matched to, shape
        (n_students,), -1 for an unmatched student. The first column of the
        matching `fast_DAT` returns.

        disadvantaged (ArrayLike): Whether each student is disadvantaged,
        shape (n_students,).

        n_schools (int): Number of schools the matching indexes.

    Returns:
        float: The index, in [0, 1].
    """
    terms = dissimilarity_terms(
        *school_intake(matched_school, disadvantaged, n_schools)
    )
    return 0.5 * float(np.abs(terms).sum())


def lorenz_curve(
    group_a: ArrayLike, group_b: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """The Lorenz curve of the first group against every student, over schools.

    Schools are ordered by the share of their intake in group a, lowest first,
    and the curve joins (0, 0) to the point after each school: the cumulative
    share of all students seated against that of group a's. A school nobody
    is seated at adds no length to the curve, so it is left out.

    The curve is taken against every student rather than against group b
    alone, so with P the share of students in group a, its largest gap from
    the diagonal is (1 - P) times the dissimilarity index.

    Args:
        group_a (ArrayLike): Students of the first group at each school,
        shape (n_schools,).

        group_b (ArrayLike): Students of the second group at each school,
        shape (n_schools,).

    Returns:
        tuple[np.ndarray, np.ndarray]: The cumulative shares of all students
        and of group a, each running from 0 to 1, one point per school seating
        a student after the origin.
    """
    group_a = np.asarray(group_a, dtype=np.float64)
    group_b = np.asarray(group_b, dtype=np.float64)
    if group_a.shape != group_b.shape:
        raise ValueError(
            f"group_a and group_b must align: got {group_a.shape} and {group_b.shape}."
        )
    if group_a.sum() == 0:
        raise ValueError(
            "No student of the first group holds a seat, so the curve is undefined."
        )
    total = group_a + group_b
    seated = total > 0
    group_a, total = group_a[seated], total[seated]
    order = np.argsort(group_a / total, kind="stable")
    x = np.concatenate(([0.0], np.cumsum(total[order]) / total.sum()))
    y = np.concatenate(([0.0], np.cumsum(group_a[order]) / group_a.sum()))
    return x, y


def gini(x: np.ndarray, y: np.ndarray) -> float:
    """Gini coefficient of a Lorenz curve, as `lorenz_curve` returns it.

    One less twice the area under the curve, taken by the trapezoid rule,
    which is exact for a curve joining its points by straight lines. 0 is
    every school holding the region-wide mix. Against every student, as
    `lorenz_curve` draws it, complete segregation reaches 1 - P rather than
    1, with P the share of students in the first group.

    Args:
        x (np.ndarray): Cumulative share of all students, from 0 to 1.

        y (np.ndarray): Cumulative share of the first group, from 0 to 1.

    Returns:
        float: The coefficient.
    """
    return 1 - 2 * float(np.trapezoid(y, x))


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


def routes_t_test(results: pd.DataFrame) -> pd.Series:
    """Paired t-test of the dissimilarity index with routes against without.

    Both scenarios of a seed match the same students, so the indices pair by
    seed and the test is taken over each seed's difference, with routes less
    without. It is undefined, t and p NaN, with fewer than two seeds, and
    with more where every seed's difference is 0, the only case in which
    `ttest_rel` returns NaN on finite indices.

    Args:
        results (pd.DataFrame): The scenario rows of one setting, carrying
        "seed", "scenario" and "dissimilarity".

    Returns:
        pd.Series: "seeds" (the pairs tested), "difference" (their mean
        difference), "t" and "p" (two-sided).

    Raises:
        ValueError: If a seed holds more than one row of a scenario, or a
        row of one scenario alone.
    """
    paired = results.pivot(index="seed", columns="scenario", values="dissimilarity")
    paired = paired[["with routes", "without routes"]]
    if paired.isna().any(axis=None):
        raise ValueError(
            "Seeds scored in one scenario alone have no pair to test: "
            + ", ".join(str(seed) for seed in paired.index[paired.isna().any(axis=1)])
        )
    if len(paired) < 2:
        t = p = np.nan
    else:
        test = ttest_rel(paired["with routes"], paired["without routes"])
        t, p = float(test.statistic), float(test.pvalue)
    return pd.Series(
        {
            "seeds": len(paired),
            "difference": (paired["with routes"] - paired["without routes"]).mean(),
            "t": t,
            "p": p,
        }
    )


def holm(p: pd.Series) -> pd.Series:
    """Holm's step-down adjustment of the p-values `p` for testing them all.

    With the m p-values defined in ascending order, the i-th is adjusted to
    the largest of (m - j + 1) p_j over j <= i, capped at 1. A NaN p-value,
    a test left undefined, stays NaN and is not counted in m.

    Args:
        p (pd.Series): P-values, uniquely indexed.

    Returns:
        pd.Series: The adjusted p-values, indexed as `p`.
    """
    defined = p.dropna().sort_values()
    adjusted = (defined * np.arange(len(defined), 0, -1)).cummax().clip(upper=1)
    return adjusted.reindex(p.index)
