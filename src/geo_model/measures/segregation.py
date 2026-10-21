import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


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


def fsm_share(secondary_schools: pd.DataFrame) -> pd.Series:
    """The share of each school's pupils eligible for free school meals.

    The register's own measure of a school's disadvantage, so the point of
    comparison for the share the model seats. The two are not the same
    measure: the model draws students disadvantaged in proportion to their
    district's IDACI score, or under DISADVANTAGE "decile" counts them by
    their district's decile, while the register counts pupils by their own
    eligibility.

    Args:
        secondary_schools (pd.DataFrame): Schools carrying
        "EstablishmentName" and "PercentageFSM".

    Returns:
        pd.Series: The share, in [0, 1], indexed by establishment name. A
        school the register holds no figure for is named and left NaN.
    """
    fsm = secondary_schools.set_index("EstablishmentName")["PercentageFSM"] / 100
    if fsm.isna().any():
        print(
            "No FSM percentage recorded, so drawn blank: "
            + ", ".join(fsm.index[fsm.isna()])
        )
    return fsm


def fsm_term(secondary_schools: pd.DataFrame) -> pd.Series:
    """Each school's term of the dissimilarity index, pupils eligible for free
    school meals against the rest, as the register counts them.

    The register's own measure of how a school's intake departs from the
    region's, on the scale of the terms the model seats, though not the same
    measure: the model draws students disadvantaged in proportion to their
    district's IDACI score, or under DISADVANTAGE "decile" counts them by
    their district's decile, while the register counts pupils by their own
    eligibility. The
    register takes "PercentageFSM" over fewer pupils than "NumberOfPupils"
    at schools with a sixth form, so the pupils it was taken over are
    recovered as FSM * 100 / PercentageFSM, to within the percentage's
    rounding.

    Args:
        secondary_schools (pd.DataFrame): Schools carrying
        "EstablishmentName", "FSM" and "PercentageFSM".

    Returns:
        pd.Series: The term, indexed by establishment name. A school the
        register holds no figures for is named, left NaN and left out of the
        totals the other terms are taken over.

    Raises:
        ValueError: If a school records no pupil eligible, from which the
        pupils the percentage was taken over cannot be recovered.
    """
    fsm = secondary_schools.set_index("EstablishmentName")[["FSM", "PercentageFSM"]]
    missing = fsm.isna().any(axis=1)
    if missing.any():
        print(
            "No FSM count recorded, so drawn blank and left out of the FSM "
            "totals: " + ", ".join(fsm.index[missing])
        )
    held = fsm[~missing]
    if (held["PercentageFSM"] == 0).any():
        raise ValueError(
            "No pupil recorded eligible, so the pupils the percentage was taken "
            "over cannot be recovered: "
            + ", ".join(held.index[held["PercentageFSM"] == 0])
        )
    assessed = held["FSM"] * 100 / held["PercentageFSM"]
    terms = dissimilarity_terms(held["FSM"], assessed - held["FSM"])
    return pd.Series(terms, index=held.index).reindex(fsm.index)
