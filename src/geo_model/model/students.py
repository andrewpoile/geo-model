from collections.abc import Iterator

import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from geo_model.data.areas import PHASE_AGE
from geo_model.data.network import Distance
from geo_model.model.settings import DISADVANTAGE_CHOICES, SEED

# A student sample: the coordinates of every student, the positional index of
# the district each was drawn in, whether each was drawn disadvantaged, the
# seed their tastes for the schools are drawn from, and the distance from every
# student to every school, at single precision, since every sample of a run is
# held at once, in the parent and in every worker.
Sample = tuple[np.ndarray, np.ndarray, np.ndarray, np.random.SeedSequence, np.ndarray]


def cohort_sizes(areas: pd.DataFrame, phase: str) -> np.ndarray:
    """Expected students in each area's cohort for a phase, the mean over the
    years. Route seats are sized on it, so one route set serves every sample.

    Args:
        areas (pd.DataFrame): Districts carrying the "Age 4 Mean" and "Age 11
        Mean" columns `load_areas` gives them.

        phase (str): "primary" takes the four-year-olds, "secondary" the
        eleven-year-olds.

    Returns:
        np.ndarray: One mean cohort per area, aligned with `areas`.
    """
    return areas[f"Age {PHASE_AGE[phase]} Mean"].to_numpy(dtype=float)


def draw_cohort_sizes(
    areas: pd.DataFrame, phase: str, rng: np.random.Generator
) -> np.ndarray:
    """Students to sample in each area for a phase, drawn from a normal of
    its cohort's mean and standard deviation over the years.

    Each draw is rounded to the nearest student, and a negative draw places
    none.

    Args:
        areas (pd.DataFrame): Districts carrying the "Age 4 Mean", "Age 4 SD",
        "Age 11 Mean" and "Age 11 SD" columns `load_areas` gives them.

        phase (str): "primary" takes the four-year-olds, "secondary" the
        eleven-year-olds.

        rng (np.random.Generator): Source of randomness.

    Returns:
        np.ndarray: One count per area, aligned with `areas`.
    """
    age = PHASE_AGE[phase]
    draw = rng.normal(
        areas[f"Age {age} Mean"].to_numpy(), areas[f"Age {age} SD"].to_numpy()
    )
    return np.maximum(np.rint(draw), 0).astype(np.int64)


def sample_in_polygon(
    geom: BaseGeometry,
    centre: tuple[float, float],
    size: int,
    rng: np.random.Generator,
    oversample: int = 6,
) -> np.ndarray:
    """Sample points from a bivariate normal, truncated to a polygon.

    Points are drawn i.i.d. per axis from a normal centred on `centre` with
    standard deviation equal to half the larger side of the polygon's bounding
    box, and rejected unless they fall strictly inside `geom`.

    Candidates are drawn and tested in batches rather than one at a time. Only
    ~17% of candidates are accepted for a typical LSOA, so `oversample` batches
    enough candidates to satisfy the request in a single pass.

    Args:
        geom (BaseGeometry): Polygon the points must fall inside.

        centre (tuple[float, float]): Centre of the sampling distribution, in
        the same CRS as `geom`.

        size (int): Number of points to return.

        rng (np.random.Generator): Source of randomness.

        oversample (int, optional): Candidates drawn per point still needed.
        Defaults to 6.

    Returns:
        np.ndarray: Array of shape (size, 2) of accepted coordinates.
    """
    if geom.is_empty or size == 0:
        return np.empty((0, 2))

    shapely.prepare(geom)  # build the GEOS index once, not once per candidate
    xmin, ymin, xmax, ymax = shapely.bounds(geom)
    sd = max((xmax - xmin) / 2, (ymax - ymin) / 2)

    accepted, needed = [], size
    for _ in range(100):
        candidates = rng.normal(size=(max(needed * oversample, 64), 2)) * sd + centre
        inside = candidates[
            shapely.contains_xy(geom, candidates[:, 0], candidates[:, 1])
        ]
        accepted.append(inside)
        needed -= len(inside)
        if needed <= 0:
            return np.vstack(accepted)[:size]

    raise RuntimeError(
        f"Rejection sampling failed to place {size} points in polygon with bounds "
        f"{(xmin, ymin, xmax, ymax)} after 100 batches; {needed} still needed. "
        f"The polygon may be degenerate or `centre` may lie far outside it."
    )


def sample_students(
    borders: pd.Series,
    centroids: pd.Series,
    sizes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample student locations for every area.

    Args:
        borders (pd.Series): Area polygons.

        centroids (pd.Series): Population-weighted centre of each area,
        aligned with `borders`.

        sizes (np.ndarray): Number of students to place in each area, aligned
        with `borders`.

        rng (np.random.Generator): Source of randomness.

    Returns:
        tuple[np.ndarray, np.ndarray]: Coordinates of shape (n_students, 2),
        and the positional index of the area each student was drawn in.
    """
    centre_x = shapely.get_x(np.asarray(centroids))
    centre_y = shapely.get_y(np.asarray(centroids))
    sizes = np.asarray(sizes, dtype=np.int64)

    if not (len(borders) == len(centre_x) == len(sizes)):
        raise ValueError(
            f"borders, centroids and sizes must align: got {len(borders)}, "
            f"{len(centre_x)} and {len(sizes)}."
        )

    chunks = [
        sample_in_polygon(geom, (cx, cy), int(n), rng)
        for geom, cx, cy, n in zip(borders, centre_x, centre_y, sizes)
    ]
    student_xy = np.vstack(chunks) if chunks else np.empty((0, 2))
    area_index = np.repeat(np.arange(len(sizes)), [len(c) for c in chunks])
    return student_xy, area_index


def disadvantaged_cohort(areas: pd.DataFrame, sizes: np.ndarray) -> np.ndarray:
    """Disadvantaged students in each area's cohort, to the nearest student.

    An area's IDACI score is the share of its children, aged 0 to 15, living
    in income-deprived families, so it is taken as the share of the cohort
    sampled there who are disadvantaged.

    Args:
        areas (pd.DataFrame): Areas carrying an "IDACI Score" column.

        sizes (np.ndarray): Students in each area's cohort, aligned with
        `areas`.

    Returns:
        np.ndarray: Disadvantaged students in each area, aligned with `areas`.
    """
    return np.rint(areas["IDACI Score"].to_numpy() * sizes).astype(np.int64)


def disadvantaged_students(
    student_lsoa: np.ndarray, areas: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Draw which students are disadvantaged, each area in its IDACI share.

    Of the students sampled in each area, `disadvantaged_cohort` of them,
    picked at random, are disadvantaged, so every area holds its IDACI score's
    share of disadvantaged students to within half a student.

    Args:
        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        areas (pd.DataFrame): Districts carrying an "IDACI Score" column, in
        the order students were sampled from.

        rng (np.random.Generator): Source of randomness.

    Returns:
        np.ndarray: Boolean, one per student.
    """
    counts = disadvantaged_cohort(
        areas, np.bincount(student_lsoa, minlength=len(areas))
    )
    disadvantaged = np.zeros(len(student_lsoa), dtype=bool)
    for area in np.flatnonzero(counts):
        students = np.flatnonzero(student_lsoa == area)
        disadvantaged[rng.choice(students, counts[area], replace=False)] = True
    return disadvantaged


def disadvantaged_group(
    student_lsoa: np.ndarray,
    drawn: np.ndarray,
    areas: pd.DataFrame,
    decile: int,
    disadvantage: str,
) -> np.ndarray:
    """Whether each student is disadvantaged under `disadvantage`.

    Args:
        student_lsoa (np.ndarray): Positional index into `areas` of the
        district each student was sampled in.

        drawn (np.ndarray): Whether each student was drawn disadvantaged, as
        `disadvantaged_students` draws it.

        areas (pd.DataFrame): Districts carrying an "IDACI Decile" column, in
        the order students were sampled from.

        decile (int): Under "decile", every student of a district at or below
        this IDACI decile is disadvantaged.

        disadvantage (str): One of DISADVANTAGE_CHOICES.

    Returns:
        np.ndarray: Boolean, one per student: `drawn`, or under "decile"
        whether the student's district sits at or below `decile`.
    """
    if disadvantage not in DISADVANTAGE_CHOICES:
        raise ValueError(
            f"disadvantage must be one of {DISADVANTAGE_CHOICES}, got {disadvantage!r}."
        )
    if disadvantage == "decile":
        return areas["IDACI Decile"].to_numpy()[student_lsoa] <= decile
    return drawn


def student_samples(
    areas: pd.DataFrame, n_seeds: int, to_school: Distance
) -> Iterator[Sample]:
    """Draw one secondary student sample per seed, reproducibly from SEED.

    Each seed draws every area's cohort, as `draw_cohort_sizes` does, then
    where its students live, then which of them are disadvantaged, all from
    the same stream, so the locations of a seed do not depend on the group.
    The students' tastes are drawn from a seed spawned off the stream, so the
    draws above are those of a stream without one.

    Args:
        areas (pd.DataFrame): Districts carrying "Borders", "Centroids",
        "IDACI Score" and the columns `draw_cohort_sizes` reads.

        n_seeds (int): Number of samples to draw.

        to_school (Distance): Distance from any point to every school, as
        `school_distance` measures it, taken once per sample so every
        matching of the sample ranks on the same distances.

    Yields:
        Sample: Coordinates and positional district index of each student, as
        returned by `sample_students`, whether each is disadvantaged, as
        drawn by `disadvantaged_students`, the seed of their tastes and their
        distance to every school.
    """
    for stream in np.random.SeedSequence(SEED).spawn(n_seeds):
        rng = np.random.default_rng(stream)
        student_xy, student_lsoa = sample_students(
            areas["Borders"],
            areas["Centroids"],
            draw_cohort_sizes(areas, "secondary", rng),
            rng,
        )
        drawn = disadvantaged_students(student_lsoa, areas, rng)
        yield (
            student_xy,
            student_lsoa,
            drawn,
            stream.spawn(1)[0],
            to_school(student_xy).astype(np.float32),
        )
