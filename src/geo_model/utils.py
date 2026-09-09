import numpy as np
import geopandas as gpd
import shapely
from scipy.spatial import distance as spdist


def sample_in_polygon(
    geom: shapely.Geometry,
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
        geom (shapely.Geometry): Polygon the points must fall inside.

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
    borders: gpd.GeoSeries,
    centroids: gpd.GeoSeries,
    sizes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample student locations for every area.

    Args:
        borders (gpd.GeoSeries): Area polygons.

        centroids (gpd.GeoSeries): Population-weighted centre of each area,
        aligned with `borders`.

        sizes (np.ndarray): Number of students to place in each area, aligned
        with `borders`.

        rng (np.random.Generator): Source of randomness.

    Returns:
        tuple[np.ndarray, np.ndarray]: Coordinates of shape (n_students, 2),
        and the positional index of the area each student was drawn in.
    """
    borders = np.asarray(borders)
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


def _spread(values: np.ndarray, name: str) -> float:
    """Standard deviation of `values`, rejecting the degenerate zero case.

    Dividing by a zero spread yields NaN, which `np.argsort` orders arbitrarily
    rather than failing, so the ranking would be silently meaningless.
    """
    sd = float(values.std())
    if sd == 0:
        raise ValueError(f"{name} have zero spread, so cannot be scaled for ranking.")
    return sd


def rank_schools(
    student_xy: np.ndarray,
    school_xy: np.ndarray,
    school_scores: np.ndarray | None = None,
    performance_weight: float = 0.0,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank schools for each student and students for each school.

    Student preferences trade travel off against school performance, while
    school priorities are the transpose of the same pairwise distances, so one
    distance matrix serves both.

    Distances and scores are each divided by their own standard deviation before
    being combined, so `performance_weight` is a unit-free share rather than a
    metres-per-score-point rate. Preference cost is

        (1 - w) * distance / sd(distance) - w * score / sd(score)

    ranked ascending, so nearer and higher-scoring schools come first.

    Args:
        student_xy (np.ndarray): Student coordinates, shape (n_students, 2).

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2). Must
        share a CRS with `student_xy`.

        school_scores (np.ndarray | None, optional): Performance score per
        school, aligned with `school_xy`, higher being better. Required when
        `performance_weight` > 0. Defaults to None.

        performance_weight (float, optional): Share of the preference ranking
        driven by performance rather than distance, in [0, 1]. 0 gives
        nearest-first ordering, 1 ranks on performance alone. School priorities
        are unaffected either way. Defaults to 0.0.

        noise_scale (float, optional): Standard deviation of Gaussian noise
        added to the combined preference cost, measured in units of that cost
        rather than in metres. Useful to break ties or add mild randomness
        without destroying the underlying signal. Set to 0 for a deterministic
        ordering. School priorities are always ranked on the unperturbed
        distances. Defaults to 0.0.

        rng (np.random.Generator | None, optional): Source of randomness, used
        only when `noise_scale` > 0. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Student preferences of shape
        (n_students, n_schools) holding school indices best-first, and school
        priorities of shape (n_schools, n_students) holding student indices
        nearest-first.
    """
    if not 0.0 <= performance_weight <= 1.0:
        raise ValueError(
            f"performance_weight must lie in [0, 1], got {performance_weight}."
        )

    distances = spdist.cdist(student_xy, school_xy)
    cost = (1 - performance_weight) * distances / _spread(distances, "Distances")

    if performance_weight > 0:
        if school_scores is None:
            raise ValueError("school_scores is required when performance_weight > 0.")
        scores = np.asarray(school_scores, dtype=float)
        if scores.shape != (school_xy.shape[0],):
            raise ValueError(
                f"school_scores must hold one score per school: expected shape "
                f"{(school_xy.shape[0],)}, got {scores.shape}."
            )
        if not np.isfinite(scores).all():
            raise ValueError("school_scores holds non-finite values.")
        cost = cost - performance_weight * scores / _spread(scores, "School scores")

    if noise_scale > 0:
        rng = rng or np.random.default_rng()
        cost = cost + rng.normal(0, noise_scale, size=cost.shape)

    student_preferences = np.argsort(cost, axis=1).astype(np.int32)
    school_priorities = np.argsort(distances, axis=0).T.astype(np.int32)
    return student_preferences, school_priorities
