import numpy as np
import geopandas as gpd
from pointpats import random as pprandom
from geopandas import points_from_xy
from shapely.geometry import Point, MultiPoint


def sample_with_centroid_primary(row):
    geom = row["Borders"]
    centre = (row["Centroids"].x, row["Centroids"].y)
    size = int(row["F4"]) + int(row["M4"])
    if geom.is_empty or size == 0:
        return MultiPoint()
    pts = pprandom.normal(geom, centre, size=size)
    return points_from_xy(*pts.T).union_all()

def sample_with_centroid_secondary(row):
    geom = row["Borders"]
    centre = (row["Centroids"].x, row["Centroids"].y)
    size = int(row["F11"]) + int(row["M11"])
    if geom.is_empty or size == 0:
        return MultiPoint()
    pts = pprandom.normal(geom, centre, size=size)
    return points_from_xy(*pts.T).union_all()


def distance_based_prefs(
    student_point: Point,
    school_points: gpd.GeoSeries,
    noise_scale: float = 0.0,   # add > 0 to break ties randomly (same units as CRS)
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return student preference lists based on distance to schools with adjustable noise term.

    Args:
        student_point (sp.Point): The student's location.

        school_points (list): School locations (same CRS as student_point).

        noise_scale (float, optional): Standard deviation of Gaussian noise added to distances before ranking.
        Useful to break ties or add mild randomness without destroying proximity
        signal. Set to 0 for deterministic nearest-first ordering. Defaults to 0.0.

    Returns:
        list: List of preferences for each student.
    """
    if student_point:
        distances = np.array([student_point.distance(sp) for sp in list(school_points)])
    else:
        return None
    if noise_scale > 0:
        rng = rng or np.random.default_rng()
        distances = distances + rng.normal(0, noise_scale, size=len(distances))

    order = np.argsort(distances)
    return order

def build_student_preferences(
        row: MultiPoint,
        school_points: gpd.GeoSeries,
) -> list[list]:
    """Return a list of preference lists, one per student per area.

    Args:
        row (MultiPoint): Student locations.

    Returns:
        list[list]: Preferences of students based on distance to schools.
    """
    points = list(row.geoms)   # individual Points from the MultiPoint
    return np.array([
        distance_based_prefs(pt, school_points, noise_scale=0.0)
        for pt in points
    ])


def distance_based_priorities(
    school_point: Point,
    student_clusters: gpd.GeoSeries,
) -> np.ndarray:
    """_summary_

    Args:
        school_point (Point): _description_
        student_clusters (list[MultiPoint]): _description_
        student_ids (list): _description_
        noise_scale (float, optional): _description_. Defaults to 0.0.
        rng (np.random.Generator | None, optional): _description_. Defaults to None.

    Returns:
        list[list]: _description_
    """
    flat_distances = np.array([
        school_point.distance(student_point)
        for student_points in student_clusters
        for student_point in student_points.geoms
    ])
    # for cluster_geom, cluster_ids in zip(student_clusters, student_ids):
    #     for pt, sid in zip(cluster_geom.geoms, cluster_ids):
    #         flat_ids.append(sid)
    #         flat_distances.append(school_point.distance(pt))

    # flat_distances = np.array(flat_distances)

    order = np.argsort(flat_distances)
    return order

def build_school_priorities(
    school_point: Point,
    student_points: gpd.GeoSeries,
) -> list:
    """
    Add a `priority_list` column to schools_gdf.
    Each entry is a flat list of student IDs ordered nearest-first.
    """
    return np.array([
        distance_based_priorities(school_point, student_points)
    ])