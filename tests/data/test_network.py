import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString

from geo_model.data import network as nw
from geo_model.data import sources


def links(rows):
    """A stand-in for `load_links`: (node ids, coordinates, foot) per link."""
    return gpd.GeoDataFrame(
        {
            "nodes": [",".join(ids) for ids, _, _ in rows],
            "foot": [foot for _, _, foot in rows],
            "rail": "no",
            "ferry": "no",
        },
        geometry=[LineString(xy) for _, xy, _ in rows],
        crs=sources.CRS,
    )


# A 100m square a-b-c-d whose side d-a is closed to walking, so a path from a
# to d goes the long way round. A second, longer a-b link bends through a node
# of its own, and must give way to the straight one.
SQUARE = [
    (["a", "b"], [(0, 0), (100, 0)], "yes"),
    (["a", "m", "b"], [(0, 0), (50, 50), (100, 0)], "designated"),
    (["b", "c"], [(100, 0), (100, 100)], "yes"),
    (["c", "d"], [(100, 100), (0, 100)], "yes"),
    (["d", "a"], [(0, 100), (0, 0)], "no"),
]


def test_network_distance_follows_the_walking_links():
    distance = nw.NetworkDistance(
        nw.walk_network(links(SQUARE)), np.array([[0, 100], [50, 100]])
    )
    # (50, -3) meets a-b halfway and goes round by b; (40, 103) meets c-d 60m
    # from c, and walks along it to either school without leaving it.
    assert distance(np.array([[50, -3], [40, 103]])) == pytest.approx(
        np.array([[253, 203], [43, 13]])
    )


def test_unruled_foot_values_raise():
    with pytest.raises(ValueError, match="no walking rule"):
        nw.walk_network(
            links([*SQUARE, (["c", "e"], [(100, 100), (200, 100)], "private")])
        )


def test_points_off_the_network_are_placed_on_its_largest_part():
    island = [*SQUARE, (["e", "f"], [(1000, 0), (1100, 0)], "yes")]
    distance = nw.NetworkDistance(nw.walk_network(links(island)), np.array([[1100, 0]]))
    # Both are placed at b, ignoring the island they sit on.
    assert distance(np.array([[1000, 0]]))[0, 0] == pytest.approx(1900)


def test_network_is_never_shorter_than_the_straight_line(
    areas, secondary, network_cache, monkeypatch
):
    monkeypatch.setattr(sources, "NETWORK_CACHE_DIR", network_cache)
    region = "+".join(sorted(sources.LAS))
    school_xy = secondary[["Easting", "Northing"]].to_numpy()
    xy = np.column_stack([areas.centroid.x, areas.centroid.y])
    read = nw.NetworkDistance(nw.load_walk_network(areas, region), school_xy)(xy)
    assert (read >= nw.Euclidean(school_xy)(xy)).all()

    def unread(*args):
        raise AssertionError("the cached links were not reused")

    monkeypatch.setattr(nw, "load_links", unread)
    cached = nw.NetworkDistance(nw.load_walk_network(areas, region), school_xy)(xy)
    assert np.array_equal(cached, read)
