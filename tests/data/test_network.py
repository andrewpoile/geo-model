import geopandas as gpd
import numpy as np
import pandas as pd
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


def bus_links(rows):
    """A stand-in for `load_links` of roads: (node ids, coordinates, tags) per
    link, every tag absent that a link does not give."""
    frame = links([(ids, xy, "yes") for ids, xy, _ in rows])
    columns = [column for column in nw.LINK_COLUMNS if column not in frame]
    return frame.join(pd.DataFrame([tags for _, _, tags in rows], columns=columns))


# Roads a-b-c, a right-angled triangle of 1km sides. a-b runs at 12 km/h at
# the morning peak, 30 km/h before it; b-c is one-way, at an average 60 km/h
# under a 30 km/h limit; the 1414m hypotenuse c-a has a limit of 6 km/h alone.
PEAKS = "30 @ (Mo-Fr 04:00-07:00); 12 @ (Mo-Fr 07:00-09:00)"
TRIANGLE = [
    (
        ["a", "b"],
        [(0, 0), (1000, 0)],
        {"bus": "yes", **{f"avgspeed:{d}:conditional": PEAKS for d in nw.DIRECTIONS}},
    ),
    (
        ["b", "c"],
        [(1000, 0), (1000, 1000)],
        {
            "bus": "yes",
            "oneway": "yes",
            "avgspeed:forward": 60.0,
            "maxspeed:forward": 30.0,
        },
    ),
    (
        ["c", "a"],
        [(1000, 1000), (0, 0)],
        {"bus": "yes", "maxspeed:forward": 6.0, "maxspeed:backward": 6.0},
    ),
]


def test_bus_minutes_follow_the_peak_speeds_and_one_way_roads():
    xy = np.array([[1000, 0], [1000, 1000]])
    minutes = nw.bus_minutes(bus_links(TRIANGLE), xy, xy)
    # b to c takes a minute; c to b goes round by a, 1414m at 100m a minute
    # then 1000m at 200m a minute.
    np.testing.assert_allclose(minutes, [[0, 1], [np.hypot(1000, 1000) / 100 + 5, 0]])


@pytest.mark.parametrize(
    "tags, message",
    [
        ({"bus": "permissive"}, "bus values with no rule"),
        ({"bus": "yes", "oneway": "yes", "bus:backward": "yes"}, "against their"),
    ],
    ids=["unruled bus value", "bus tag against the oneway"],
)
def test_unruled_bus_tags_raise(tags, message):
    road = [(["a", "b"], [(0, 0), (100, 0)], tags)]
    with pytest.raises(ValueError, match=message):
        nw.bus_network(bus_links(road))


def test_network_is_never_shorter_than_the_straight_line(
    areas, secondary, route_data, monkeypatch
):
    region = "+".join(sorted(sources.LAS))
    school_xy = secondary[["Easting", "Northing"]].to_numpy()
    xy = np.column_stack([areas.centroid.x, areas.centroid.y])
    read = nw.NetworkDistance(
        nw.walk_network(nw.region_links(areas, region)), school_xy
    )(xy)
    assert (read >= nw.Euclidean(school_xy)(xy)).all()

    def unread(*args):
        raise AssertionError("the cached links were not reused")

    monkeypatch.setattr(nw, "load_links", unread)
    cached = nw.NetworkDistance(
        nw.walk_network(nw.region_links(areas, region)), school_xy
    )(xy)
    assert np.array_equal(cached, read)
