"""Read a region's walking and bus networks from the OS Multi-modal Routing
Network and measure distances and peak bus journey times along them to every
school.

A link runs between the first and last OSM node ids of its "nodes" column,
which sit at the two ends of its geometry; links meet only at shared end ids,
so two links crossing at a bridge, sharing a point but not an id, never meet.
Turn restrictions are not read, so a path may take a turn the road forbids.
"""

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import KDTree
from scipy.spatial import distance as spdist

from geo_model.data import sources

# Metres the region's bounding box is widened by before the network is read,
# so a shortest path may leave the region and come back.
NETWORK_BUFFER = 4000
DIRECTIONS = ("forward", "backward")
LINK_COLUMNS = [
    "nodes",
    "foot",
    "rail",
    "ferry",
    "highway",
    "bus",
    "oneway",
    "oneway:bus",
    "oneway:psv",
    *(
        f"{tag}:{direction}{suffix}"
        for direction in DIRECTIONS
        for tag, suffix in (
            ("bus", ""),
            ("maxspeed", ""),
            ("avgspeed", ""),
            ("avgspeed", ":conditional"),
        )
    ),
]
# The "foot" values of a link open to walking, and of one closed to it. Any
# other value is one the model has no rule for, and raises.
FOOT_OPEN = {"yes", "designated"}
FOOT_CLOSED = {"no"}
# The "bus" values of a link direction open to buses, and of one closed to
# them. A road open only to traffic bound for it, or to deliveries, is
# closed, since a route passes through. Any other value raises.
BUS_OPEN = {"yes"}
BUS_CLOSED = {"no", "private", "destination", "delivery"}
# The direction each "oneway" value leaves open, every direction where it is
# absent. Any other value raises.
ONEWAY = {"yes": ("forward",), "-1": ("backward",), "no": DIRECTIONS}
# Roads a bus may not stop on, so no bus journey starts or ends at a node
# only they reach.
NO_STOPPING = {"motorway", "motorway_link"}
# The weekday morning peak a school route runs in, as the conditional
# average speeds name it.
PEAK = "Mo-Fr 07:00-09:00"
# How every distance is measured, one of DISTANCE_CHOICES, as
# `school_distance` measures it:
#   "network"    Along the walking network of the OS Multi-modal Routing
#                Network.
#   "euclidean"  In a straight line: the model as it was before.
DISTANCE_CHOICES = ("network", "euclidean")
DISTANCE = "network"


class Distance(Protocol):
    """Distance from each of a set of points, shape (n_points, 2), to every
    school, shape (n_points, n_schools), and the rule of DISTANCE_CHOICES it
    is measured by."""

    rule: str

    def __call__(self, xy: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Network:
    """A network: each node's BNG position, the cost of the cheapest link
    from each node to each other, every link's geometry with the nodes at its
    start and end, and whether a journey may start or end at each node."""

    node_xy: np.ndarray
    graph: sparse.csr_array
    lines: np.ndarray
    start: np.ndarray
    end: np.ndarray
    stops: np.ndarray


def load_links(
    areas: gpd.GeoDataFrame, buffer: float = NETWORK_BUFFER
) -> gpd.GeoDataFrame:
    """Import every link of MRN_GPKG crossing the region's bounding box,
    widened by `buffer`.

    Args:
        areas (gpd.GeoDataFrame): The region's districts, in sources.CRS.

        buffer (float, optional): Metres the bounding box is widened by.
        Defaults to NETWORK_BUFFER.

    Returns:
        gpd.GeoDataFrame: The links, with LINK_COLUMNS, in sources.CRS.
    """
    xmin, ymin, xmax, ymax = areas.total_bounds
    box = shapely.box(xmin, ymin, xmax, ymax).buffer(buffer, join_style="mitre")
    # Densified every 100m, so the box's edges, which bend in the file's CRS,
    # stay within the bounds taken there.
    bbox = (
        gpd.GeoSeries([box.segmentize(100)], crs=sources.CRS)
        .to_crs("EPSG:4326")
        .total_bounds
    )
    links = gpd.read_file(
        sources.MRN_GPKG,
        layer=sources.MRN_LINKS,
        bbox=tuple(bbox),
        columns=LINK_COLUMNS,
        engine="pyogrio",
    )
    return links.to_crs(sources.CRS)


def walk_network(links: gpd.GeoDataFrame) -> Network:
    """The network of links open to walking: of the links that are neither
    rail lines nor ferries, those whose "foot" is in FOOT_OPEN.

    Args:
        links (gpd.GeoDataFrame): Links, as `load_links` returns them.

    Returns:
        Network: Its nodes and links.

    Raises:
        ValueError: If a link that is neither rail nor ferry has a "foot" in
        neither FOOT_OPEN nor FOOT_CLOSED, a link's "nodes" do not follow its
        geometry, one node id sits at two positions, or a link has no length.
    """
    land = links[(links["rail"] != "yes") & (links["ferry"] != "yes")]
    unknown = land["foot"][~land["foot"].isin(FOOT_OPEN | FOOT_CLOSED)]
    if len(unknown):
        raise ValueError(
            "Links carry foot values with no walking rule: "
            f"{unknown.value_counts(dropna=False).to_dict()}"
        )
    walk = land[land["foot"].isin(FOOT_OPEN)]
    node_xy, start, end = link_ends(walk)
    lengths = walk.length.to_numpy()
    # Each link runs both ways.
    graph = cheapest_links(
        np.r_[start, end], np.r_[end, start], np.r_[lengths, lengths], len(node_xy)
    )
    return Network(
        node_xy,
        graph,
        walk.geometry.to_numpy(),
        start,
        end,
        np.ones(len(node_xy), bool),
    )


def link_ends(links: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The nodes the links run between.

    Args:
        links (gpd.GeoDataFrame): Links, as `load_links` returns them.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Each node's position, shape
        (n_nodes, 2), and the node at each link's start and at its end, each
        of shape (n_links,).

    Raises:
        ValueError: If a link's "nodes" do not follow its geometry, one node id
        sits at two positions, or a link has no length.
    """
    ids = links["nodes"].str.split(",")
    if (shapely.get_num_coordinates(links.geometry.values) != ids.map(len)).any():
        raise ValueError("A link's node ids do not follow its geometry.")
    if (links.length <= 0).any():
        raise ValueError(f"{(links.length <= 0).sum()} links have no length.")

    # Every link's first end, then every link's last.
    end_xy = np.vstack(
        [
            shapely.get_coordinates(shapely.get_point(links.geometry.values, i))
            for i in (0, -1)
        ]
    )
    codes, unique = pd.factorize(pd.concat([ids.str[0], ids.str[-1]]))
    node_xy = np.empty((len(unique), 2))
    node_xy[codes] = end_xy
    if not np.array_equal(node_xy[codes], end_xy):
        raise ValueError("A node id sits at more than one position.")
    return node_xy, codes[: len(links)], codes[len(links) :]


def cheapest_links(
    u: np.ndarray, v: np.ndarray, cost: np.ndarray, n_nodes: int
) -> sparse.csr_array:
    """The graph of the cheapest link from each node u to each node v.

    Args:
        u (np.ndarray): The node each link leaves.

        v (np.ndarray): The node each link reaches, aligned with `u`.

        cost (np.ndarray): Each link's cost, aligned with `u`.

        n_nodes (int): Nodes in the graph.

    Returns:
        sparse.csr_array: The cost from each node to each, shape
        (n_nodes, n_nodes), absent where no link runs.
    """
    edges = pd.DataFrame({"u": u, "v": v, "cost": cost})
    # A loop never shortens a path, and coo_array would sum parallel links.
    edges = edges[edges["u"] != edges["v"]].groupby(["u", "v"], as_index=False).min()
    return sparse.coo_array(
        (edges["cost"], (edges["u"], edges["v"])), shape=(n_nodes, n_nodes)
    ).tocsr()


def peak_speed(links: pd.DataFrame, direction: str) -> pd.Series:
    """Each link's speed in `direction` at the weekday morning peak, in km/h:
    the average speed at PEAK, else the average speed at any time, else the
    speed limit.

    Args:
        links (pd.DataFrame): Links, as `load_links` returns them.

        direction (str): One of DIRECTIONS.

    Returns:
        pd.Series: The speed, NaN where none is given.

    Raises:
        ValueError: If a link gives average speeds by time but none at PEAK.
    """
    conditional = links[f"avgspeed:{direction}:conditional"]
    at_peak = conditional.str.extract(rf"([\d.]+) @ \({PEAK}\)", expand=False)
    unpeaked = conditional.notna() & at_peak.isna()
    if unpeaked.any():
        raise ValueError(
            f"{unpeaked.sum()} links give {direction} average speeds by time but "
            f"none at {PEAK}, as {conditional[unpeaked].iloc[0]!r}."
        )
    return (
        at_peak.astype(float)
        .fillna(links[f"avgspeed:{direction}"])
        .fillna(links[f"maxspeed:{direction}"])
    )


def bus_network(links: gpd.GeoDataFrame) -> Network:
    """The directed network of link directions open to buses, each costing
    the minutes it takes at its `peak_speed`. Of the links that are neither
    rail lines nor ferries, a direction is open where its "bus:<direction>",
    else its "bus", is in BUS_OPEN and its "oneway:bus", else its
    "oneway:psv", else its "oneway", leaves it open by ONEWAY. A journey may
    start or end at a node a link of "highway" outside NO_STOPPING reaches.

    Args:
        links (gpd.GeoDataFrame): Links, as `load_links` returns them.

    Returns:
        Network: Its nodes and links.

    Raises:
        ValueError: If a link direction has a bus access in neither BUS_OPEN
        nor BUS_CLOSED, or a oneway value ONEWAY has no rule for, if its
        "bus:<direction>" opens a direction its oneway closes, if a
        direction open to buses has no speed or none above 0, or as
        `peak_speed` and `link_ends` raise of the links open to buses.
    """
    land = links[(links["rail"] != "yes") & (links["ferry"] != "yes")]
    oneway = land["oneway:bus"].fillna(land["oneway:psv"]).fillna(land["oneway"])
    unruled = oneway.dropna()[~oneway.dropna().isin(ONEWAY)]
    if len(unruled):
        raise ValueError(
            f"Links carry oneway values with no rule: {unruled.value_counts().to_dict()}"
        )
    opened = {}
    for direction in DIRECTIONS:
        access = land[f"bus:{direction}"].fillna(land["bus"])
        unknown = access[~access.isin(BUS_OPEN | BUS_CLOSED)]
        if len(unknown):
            raise ValueError(
                f"Links carry {direction} bus values with no rule: "
                f"{unknown.value_counts(dropna=False).to_dict()}"
            )
        leaving_open = [value for value, ways in ONEWAY.items() if direction in ways]
        left_open = oneway.isna() | oneway.isin(leaving_open)
        # A bus tag of the direction's own opening a direction the oneway
        # closes contradicts it, unless "oneway:bus" or "oneway:psv" lifts it.
        contradicted = land[f"bus:{direction}"].isin(BUS_OPEN) & ~left_open
        if contradicted.any():
            raise ValueError(
                f"{contradicted.sum()} links open {direction} to buses by "
                f'"bus:{direction}" against their "oneway", with no rule for which '
                "holds."
            )
        opened[direction] = (access.isin(BUS_OPEN) & left_open).to_numpy()
    either = opened["forward"] | opened["backward"]
    road = land[either]
    node_xy, start, end = link_ends(road)
    minutes, u, v = [], [], []
    for direction, ends in zip(DIRECTIONS, ((start, end), (end, start))):
        open_ = opened[direction][either]
        speed = peak_speed(road[open_], direction).to_numpy()
        if not (speed > 0).all():
            raise ValueError(
                f"{(~(speed > 0)).sum()} links open to buses {direction} have no "
                "speed above 0."
            )
        # km/h to metres a minute.
        minutes.append(road.length.to_numpy()[open_] / (speed * 1000 / 60))
        u.append(ends[0][open_])
        v.append(ends[1][open_])
    graph = cheapest_links(
        np.concatenate(u), np.concatenate(v), np.concatenate(minutes), len(node_xy)
    )
    stopping = ~road["highway"].isin(NO_STOPPING).to_numpy()
    stops = np.zeros(len(node_xy), bool)
    stops[start[stopping]] = True
    stops[end[stopping]] = True
    return Network(node_xy, graph, road.geometry.to_numpy(), start, end, stops)


def region_links(
    areas: gpd.GeoDataFrame, region: str, buffer: float = NETWORK_BUFFER
) -> gpd.GeoDataFrame:
    """The region's links, as `load_links` reads them, cached in
    NETWORK_CACHE_DIR, a file per region and buffer, and reused until
    MRN_GPKG, the region's bounds or LINK_COLUMNS change. MRN_GPKG must be
    present even when the cache is, since the cache is checked against it.

    Args:
        areas (gpd.GeoDataFrame): The region's districts, in sources.CRS.

        region (str): The region's name, which its cache file takes.

        buffer (float, optional): Passed to `load_links`. Defaults to
        NETWORK_BUFFER.

    Returns:
        gpd.GeoDataFrame: The region's links.

    Raises:
        FileNotFoundError: If MRN_GPKG is absent.
    """
    if not sources.MRN_GPKG.exists():
        raise FileNotFoundError(
            f"{sources.MRN_GPKG} is not present, and the network cache cannot be "
            "checked without it."
        )
    stat = sources.MRN_GPKG.stat()
    key = (
        stat.st_mtime_ns,
        stat.st_size,
        tuple(areas.total_bounds),
        buffer,
        tuple(LINK_COLUMNS),
    )
    cache = sources.NETWORK_CACHE_DIR / f"{region}.{buffer:g}m.pkl"
    if cache.exists():
        links = pd.read_pickle(cache)
        if links.attrs.get("source_key") == key:
            return links

    links = load_links(areas, buffer)
    links.attrs["source_key"] = key
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place whole, so a run started meanwhile
    # never reads half a cache.
    partial = cache.with_suffix(f".{os.getpid()}.tmp")
    links.to_pickle(partial)
    partial.replace(cache)
    return links


class NetworkDistance:
    """Distance from any point to every school along a network: the straight
    line to the nearest point of the nearest link, along that link to one of
    its ends, then the shortest path to an end of the school's own nearest
    link and along it to the straight line to the school. A point on the
    school's own link may go along it directly.

    Points and schools are placed only on the largest connected part of the
    network, so every distance is finite. Of links equally near, the first is
    taken.
    """

    rule = "network"

    def __init__(self, network: Network, school_xy: np.ndarray) -> None:
        """
        Args:
            network (Network): The network, in the CRS of `school_xy`.

            school_xy (np.ndarray): School coordinates, shape (n_schools, 2).
        """
        _, component = csgraph.connected_components(network.graph, directed=False)
        main = component[network.start] == np.bincount(component).argmax()
        self.lines = network.lines[main]
        self.start, self.end = network.start[main], network.end[main]
        self.tree = shapely.STRtree(self.lines)

        self.school_link, self.school_leg, self.school_along = self.place(school_xy)
        ends = np.r_[self.start[self.school_link], self.end[self.school_link]]
        paths = csgraph.dijkstra(network.graph, directed=False, indices=ends)
        n_schools = len(school_xy)
        rest = shapely.length(self.lines[self.school_link]) - self.school_along
        to_school = np.minimum(
            paths[:n_schools] + self.school_along[:, None],
            paths[n_schools:] + rest[:, None],
        )
        self.to_school = np.ascontiguousarray((to_school + self.school_leg[:, None]).T)

    def place(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each point's nearest link, the straight line to it and how far
        along the link it is met.

        Args:
            xy (np.ndarray): Point coordinates, shape (n_points, 2).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The link, as an index
            into the kept links, the straight-line distance to it and the
            distance along it from its start, each of shape (n_points,).
        """
        points = shapely.points(xy)
        point, link = self.tree.query_nearest(points)
        # Of links equally near a point, the lowest index, so the choice is
        # the same on every run.
        order = np.lexsort((link, point))
        _, first = np.unique(point[order], return_index=True)
        link = link[order][first]
        line = self.lines[link]
        return (
            link,
            shapely.distance(points, line),
            shapely.line_locate_point(line, points),
        )

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        """Distance from each point to every school, shape (n_points, n_schools)."""
        link, leg, along = self.place(xy)
        rest = shapely.length(self.lines[link]) - along
        distance = np.minimum(
            along[:, None] + self.to_school[self.start[link]],
            rest[:, None] + self.to_school[self.end[link]],
        )
        direct = np.abs(along[:, None] - self.school_along) + self.school_leg
        shared = link[:, None] == self.school_link
        return leg[:, None] + np.where(shared, np.minimum(distance, direct), distance)


class Euclidean:
    """Straight-line distance from any point to every school."""

    rule = "euclidean"

    def __init__(self, school_xy: np.ndarray) -> None:
        self.school_xy = school_xy

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        """Distance from each point to every school, shape (n_points, n_schools)."""
        return spdist.cdist(xy, self.school_xy)


def school_distance(
    rule: str, links: Callable[[], gpd.GeoDataFrame]
) -> Callable[[np.ndarray], Distance]:
    """Distance to a set of schools, measured by `rule`, the walking network
    built once however many sets of schools are measured along it.

    Args:
        rule (str): One of DISTANCE_CHOICES: "network" along the walking
        network `walk_network` builds of the links, "euclidean" in a straight
        line.

        links (Callable[[], gpd.GeoDataFrame]): The region's links, as
        `region_links` gives them, read only under "network".

    Returns:
        Callable[[np.ndarray], Distance]: From school coordinates, shape
        (n_schools, 2), the `NetworkDistance` or `Euclidean` to them.
    """
    if rule not in DISTANCE_CHOICES:
        raise ValueError(f"rule must be one of {DISTANCE_CHOICES}, got {rule!r}.")
    if rule == "euclidean":
        return Euclidean
    return functools.partial(NetworkDistance, walk_network(links()))


def bus_minutes(
    links: gpd.GeoDataFrame, school_xy: np.ndarray, xy: np.ndarray
) -> np.ndarray:
    """Minutes by bus at the weekday morning peak from each point to every
    school, along the shortest path of the network `bus_network` builds of
    `links`, from the stop nearest the point to the stop nearest the school.
    The walks to and from those stops are not counted. Points and schools are
    placed only on the largest part of the network every node of which
    reaches every other, so every time is finite.

    Args:
        links (gpd.GeoDataFrame): The region's links, as `region_links` gives
        them, in the CRS of `school_xy`.

        school_xy (np.ndarray): School coordinates, shape (n_schools, 2).

        xy (np.ndarray): Point coordinates, shape (n_points, 2).

    Returns:
        np.ndarray: Minutes, shape (n_points, n_schools).
    """
    network = bus_network(links)
    _, component = csgraph.connected_components(
        network.graph, directed=True, connection="strong"
    )
    main = np.flatnonzero(
        (component == np.bincount(component).argmax()) & network.stops
    )
    tree = KDTree(network.node_xy[main])
    school_node = main[tree.query(school_xy)[1]]
    # Paths into each school, read off the network with every link reversed.
    to_school = csgraph.dijkstra(network.graph.T, directed=True, indices=school_node)
    return to_school[:, main[tree.query(xy)[1]]].T
