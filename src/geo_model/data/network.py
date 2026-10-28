"""Read a region's walking network from the OS Multi-modal Routing Network
and measure distances along it to every school.

A link runs between the first and last OSM node ids of its "nodes" column,
which sit at the two ends of its geometry; links meet only at shared end ids,
so two links crossing at a bridge, sharing a point but not an id, never meet.
"""

import os
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import distance as spdist

from geo_model.data import sources

# Metres the region's bounding box is widened by before the network is read,
# so a shortest path may leave the region and come back.
NETWORK_BUFFER = 4000
LINK_COLUMNS = ["nodes", "foot", "rail", "ferry"]
# The "foot" values of a link open to walking, and of one closed to it. Any
# other value is one the model has no rule for, and raises.
FOOT_OPEN = {"yes", "designated"}
FOOT_CLOSED = {"no"}


@dataclass(frozen=True)
class Network:
    """An undirected network: each node's BNG position, the shortest link
    length between every two nodes, in metres, and every link's geometry with
    the nodes at its start and end."""

    node_xy: np.ndarray
    graph: sparse.csr_array
    lines: np.ndarray
    start: np.ndarray
    end: np.ndarray


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
    ids = walk["nodes"].str.split(",")
    if (shapely.get_num_coordinates(walk.geometry.values) != ids.map(len)).any():
        raise ValueError("A link's node ids do not follow its geometry.")
    lengths = walk.length.to_numpy()
    if (lengths <= 0).any():
        raise ValueError(f"{(lengths <= 0).sum()} walking links have no length.")

    # Every link's first end, then every link's last.
    end_xy = np.vstack(
        [
            shapely.get_coordinates(shapely.get_point(walk.geometry.values, i))
            for i in (0, -1)
        ]
    )
    codes, unique = pd.factorize(pd.concat([ids.str[0], ids.str[-1]]))
    node_xy = np.empty((len(unique), 2))
    node_xy[codes] = end_xy
    if not np.array_equal(node_xy[codes], end_xy):
        raise ValueError("A node id sits at more than one position.")

    n_links = len(walk)
    start, end = codes[:n_links], codes[n_links:]
    edges = pd.DataFrame(
        {
            "u": np.minimum(start, end),
            "v": np.maximum(start, end),
            "length": lengths,
        }
    )
    # A loop never shortens a path, and coo_array would sum parallel links.
    edges = edges[edges["u"] != edges["v"]].groupby(["u", "v"], as_index=False).min()
    graph = sparse.coo_array(
        (
            np.r_[edges["length"], edges["length"]],
            (np.r_[edges["u"], edges["v"]], np.r_[edges["v"], edges["u"]]),
        ),
        shape=(len(unique), len(unique)),
    ).tocsr()
    return Network(node_xy, graph, walk.geometry.to_numpy(), start, end)


def load_walk_network(
    areas: gpd.GeoDataFrame, region: str, buffer: float = NETWORK_BUFFER
) -> Network:
    """The region's walking network, as `walk_network` builds it from the
    links `load_links` reads. The links are cached in NETWORK_CACHE_DIR and
    reused until MRN_GPKG, the region's bounds, `buffer` or LINK_COLUMNS
    change.

    Args:
        areas (gpd.GeoDataFrame): The region's districts, in sources.CRS.

        region (str): The region's name, which its cache file takes.

        buffer (float, optional): Passed to `load_links`. Defaults to
        NETWORK_BUFFER.

    Returns:
        Network: The region's walking network.
    """
    stat = sources.MRN_GPKG.stat()
    key = (
        stat.st_mtime_ns,
        stat.st_size,
        tuple(areas.total_bounds),
        buffer,
        tuple(LINK_COLUMNS),
    )
    cache = sources.NETWORK_CACHE_DIR / f"{region}.pkl"
    if cache.exists():
        links = pd.read_pickle(cache)
        if links.attrs.get("source_key") == key:
            return walk_network(links)

    links = load_links(areas, buffer)
    links.attrs["source_key"] = key
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place whole, so a run started meanwhile
    # never reads half a cache.
    partial = cache.with_suffix(f".{os.getpid()}.tmp")
    links.to_pickle(partial)
    partial.replace(cache)
    return walk_network(links)


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

    def __init__(self, school_xy: np.ndarray) -> None:
        self.school_xy = school_xy

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        """Distance from each point to every school, shape (n_points, n_schools)."""
        return spdist.cdist(xy, self.school_xy)
