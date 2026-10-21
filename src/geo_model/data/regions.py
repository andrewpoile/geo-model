from collections.abc import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from geo_model.data.areas import load_areas


def adjacent_groups(areas: gpd.GeoDataFrame, las: list[str]) -> list[list[str]]:
    """Group `las` into regions of neighbouring local authorities.

    Two authorities are neighbours where an LSOA of each shares a stretch of
    boundary with the other; meeting at a single point does not count. A
    region holds every authority a chain of neighbours reaches, so two that
    do not border each other share a region through one bordering both.

    Args:
        areas (gpd.GeoDataFrame): LSOAs carrying "LA (name)" and a "Borders"
        polygon, uniquely indexed, as `load_areas` returns them.

        las (list[str]): The authorities to group, each holding LSOAs in
        `areas`.

    Returns:
        list[list[str]]: The regions, each naming its authorities in `las`
        order, ordered by where their first authority stands in `las`.
    """
    borders = areas[["LA (name)", "Borders"]]
    pairs = gpd.sjoin(borders, borders, predicate="intersects")
    across = pairs[pairs["LA (name)_left"] != pairs["LA (name)_right"]]
    shared = shapely.length(
        shapely.intersection(
            np.asarray(across["Borders"]),
            np.asarray(areas.loc[across["index_right"], "Borders"]),
        )
    )
    neighbours = across[shared > 0]

    position = pd.Series(np.arange(len(las)), index=las)
    graph = coo_array(
        (
            np.ones(len(neighbours)),
            (
                position[neighbours["LA (name)_left"]].to_numpy(),
                position[neighbours["LA (name)_right"]].to_numpy(),
            ),
        ),
        shape=(len(las), len(las)),
    )
    _, region = connected_components(graph, directed=False)
    groups: dict[int, list[str]] = {}
    for la, label in zip(las, region):
        groups.setdefault(int(label), []).append(la)
    return list(groups.values())


def regions(
    las: list[str], merge: bool
) -> Iterator[tuple[list[str], gpd.GeoDataFrame]]:
    """The regions to simulate: one per local authority in `las`, or with
    `merge` one per group of neighbours, as `adjacent_groups` forms them.

    Every authority's LSOAs are read at once and each region takes its own,
    so a region simulated alone holds the same LSOAs in the same order as
    one read by itself.

    Args:
        las (list[str]): Local authorities, as the register's "LA (name)"
        spells them.

        merge (bool): Simulate neighbouring authorities as one region.

    Yields:
        tuple[list[str], gpd.GeoDataFrame]: The region's authorities, and its
        LSOAs as `load_areas` returns them, positionally indexed.
    """
    if len(set(las)) != len(las):
        raise ValueError(f"Each local authority may be named once, got {las}.")
    areas = load_areas(las)
    groups = adjacent_groups(areas, las) if merge else [[la] for la in las]
    print("Simulating " + "; ".join(" + ".join(group) for group in groups) + ".")
    for group in groups:
        yield group, areas[areas["LA (name)"].isin(group)].reset_index(drop=True)
