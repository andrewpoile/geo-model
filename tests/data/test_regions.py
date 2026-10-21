import geopandas as gpd
import pandas as pd
import pytest
import shapely

from geo_model.data import regions as rg
from geo_model.data.regions import adjacent_groups, regions


@pytest.fixture
def four_authorities():
    """LSOAs of four authorities along a line: A's two squares share an edge
    with each other and with B's, C's meets B's at a corner alone, and D's
    stands apart."""
    return gpd.GeoDataFrame(
        {
            "LA (name)": ["A", "A", "B", "C", "D"],
            "Borders": [
                shapely.box(0, 0, 1, 1),
                shapely.box(1, 0, 2, 1),
                shapely.box(2, 0, 3, 1),
                shapely.box(3, 1, 4, 2),
                shapely.box(10, 10, 11, 11),
            ],
        },
        geometry="Borders",
    )


def test_adjacent_groups_joins_authorities_sharing_a_stretch_of_boundary(
    four_authorities,
):
    # Ordered as named, and C, touching B at a single point, stays apart.
    assert adjacent_groups(four_authorities, ["D", "B", "C", "A"]) == [
        ["D"],
        ["B", "A"],
        ["C"],
    ]


def test_adjacent_groups_chains_neighbours_into_one_region(four_authorities):
    # C now shares an edge with B, so A and C share a region through B.
    areas = four_authorities.copy()
    areas.loc[3, "Borders"] = shapely.box(3, 0, 4, 1)

    assert adjacent_groups(areas, ["A", "B", "C", "D"]) == [["A", "B", "C"], ["D"]]


@pytest.mark.parametrize(
    ("merge", "expected"),
    [(False, [["A"], ["B"], ["C"], ["D"]]), (True, [["A", "B"], ["C"], ["D"]])],
)
def test_regions_yields_each_region_its_own_lsoas_positionally_indexed(
    four_authorities, monkeypatch, merge, expected
):
    monkeypatch.setattr(rg, "load_areas", lambda las: four_authorities)

    yielded = list(regions(["A", "B", "C", "D"], merge))

    assert [las for las, _ in yielded] == expected
    for las, areas in yielded:
        assert set(areas["LA (name)"]) == set(las)
        assert areas.index.equals(pd.RangeIndex(len(areas)))


def test_regions_rejects_an_authority_named_twice():
    with pytest.raises(ValueError, match="named once"):
        list(regions(["A", "B", "A"], merge=False))
