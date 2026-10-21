import numpy as np
import pandas as pd
import pytest

from geo_model.data import areas as ar
from geo_model.data import sources

POPULATION_COLUMNS = [
    "LSOA21CD",
    "Age 4 Mean",
    "Age 4 SD",
    "Age 11 Mean",
    "Age 11 SD",
]
LSOAS = ["E01011949", "E01011950"]


def estimates(f4, m4, f11, m11, lsoas=LSOAS):
    """One year's sheet: the columns the model reads, plus two it must drop."""
    return {
        "LAD 2023 Code": ["E06000045"] * len(lsoas),
        "LSOA 2021 Code": lsoas,
        "Total": [1898] * len(lsoas),
        "F4": f4,
        "M4": m4,
        "F11": f11,
        "M11": m11,
    }


def write_population_workbook(path, sheets):
    """A stand-in for an ONS workbook: a cover sheet the loader must pass
    over, then each year's sheet, three banner rows above its header.

    The real release carries 187 single-year-of-age columns; only the five
    the model uses are written here, plus two it must drop.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"Cover": ["not a year"]}).to_excel(
            writer, sheet_name="Cover sheet", index=False
        )
        for sheet, rows in sheets.items():
            # startrow=3 leaves the three banner rows read_excel is told to skip.
            pd.DataFrame(rows).to_excel(
                writer, sheet_name=sheet, startrow=3, index=False
            )


@pytest.fixture
def population_source(tmp_path, monkeypatch):
    """Point the loader at a folder of two workbooks and a cache of its own.

    Both workbooks hold mid-2012, the later one revising it, as the ONS
    releases revise mid-2022.
    """
    source = tmp_path / "estimates"
    monkeypatch.setattr(sources, "POPULATION_DIR", source)
    monkeypatch.setattr(
        sources, "POPULATION_CACHE", tmp_path / "cache" / "population.pkl"
    )
    write_population_workbook(
        source / "sapelsoasyoa20112012.xlsx",
        {
            "Mid-2011 LSOA 2021": estimates([10, 3], [6, 1], [9, 7], [11, 5]),
            "Mid-2012 LSOA 2021": estimates([100] * 2, [100] * 2, [100] * 2, [100] * 2),
        },
    )
    write_population_workbook(
        source / "sapelsoasyoa20122013.xlsx",
        {
            "Mid-2012 LSOA 2021": estimates([8, 2], [10, 4], [12, 6], [10, 8]),
            "Mid-2013 LSOA 2021": estimates([9, 1], [11, 4], [10, 4], [14, 6]),
        },
    )
    return source


# --------------------------------------------------------------------------
# load_population
# --------------------------------------------------------------------------


def test_load_population_summarises_each_cohort_over_the_years(population_source):
    population = ar.load_population()

    assert list(population.columns) == POPULATION_COLUMNS
    assert population.attrs["years"] == [2011, 2012, 2013]
    np.testing.assert_array_equal(population["LSOA21CD"], LSOAS)
    # Girls and boys together, mid-2012 read from the later workbook alone:
    # the four-year-olds are 16, 18, 20 and 4, 6, 5, the eleven-year-olds 20,
    # 22, 24 and 12, 14, 10. The standard deviation is the sample one.
    np.testing.assert_array_equal(population["Age 4 Mean"], [18, 5])
    np.testing.assert_array_equal(population["Age 4 SD"], [2, 1])
    np.testing.assert_array_equal(population["Age 11 Mean"], [22, 12])
    np.testing.assert_array_equal(population["Age 11 SD"], [2, 2])


def test_load_population_writes_a_cache_keyed_on_every_workbook(population_source):
    population = ar.load_population()

    assert sources.POPULATION_CACHE.exists()
    assert population.attrs["source_key"] == tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(population_source.iterdir())
    )


def test_load_population_reuses_the_cache_without_reading_the_workbooks(
    population_source, monkeypatch
):
    first = ar.load_population()

    # Any second read of a workbook would now fail, so a returned frame
    # proves the cache alone served it.
    def fail(*args, **kwargs):
        pytest.fail("re-read a cached workbook")

    monkeypatch.setattr(ar.pd, "read_excel", fail)
    monkeypatch.setattr(ar.pd, "ExcelFile", fail)
    pd.testing.assert_frame_equal(ar.load_population(), first)


def test_load_population_rebuilds_a_cache_left_behind_by_another_file(
    population_source,
):
    ar.load_population()

    # The cache now holds a key from a workbook that has since changed.
    write_population_workbook(
        population_source / "sapelsoasyoa20122013.xlsx",
        {
            "Mid-2012 LSOA 2021": estimates([1], [1], [2], [2], ["E01099999"]),
            "Mid-2013 LSOA 2021": estimates([1], [1], [3], [3], ["E01099999"]),
        },
    )
    (population_source / "sapelsoasyoa20112012.xlsx").unlink()

    rebuilt = ar.load_population()
    np.testing.assert_array_equal(rebuilt["LSOA21CD"], ["E01099999"])
    np.testing.assert_array_equal(rebuilt["Age 11 Mean"], [5])


def test_load_population_fails_loudly_when_no_workbook_is_held(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "POPULATION_DIR", tmp_path / "absent")
    monkeypatch.setattr(sources, "POPULATION_CACHE", tmp_path / "cache.pkl")

    with pytest.raises(FileNotFoundError):
        ar.load_population()


def test_load_population_rejects_a_single_year(population_source):
    for path in population_source.iterdir():
        path.unlink()
    write_population_workbook(
        population_source / "sapelsoasyoa2011.xlsx",
        {"Mid-2011 LSOA 2021": estimates([10, 3], [6, 1], [9, 7], [11, 5])},
    )

    with pytest.raises(ValueError, match="fewer than two years"):
        ar.load_population()


def test_load_population_rejects_a_year_missing_an_lsoa(population_source):
    write_population_workbook(
        population_source / "sapelsoasyoa2014.xlsx",
        {"Mid-2014 LSOA 2021": estimates([1], [1], [1], [1], LSOAS[:1])},
    )

    with pytest.raises(ValueError, match=LSOAS[1]):
        ar.load_population()


def test_load_population_leaves_no_partial_cache_behind(population_source):
    ar.load_population()

    # The cache is written aside and moved into place, so nothing but the
    # cache itself is left in its folder.
    assert [path.name for path in sources.POPULATION_CACHE.parent.iterdir()] == [
        sources.POPULATION_CACHE.name
    ]


# --------------------------------------------------------------------------
# load_areas: against the real data folder
# --------------------------------------------------------------------------


def test_load_areas_carries_an_idaci_score_the_decile_ranks(areas):
    assert areas["IDACI Score"].between(0, 1).all()
    # The decile ranks LSOAs on the score, 1 the most deprived, so no LSOA
    # scores above one in a more deprived decile.
    by_decile = areas.groupby("IDACI Decile")["IDACI Score"].agg(["min", "max"])
    assert (by_decile["max"].iloc[1:].to_numpy() <= by_decile["min"].iloc[:-1]).all()


def test_load_areas_holds_every_lsoa_of_the_authority_and_no_other(areas):
    # The fixture loads the default region, Southampton alone.
    assert sources.LAS == ["Southampton"]

    # Southampton is a unitary authority, so its LSOAs are those its own name
    # heads, as they were selected before authorities could be named.
    assert len(areas) == 152
    assert (areas["LA (name)"] == "Southampton").all()
    assert areas["LSOA21NM"].str.startswith("Southampton ").all()
    assert areas.index.equals(pd.RangeIndex(152))


@pytest.mark.usefixtures("data")
def test_load_areas_reads_a_county_through_its_districts():
    areas = ar.load_areas(["Hampshire"])

    # Hampshire's LSOAs are named for its districts, never for the county,
    # and the lookup places those districts in it.
    assert not areas["LSOA21NM"].str.startswith("Hampshire").any()
    assert areas["LSOA21NM"].str.startswith("Winchester ").any()
    assert not areas["LSOA21NM"].str.startswith("Southampton ").any()


@pytest.mark.usefixtures("data")
def test_load_areas_rejects_an_authority_the_deprivation_files_do_not_rank():
    # Welsh LSOAs are ranked by an index of their own, outside the IoD.
    with pytest.raises(ValueError, match=r"ranks no LSOA in \['Cardiff'\]"):
        ar.load_areas(["Cardiff"])


# --------------------------------------------------------------------------
# authority_of_district
# --------------------------------------------------------------------------


@pytest.fixture
def county_lookup(tmp_path, monkeypatch):
    """Point the loader at a lookup of one unitary authority and two of a
    county's districts, written as the ONS publishes it."""
    path = tmp_path / "lookup.csv"
    path.write_text(
        "LAD25CD,LAD25NM,LAD25NMW,CTYUA25CD,CTYUA25NM,CTYUA25NMW,ObjectId\n"
        "E06000045,Southampton,,E06000045,Southampton,,1\n"
        "E07000086,Eastleigh,,E10000014,Hampshire,,2\n"
        "E07000094,Winchester,,E10000014,Hampshire,,3\n",
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(sources, "COUNTY_LOOKUP_CSV", path)


def test_authority_of_district_maps_a_two_tier_district_to_its_county(
    county_lookup,
):
    districts = pd.Series(["E07000086", "E06000045", "E07000094", "E08000016"])

    authority = ar.authority_of_district(districts)

    # Every other district is an authority of its own, whether the lookup
    # lists it or, as with Barnsley's earlier code, does not.
    assert list(authority) == ["E10000014", "E06000045", "E10000014", "E08000016"]


def test_authority_of_district_rejects_a_two_tier_district_it_cannot_place(
    county_lookup,
):
    with pytest.raises(ValueError, match="E07000099"):
        ar.authority_of_district(pd.Series(["E07000086", "E07000099"]))


# --------------------------------------------------------------------------
# authority_codes: against a register of its own
# --------------------------------------------------------------------------


def test_authority_codes_reads_each_authority_its_one_code(register_source):
    codes = ar.authority_codes(["Beta", "Alpha"])

    # Beta's placeholder is no code, so its one real code stands.
    assert codes.to_dict() == {"Beta": "E10000002", "Alpha": "E06000001"}
    assert list(codes.index) == ["Beta", "Alpha"]


@pytest.mark.parametrize(
    ("las", "message"),
    [
        (["Alpha", "Omega"], r"named \['Omega'\]"),
        # Gamma's only school is closed, so no open school is run by it.
        (["Gamma"], r"named \['Gamma'\]"),
        (["Delta"], "more than one ONS code"),
    ],
)
def test_authority_codes_rejects_an_authority_it_cannot_code(
    register_source, las, message
):
    with pytest.raises(ValueError, match=message):
        ar.authority_codes(las)
