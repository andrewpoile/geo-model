import numpy as np
import pandas as pd
import pytest

from geo_model import load_data as ld

POPULATION_COLUMNS = ["LSOA21CD", "Total", "F4", "F11", "M4", "M11"]


def write_population_workbook(path, rows, sheet="Mid-2024 LSOA 2021"):
    """A stand-in for the ONS workbook: three banner rows, then the header.

    The real release carries 187 single-year-of-age columns; only the six the
    model uses are written here, plus one it must drop.
    """
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # startrow=3 leaves the three banner rows read_excel is told to skip.
        frame.to_excel(writer, sheet_name=sheet, startrow=3, index=False)
    return frame


@pytest.fixture
def population_source(tmp_path, monkeypatch):
    """Point the loader at a workbook and a cache of its own."""
    source = tmp_path / "population.xlsx"
    monkeypatch.setattr(ld, "POPULATION_XLSX", source)
    monkeypatch.setattr(ld, "POPULATION_CACHE", tmp_path / "cache" / "population.pkl")
    write_population_workbook(
        source,
        {
            "LAD 2023 Code": ["E06000045", "E06000045"],
            "LSOA 2021 Code": ["E01011949", "E01011950"],
            "Total": [1898, 1247],
            "F4": [11, 13],
            "F11": [9, 7],
            "M4": [4, 7],
            "M11": [18, 10],
        },
    )
    return source


# --------------------------------------------------------------------------
# load_population
# --------------------------------------------------------------------------


def test_load_population_keeps_only_the_columns_the_model_uses(population_source):
    population = ld.load_population()

    assert list(population.columns) == POPULATION_COLUMNS
    np.testing.assert_array_equal(population["LSOA21CD"], ["E01011949", "E01011950"])
    np.testing.assert_array_equal(population["F11"], [9, 7])


def test_load_population_writes_a_cache_keyed_on_the_source_file(population_source):
    population = ld.load_population()

    assert ld.POPULATION_CACHE.exists()
    stat = population_source.stat()
    assert population.attrs["source_key"] == (
        population_source.name,
        stat.st_mtime_ns,
        stat.st_size,
    )


def test_load_population_reuses_the_cache_without_reading_the_workbook(
    population_source, monkeypatch
):
    first = ld.load_population()

    # Any second read of the workbook would now fail, so a returned frame
    # proves the cache alone served it.
    monkeypatch.setattr(
        ld.pd, "read_excel", lambda *a, **k: pytest.fail("re-read a cached workbook")
    )
    pd.testing.assert_frame_equal(ld.load_population(), first)


def test_load_population_rebuilds_a_cache_left_behind_by_another_file(
    population_source, tmp_path
):
    ld.load_population()

    # The cache now holds a key from a workbook that is no longer the source.
    write_population_workbook(
        population_source,
        {
            "LAD 2023 Code": ["E06000045"],
            "LSOA 2021 Code": ["E01099999"],
            "Total": [1],
            "F4": [2],
            "F11": [3],
            "M4": [4],
            "M11": [5],
        },
    )

    rebuilt = ld.load_population()
    np.testing.assert_array_equal(rebuilt["LSOA21CD"], ["E01099999"])
    np.testing.assert_array_equal(rebuilt["M11"], [5])


def test_load_population_fails_loudly_when_the_workbook_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ld, "POPULATION_XLSX", tmp_path / "absent.xlsx")
    monkeypatch.setattr(ld, "POPULATION_CACHE", tmp_path / "cache.pkl")

    with pytest.raises(FileNotFoundError):
        ld.load_population()


def test_load_population_leaves_no_partial_cache_behind(population_source):
    ld.load_population()

    # The cache is written aside and moved into place, so nothing but the
    # cache itself is left in its folder.
    assert [path.name for path in ld.POPULATION_CACHE.parent.iterdir()] == [
        ld.POPULATION_CACHE.name
    ]


# --------------------------------------------------------------------------
# load_p8
# --------------------------------------------------------------------------

KS4_HEADER = "RECTYPE,LEA,ESTAB,URN,SCHNAME,P8MEA\n"


@pytest.fixture
def ks4_source(tmp_path, monkeypatch):
    def write(body):
        path = tmp_path / "ks4final.csv"
        # The release is published with a byte order mark, as the loader expects.
        path.write_text(KS4_HEADER + body, encoding="utf-8-sig")
        monkeypatch.setattr(ld, "KS4_CSV", path)
        return path

    return write


def test_load_p8_keeps_school_rows_with_a_published_score(ks4_source):
    ks4_source(
        "1,852,4278,116458,Bitterne Park School,-0.19\n"
        "1,852,4311,116469,Cantell School,0.21\n"
        "2,852,,,Southampton LA average,-0.35\n"  # local authority aggregate
        "4,,,,England average,0.00\n"  # national aggregate
    )

    p8 = ld.load_p8()

    assert list(p8.columns) == ["LEA", "ESTAB", "P8MEA"]
    np.testing.assert_array_equal(p8["ESTAB"], [4278, 4311])
    np.testing.assert_allclose(p8["P8MEA"], [-0.19, 0.21])


@pytest.mark.parametrize("code", ["NP", "NE", "SUPP"])
def test_load_p8_drops_every_suppression_code_the_release_carries(ks4_source, code):
    ks4_source(
        "1,852,4278,116458,Bitterne Park School,-0.19\n"
        f"1,852,9999,116999,Suppressed School,{code}\n"
    )

    p8 = ld.load_p8()

    np.testing.assert_array_equal(p8["ESTAB"], [4278])
    assert p8["P8MEA"].dtype.kind == "f"  # coerced, not left as text


def test_load_p8_returns_nothing_when_no_school_publishes_a_score(ks4_source):
    ks4_source("1,852,4278,116458,Independent School,NP\n")

    p8 = ld.load_p8()

    assert p8.empty
    assert list(p8.columns) == ["LEA", "ESTAB", "P8MEA"]


def test_load_p8_keys_a_score_on_the_authority_as_well_as_the_school(ks4_source):
    # Establishment numbers are only unique within a local authority, so the
    # pair is what the register is joined on.
    ks4_source(
        "1,852,4278,116458,Southampton School,-0.19\n"
        "1,850,4278,116999,Portsmouth School,1.10\n"
    )

    p8 = ld.load_p8()

    np.testing.assert_array_equal(p8["LEA"], [852, 850])
    assert len(p8.drop_duplicates(["LEA", "ESTAB"])) == 2


def test_load_p8_fails_loudly_when_the_release_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "KS4_CSV", tmp_path / "absent.csv")

    with pytest.raises(FileNotFoundError):
        ld.load_p8()


# --------------------------------------------------------------------------
# load_places_offered
# --------------------------------------------------------------------------

OFFERS_HEADER = (
    "time_period,school_phase,school_laestab,school_name,"
    "total_number_places_offered,entry_year,school_urn\n"
)


def write_offers(path, rows):
    """A stand-in for the DfE school-level file, published with a byte order
    mark, holding the columns the loader reads and one it must ignore."""
    path.write_text(
        OFFERS_HEADER + "".join(f"{','.join(map(str, row))},1\n" for row in rows),
        encoding="utf-8-sig",
    )


@pytest.fixture
def offers_source(tmp_path, monkeypatch):
    def write(rows):
        path = tmp_path / "offers.csv"
        write_offers(path, rows)
        monkeypatch.setattr(ld, "OFFERS_CSV", path)
        return path

    return write


def test_load_places_offered_floors_the_mean_of_year_7_offers_in_the_years_named(
    offers_source,
):
    offers_source(
        [
            # 150 and 153 in the years named: a mean of 151.5, floored to 151.
            # The year before them is left out.
            (202425, "Secondary", 1014001, "Hill School", 300, 7),
            (202526, "Secondary", 1014001, "Hill School", 150, 7),
            (202627, "Secondary", 1014001, "Hill School", 153, 7),
            # An upper school's offers are for Year 9, and a primary school's
            # for reception, so neither is a Year 7 intake.
            (202526, "Secondary", 1014002, "Upper School", 90, 9),
            (202627, "Secondary", 1014002, "Upper School", 90, 9),
            (202627, "Primary", 1012001, "Low Primary", 30, "R"),
            # A new school is averaged over the one year it offered places.
            (202627, "Secondary", 1024003, "New School", 121, 7),
            # One that offered none in the years named is left out.
            (202425, "Secondary", 1024004, "Closing School", 80, 7),
        ]
    )

    places = ld.load_places_offered([202526, 202627])

    assert list(places.columns) == ["LA (code)", "EstablishmentNumber", "PlacesOffered"]
    assert places.to_dict("list") == {
        "LA (code)": [101, 102],
        "EstablishmentNumber": [4001, 4003],
        "PlacesOffered": [151, 121],
    }


def test_load_places_offered_rejects_an_admission_year_the_file_does_not_hold(
    offers_source,
):
    offers_source([(202627, "Secondary", 1014001, "Hill School", 150, 7)])

    with pytest.raises(ValueError, match="209900"):
        ld.load_places_offered([202627, 209900])


def test_load_places_offered_fails_loudly_when_the_file_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ld, "OFFERS_CSV", tmp_path / "absent.csv")

    with pytest.raises(FileNotFoundError):
        ld.load_places_offered()


# --------------------------------------------------------------------------
# load_areas: against the real data folder
# --------------------------------------------------------------------------

DATA = ld.POPULATION_XLSX.parent.parent


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_load_areas_carries_an_idaci_score_the_decile_ranks():
    areas = ld.load_areas()

    assert areas["IDACI Score"].between(0, 1).all()
    # The decile ranks LSOAs on the score, 1 the most deprived, so no LSOA
    # scores above one in a more deprived decile.
    by_decile = areas.groupby("IDACI Decile")["IDACI Score"].agg(["min", "max"])
    assert (by_decile["max"].iloc[1:].to_numpy() <= by_decile["min"].iloc[:-1]).all()


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_load_areas_holds_every_lsoa_of_the_authority_and_no_other():
    areas = ld.load_areas(["Southampton"])

    # Southampton is a unitary authority, so its LSOAs are those its own name
    # heads, as they were selected before authorities could be named.
    assert len(areas) == 152
    assert (areas["LA (name)"] == "Southampton").all()
    assert areas["LSOA21NM"].str.startswith("Southampton ").all()
    assert areas.index.equals(pd.RangeIndex(152))


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_load_areas_reads_a_county_through_its_districts():
    areas = ld.load_areas(["Hampshire"])

    # Hampshire's LSOAs are named for its districts, never for the county,
    # and the lookup places those districts in it.
    assert not areas["LSOA21NM"].str.startswith("Hampshire").any()
    assert areas["LSOA21NM"].str.startswith("Winchester ").any()
    assert not areas["LSOA21NM"].str.startswith("Southampton ").any()


@pytest.mark.skipif(not DATA.is_dir(), reason=f"the {DATA} folder is not present")
def test_load_areas_rejects_an_authority_the_deprivation_files_do_not_rank():
    # Welsh LSOAs are ranked by an index of their own, outside the IoD.
    with pytest.raises(ValueError, match=r"ranks no LSOA in \['Cardiff'\]"):
        ld.load_areas(["Cardiff"])


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
    monkeypatch.setattr(ld, "COUNTY_LOOKUP_CSV", path)


def test_authority_of_district_maps_a_two_tier_district_to_its_county(
    county_lookup,
):
    districts = pd.Series(["E07000086", "E06000045", "E07000094", "E08000016"])

    authority = ld.authority_of_district(districts)

    # Every other district is an authority of its own, whether the lookup
    # lists it or, as with Barnsley's earlier code, does not.
    assert list(authority) == ["E10000014", "E06000045", "E10000014", "E08000016"]


def test_authority_of_district_rejects_a_two_tier_district_it_cannot_place(
    county_lookup,
):
    with pytest.raises(ValueError, match="E07000099"):
        ld.authority_of_district(pd.Series(["E07000086", "E07000099"]))


# --------------------------------------------------------------------------
# authority_codes and load_schools: against a register of their own
# --------------------------------------------------------------------------

REGISTER_COLUMNS = [
    "LSOA (code)",
    "LA (code)",
    "LA (name)",
    "GSSLACode (name)",
    "EstablishmentNumber",
    "EstablishmentName",
    "EstablishmentStatus (name)",
    "TypeOfEstablishment (name)",
    "EstablishmentTypeGroup (name)",
    "PhaseOfEducation (name)",
    "StatutoryLowAge",
    "StatutoryHighAge",
    "SchoolCapacity",
    "PercentageFSM",
    "FSM",
    "Easting",
    "Northing",
]


def register_row(la_code, la, gss, estab, name, phase="Secondary", status="Open"):
    """One school of a stand-in register, carrying every column the loaders
    read."""
    low, high = (11, 16) if phase == "Secondary" else (4, 11)
    return [
        "E01000001",
        la_code,
        la,
        gss,
        estab,
        name,
        status,
        "Academy converter",
        "Academies",
        phase,
        low,
        high,
        1000,
        20.0,
        200,
        440000,
        110000,
    ]


@pytest.fixture
def register_source(tmp_path, monkeypatch):
    """Point the loaders at a register, a KS4 release and an offers file of
    their own. Alpha and Beta each run a "Hill School" under the same
    establishment number, one of Beta's schools carries the register's
    placeholder code, its Upper School admits into Year 9 and its Empty
    School offered no place, Gamma's only school is closed and Delta's
    schools carry two codes."""
    register = pd.DataFrame(
        [
            register_row(101, "Alpha", "E06000001", 4001, "Hill School"),
            register_row(101, "Alpha", "E06000001", 2001, "Low Primary", "Primary"),
            register_row(102, "Beta", "E10000002", 4001, "Hill School"),
            register_row(102, "Beta", ld.NO_GSS_CODE, 4002, "Vale School"),
            register_row(102, "Beta", "E10000002", 4006, "Upper School"),
            register_row(102, "Beta", "E10000002", 4007, "Empty School"),
            register_row(
                103, "Gamma", "E06000009", 4003, "Shut School", status="Closed"
            ),
            register_row(104, "Delta", "E06000004", 4004, "One School"),
            register_row(104, "Delta", "E06000005", 4005, "Two School"),
        ],
        columns=REGISTER_COLUMNS,
    )
    register.to_csv(tmp_path / "register.csv", index=False, encoding="latin-1")
    (tmp_path / "ks4.csv").write_text(
        KS4_HEADER
        + "1,101,4001,1,Hill School,0.1\n"
        + "1,102,4001,2,Hill School,-0.2\n"
        + "1,102,4002,3,Vale School,0.3\n"
        + "1,102,4006,4,Upper School,0.0\n"
        + "1,102,4007,5,Empty School,0.5\n",
        encoding="utf-8-sig",
    )
    write_offers(
        tmp_path / "offers.csv",
        [
            row
            for year in ld.OFFER_YEARS
            for row in (
                (year, "Secondary", 1014001, "Hill School", 150, 7),
                (year, "Secondary", 1024001, "Hill School", 180, 7),
                (year, "Secondary", 1024002, "Vale School", 210, 7),
                (year, "Secondary", 1024006, "Upper School", 90, 9),
                (year, "Secondary", 1024007, "Empty School", 0, 7),
            )
        ],
    )
    monkeypatch.setattr(ld, "REGISTER_CSV", tmp_path / "register.csv")
    monkeypatch.setattr(ld, "KS4_CSV", tmp_path / "ks4.csv")
    monkeypatch.setattr(ld, "OFFERS_CSV", tmp_path / "offers.csv")


def test_authority_codes_reads_each_authority_its_one_code(register_source):
    codes = ld.authority_codes(["Beta", "Alpha"])

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
        ld.authority_codes(las)


def test_load_schools_keeps_the_authorities_named_alone(register_source):
    primary, secondary = ld.load_schools(["Alpha"])

    assert list(primary["EstablishmentName"]) == ["Low Primary"]
    # No other authority named runs a school of the same name, so the name
    # stands as the register has it.
    assert list(secondary["EstablishmentName"]) == ["Hill School"]
    assert list(secondary["PlacesOffered"]) == [150]


def test_load_schools_tells_apart_a_name_two_authorities_share(register_source):
    _, secondary = ld.load_schools(["Alpha", "Beta"])

    assert list(secondary["EstablishmentName"]) == [
        "Hill School (Alpha)",
        "Hill School (Beta)",
        "Vale School",
    ]
    # Each keeps its own score and places, joined on its own authority.
    assert list(secondary["P8MEA"]) == [0.1, -0.2, 0.3]
    assert list(secondary["PlacesOffered"]) == [150, 180, 210]


def test_load_schools_drops_a_school_that_offered_no_year_7_place(
    register_source, capsys
):
    _, secondary = ld.load_schools(["Beta"])

    # Upper School admits into Year 9 and Empty School offered none, so
    # neither has a Year 7 intake to be matched to.
    assert list(secondary["EstablishmentName"]) == ["Hill School", "Vale School"]
    out = capsys.readouterr().out
    assert "No Year 7 place offered" in out
    assert "Upper School, Empty School" in out


def test_load_schools_rejects_an_authority_running_no_open_state_school(
    register_source,
):
    with pytest.raises(ValueError, match=r"named \['Gamma'\]"):
        ld.load_schools(["Alpha", "Gamma"])
