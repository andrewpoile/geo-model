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
# load_pan
# --------------------------------------------------------------------------

PAN_HEADER = "LA (code),EstablishmentNumber,EstablishmentName,PAN2026,PAN2027\n"


@pytest.fixture
def pan_source(tmp_path, monkeypatch):
    def write(body):
        path = tmp_path / "secondary_pan.csv"
        path.write_text(PAN_HEADER + body)
        monkeypatch.setattr(ld, "PAN_CSV", path)
        return path

    return write


def test_load_pan_reads_the_admission_year_it_is_asked_for(pan_source):
    pan_source(
        "852,4278,Bitterne Park School,360,370\n852,4311,Cantell School,250,260\n"
    )

    pan = ld.load_pan("PAN2027")

    assert list(pan.columns) == ["LA (code)", "EstablishmentNumber", "PAN"]
    np.testing.assert_array_equal(pan["EstablishmentNumber"], [4278, 4311])
    np.testing.assert_array_equal(pan["PAN"], [370, 260])


def test_load_pan_defaults_to_the_year_the_model_runs_on(pan_source):
    pan_source("852,4278,Bitterne Park School,360,370\n")

    np.testing.assert_array_equal(ld.load_pan()["PAN"], [360])


def test_load_pan_rejects_an_admission_year_the_file_does_not_hold(pan_source):
    pan_source("852,4278,Bitterne Park School,360,370\n")

    with pytest.raises(ValueError, match="PAN2099"):
        ld.load_pan("PAN2099")


def test_load_pan_fails_loudly_when_the_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "PAN_CSV", tmp_path / "absent.csv")

    with pytest.raises(FileNotFoundError):
        ld.load_pan()
