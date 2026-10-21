import pandas as pd
import pytest

from geo_model.data import schools as sc
from geo_model.data import sources

# Stand-ins for the files the school loaders read, written as each is
# published, shared by the tests of every loader that reads them.

KS4_HEADER = "RECTYPE,LEA,ESTAB,URN,SCHNAME,P8MEA\n"


@pytest.fixture
def ks4_source(tmp_path, monkeypatch):
    def write(body):
        path = tmp_path / "ks4final.csv"
        # The release is published with a byte order mark, as the loader expects.
        path.write_text(KS4_HEADER + body, encoding="utf-8-sig")
        monkeypatch.setattr(sources, "KS4_CSV", path)
        return path

    return write


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
        monkeypatch.setattr(sources, "OFFERS_CSV", path)
        return path

    return write


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
            register_row(102, "Beta", sources.NO_GSS_CODE, 4002, "Vale School"),
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
            for year in sc.OFFER_YEARS
            for row in (
                (year, "Secondary", 1014001, "Hill School", 150, 7),
                (year, "Secondary", 1024001, "Hill School", 180, 7),
                (year, "Secondary", 1024002, "Vale School", 210, 7),
                (year, "Secondary", 1024006, "Upper School", 90, 9),
                (year, "Secondary", 1024007, "Empty School", 0, 7),
            )
        ],
    )
    monkeypatch.setattr(sources, "REGISTER_CSV", tmp_path / "register.csv")
    monkeypatch.setattr(sources, "KS4_CSV", tmp_path / "ks4.csv")
    monkeypatch.setattr(sources, "OFFERS_CSV", tmp_path / "offers.csv")
