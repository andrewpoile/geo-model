from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

CRS = "EPSG:27700"  # British National Grid; eastings/northings in metres

PRIMARY_PHASES = ["All-through", "Middle deemed primary", "Primary"]
SECONDARY_PHASES = ["All-through", "Middle deemed secondary", "Secondary"]

POPULATION_XLSX = Path("data/student_data/sapelsoasyoa20222024.xlsx")
POPULATION_CACHE = Path("temp/population_lsoa.pkl")
IDACI_CSV = Path(
    "data/student_data/File_3_IoD2025 Supplementary Indices_IDACI and IDAOPI.csv"
)
BOUNDARIES_DIR = Path("data/student_data/LSOA_Boundaries_geospacial_data_2021")
CENTROIDS_DIR = Path("data/student_data/LSOA_PopCentroids_geospatial_data_2021")
REGISTER_CSV = Path("data/school_data/edubasealldata20260225.csv")

# Year 7 published admission numbers, transcribed from Southampton City
# Council's determined admission arrangements. Every school holds the same
# number in both published years, so the choice of year changes nothing today.
PAN_CSV = Path("data/school_data/secondary_pan.csv")
PAN_YEAR = "PAN2026"

# 2024-2025 is the newest release in the data folder, but it publishes no P8MEA
# for any school in England, so 2023-2024 is the latest usable year.
KS4_CSV = Path("data/school_data/Performancetables_csv/2023-2024/852_ks4final.csv")

# NTS0614a: trips to and from school by trip length, main mode and age, England,
# 2002 onwards. Education trips under 50 miles only; bus is private and local
# bus together, other is rail and everything else.
NTS_MODE_BY_LENGTH_ODS = Path("data/travel_data/nts/nts0614.ods")
NTS_MODE_BY_LENGTH_SHEET = "NTS0614a_length_by_mode"
NTS_BANDS = [
    "Under 1 mile",
    "1 to under 2 miles",
    "2 to under 5 miles",
    "5 miles and over",
]
NTS_MODES = {
    "Walk (%) [note 2]": "walk",
    "Pedal cycle (%) [note 3]": "cycle",
    "Car or van (%)": "car",
    "Bus (%) [note 4]": "bus",
    "Other transport (%) [note 5]": "other",
}
NTS_TRIPS = "Unweighted sample size: trips (thousands) (number)"
# The latest survey year, the default the shares are taken from.
NTS_YEARS = [2025]


def load_population() -> pd.DataFrame:
    """Import age-sliced population data, necessary for building primary and secondary age populations.

    Only six of the workbook's 187 columns are used, so the extract is cached
    and reused until the source file changes.
    """
    stat = POPULATION_XLSX.stat()
    source_key = (POPULATION_XLSX.name, stat.st_mtime_ns, stat.st_size)

    if POPULATION_CACHE.exists():
        cached = pd.read_pickle(POPULATION_CACHE)
        if cached.attrs.get("source_key") == source_key:
            return cached

    population = pd.read_excel(
        POPULATION_XLSX,
        "Mid-2024 LSOA 2021",
        skiprows=3,
        engine="calamine",
    )
    population = population.rename(columns={"LSOA 2021 Code": "LSOA21CD"})
    population = population[["LSOA21CD", "Total", "F4", "F11", "M4", "M11"]]
    population.attrs["source_key"] = source_key

    POPULATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    population.to_pickle(POPULATION_CACHE)
    return population


def load_p8() -> pd.DataFrame:
    """Import Progress 8 scores, keyed on local authority and establishment number.

    P8MEA is centred on the England average by construction: 0 is average,
    below 0 is below average and above 0 is above average.

    Independent schools carry the suppression code "NP" instead of a score.
    They fall outside the model's school types, so they are dropped here.
    """
    ks4 = pd.read_csv(
        KS4_CSV,
        encoding="utf-8-sig",
        usecols=["RECTYPE", "LEA", "ESTAB", "P8MEA"],
    )
    ks4 = ks4[ks4["RECTYPE"] == 1]  # school rows, not LA or national aggregates
    ks4 = ks4.assign(P8MEA=pd.to_numeric(ks4["P8MEA"], errors="coerce"))
    return ks4[["LEA", "ESTAB", "P8MEA"]].dropna(subset=["P8MEA"])


def load_pan(year: str = PAN_YEAR) -> pd.DataFrame:
    """Import published admission numbers, keyed on local authority and establishment number.

    A PAN is the number of places a school offers in Year 7, which is the
    cohort the matching admits, so it sizes a school directly. The register's
    capacity covers every year group a school teaches instead, sixth form
    included, and so cannot be read as an intake.

    Args:
        year (str, optional): Admission year column to read, as the file
        heads it. Defaults to PAN_YEAR.

    Returns:
        pd.DataFrame: One row per school, carrying "LA (code)",
        "EstablishmentNumber" and "PAN".
    """
    pan = pd.read_csv(PAN_CSV)
    if year not in pan.columns:
        raise ValueError(f"{PAN_CSV.name} holds no admission year {year!r}.")
    return pan[["LA (code)", "EstablishmentNumber", year]].rename(columns={year: "PAN"})


def load_nts_mode_shares(
    years: list[int] = NTS_YEARS, age: str = "11 to 16"
) -> pd.DataFrame:
    """Import the NTS share of school trips by main mode within each trip-length band.

    More than one year is pooled per band as a mean weighted by each year's
    unweighted trip sample, the survey's own measure of how much each year's
    estimate rests on. The 2025 diary moved from paper to digital collection,
    which the survey reports lowered the recorded number of short walks and car
    trips, so shares across that break are not strictly comparable.

    Args:
        years (list[int], optional): Survey years to draw on, 2002 onwards.
        Defaults to NTS_YEARS.

        age (str, optional): Age band as the table labels it, one of "5 to
        16", "5 to 10" and "11 to 16". Defaults to "11 to 16".

    Returns:
        pd.DataFrame: One row per band in NTS_BANDS order, one column per mode
        in NTS_MODES order, each row summing to 1.
    """
    table = pd.read_excel(
        NTS_MODE_BY_LENGTH_ODS, NTS_MODE_BY_LENGTH_SHEET, header=5, engine="calamine"
    )
    table.columns = table.columns.str.strip()
    table = table.rename(columns=NTS_MODES)

    missing = sorted(set(years) - set(table["Year"]))
    if missing:
        raise ValueError(f"NTS0614a holds no year {missing}.")
    if age not in set(table["Age"]):
        raise ValueError(f"NTS0614a holds no age band {age!r}.")
    table = table[table["Year"].isin(years) & (table["Age"] == age)]

    modes = list(NTS_MODES.values())
    shares = (
        pd.DataFrame(
            {
                band: np.average(
                    table.loc[table["Trip length"] == band, modes],
                    axis=0,
                    weights=table.loc[table["Trip length"] == band, NTS_TRIPS],
                )
                for band in NTS_BANDS
            },
            index=modes,
        ).T
        / 100
    )
    # np.average raises on an empty band, so every band is present by here.

    if not np.allclose(shares.sum(axis=1), 1, atol=1e-6):
        raise ValueError(
            "NTS0614a mode shares do not sum to 100 in every band:\n"
            f"{shares.sum(axis=1)}"
        )
    return shares


def load_areas() -> gpd.GeoDataFrame:
    """Import Southampton's LSOAs with their population, deprivation and geometry.

    Returns:
        gpd.GeoDataFrame: One row per LSOA, positionally indexed, carrying
        "LSOA21CD", "LSOA21NM", "IDACI", "IDACI Decile", "Total", "F4", "F11",
        "M4", "M11", a "Centroids" point and a "Borders" polygon.
    """
    population = load_population()

    # Import the Income Deprivation Affecting Children Index, necessary for determining route eligibility and measuring dissimilarity.
    idaci = pd.read_csv(IDACI_CSV)
    idaci.rename(
        columns={
            "LSOA code (2021)": "LSOA21CD",
            "Income Deprivation Affecting Children Index (IDACI) Rank (where 1 is most deprived)": "IDACI",
            "Income Deprivation Affecting Children Index (IDACI) Decile (where 1 is most deprived 10% of LSOAs)": "IDACI Decile",
        },
        inplace=True,
    )

    # Import spacial boundaries for LSOAs. The full-resolution file holds all 35672
    # areas, so Southampton is selected during the read rather than after it.
    geoborders = gpd.read_file(BOUNDARIES_DIR, where="LSOA21NM LIKE 'Southampton%'")
    geoborders = geoborders.rename_geometry("Borders")

    # Import the locations of the LSOA population centroids. This layer carries no
    # LSOA21NM, so it is filtered on the codes the boundary read returned.
    geocentroids = gpd.read_file(
        CENTROIDS_DIR,
        where="LSOA21CD IN ({})".format(
            ",".join(f"'{code}'" for code in geoborders["LSOA21CD"])
        ),
    )
    geocentroids = geocentroids.rename_geometry("Centroids")

    # Distances are computed on raw eastings/northings, so both layers must already
    # be on the British National Grid.
    if geoborders.crs != CRS or geocentroids.crs != CRS:
        raise ValueError(
            f"Expected boundary and centroid data in {CRS}, got "
            f"{geoborders.crs} and {geocentroids.crs}."
        )

    # Merges the above spacial data together.
    geomerge = geoborders.merge(
        geocentroids[["LSOA21CD", "Centroids"]], "inner", "LSOA21CD"
    )
    # Merges primary and secondary age population estimates with spacial data.
    geomerge = geomerge.merge(population, "inner", "LSOA21CD")
    # Merges deprivation data with spacial data.
    geomerge = geomerge.merge(
        idaci[["LSOA21CD", "IDACI", "IDACI Decile"]], "inner", "LSOA21CD"
    )
    return geomerge[
        [
            "LSOA21CD",
            "LSOA21NM",
            "IDACI",
            "IDACI Decile",
            "Total",
            "F4",
            "F11",
            "M4",
            "M11",
            "Centroids",
            "Borders",
        ]
    ]


def load_schools() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Import Southampton's open state schools, split by phase.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: The primary and the
        secondary schools, each carrying the register columns the model uses,
        "P8MEA" and a point geometry, the secondary frame carrying "PAN" as
        well. A secondary school without a Progress 8 score cannot be ranked
        on performance, so it is dropped from the secondary frame.
    """
    # Only 14 of the register's 135 columns are used, and reading the rest costs
    # more than everything the register is used for.
    schools = pd.read_csv(
        REGISTER_CSV,
        encoding="latin-1",
        usecols=[
            "LSOA (code)",
            "LA (code)",
            "LA (name)",
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
        ],
    )
    schools = schools.rename(columns={"LSOA (code)": "LSOA21CD"})
    schools = schools[schools["EstablishmentStatus (name)"] == "Open"]
    schools = schools[
        schools["EstablishmentTypeGroup (name)"].isin(
            ["Academies", "Free Schools", "Local authority maintained schools"]
        )
    ]
    schools = schools[
        schools["PhaseOfEducation (name)"].isin(
            [
                "All-through",
                "Middle deemed primary",
                "Middle deemed secondary",
                "Primary",
                "Secondary",
            ]
        )
    ]
    schools = schools[
        [
            "LSOA21CD",
            "LA (code)",
            "LA (name)",
            "EstablishmentNumber",
            "EstablishmentName",
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
    ]
    schools = gpd.GeoDataFrame(
        schools, geometry=gpd.points_from_xy(schools.Easting, schools.Northing, crs=CRS)
    )
    schools = schools[schools["LA (name)"].isin(["Southampton"])]

    # Progress 8 is joined on establishment number rather than URN: Regents Park
    # Community College became an academy in 2026 and took a new URN, while its
    # establishment number is unchanged.
    schools = schools.astype({"EstablishmentNumber": int}).merge(
        load_p8().rename(columns={"LEA": "LA (code)", "ESTAB": "EstablishmentNumber"}),
        "left",
        ["LA (code)", "EstablishmentNumber"],
    )

    primary_schools = schools[schools["PhaseOfEducation (name)"].isin(PRIMARY_PHASES)]
    secondary_schools = schools[
        schools["PhaseOfEducation (name)"].isin(SECONDARY_PHASES)
    ]

    # An all-through school that has never had a KS4 cohort has no Progress 8 score
    # and so cannot be ranked on performance. It still serves the primary phase.
    missing_p8 = secondary_schools[secondary_schools["P8MEA"].isna()]
    if len(missing_p8):
        print(
            "No Progress 8 score, dropped from the secondary choice set: "
            + ", ".join(missing_p8["EstablishmentName"])
        )
        secondary_schools = secondary_schools.dropna(subset=["P8MEA"])

    # Only the secondary phase publishes an admission number, so only this
    # frame carries one and only it is sized on an intake rather than on the
    # register's capacity across every year group.
    secondary_schools = secondary_schools.merge(
        load_pan(), "left", ["LA (code)", "EstablishmentNumber"]
    )
    missing_pan = secondary_schools[secondary_schools["PAN"].isna()]
    if len(missing_pan):
        raise ValueError(
            "No published admission number held, so an intake cannot be sized: "
            + ", ".join(missing_pan["EstablishmentName"])
        )

    return primary_schools, secondary_schools.astype({"PAN": int})
