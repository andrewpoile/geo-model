import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

CRS = "EPSG:27700"  # British National Grid; eastings/northings in metres

PRIMARY_PHASES = ["All-through", "Middle deemed primary", "Primary"]
SECONDARY_PHASES = ["All-through", "Middle deemed secondary", "Secondary"]

# ONS single-year-of-age estimates for 2021 LSOAs, one workbook per run of
# years, mid-2011 to mid-2024, each year a sheet of its own. A workbook is named
# for the years it spans, sapelsoasyoa20222024.xlsx for mid-2022 to mid-2024,
# and matched on that name, so the lock file Excel writes beside a workbook it
# has open is passed over.
POPULATION_DIR = Path("data/student_data/LSOA_PopEstimates")
POPULATION_WORKBOOKS = "sapelsoasyoa*.xlsx"
POPULATION_CACHE = Path("temp/population_lsoa.pkl")
# The ages whose cohorts are sampled: four-year-olds start primary school,
# eleven-year-olds secondary.
COHORT_AGES = [4, 11]
IDACI_CSV = Path(
    "data/student_data/IoD2025/"
    "File_3_IoD2025 Supplementary Indices_IDACI and IDAOPI.csv"
)
IDACI_SCORES_CSV = Path(
    "data/student_data/IoD2025/"
    "File_5_IoD2025_Scores_for_the_Indices_of_Deprivation(IoD2025 Scores).csv"
)
BOUNDARIES_DIR = Path("data/student_data/LSOA_Boundaries_geospacial_data_2021")
CENTROIDS_DIR = Path("data/student_data/LSOA_PopCentroids_geospatial_data_2021")
REGISTER_CSV = Path("data/school_data/edubasealldata20260225.csv")
# The register's GSSLACode for some schools, Dorset's and Bournemouth,
# Christchurch and Poole's among them: a placeholder rather than an ONS code.
NO_GSS_CODE = "X999999"
# ONS: every local authority district to the county or unitary authority it
# falls under, April 2025.
COUNTY_LOOKUP_CSV = Path(
    "data/student_data/Local_Authority_District_to_County_and_Unitary_"
    "Authority_(April_2025)_Lookup_in_UK_v2.csv"
)

# The local authorities simulated unless others are named, spelled as the
# register's "LA (name)" spells them.
LAS = ["Southampton"]

# DfE school-level applications and offers: the offers made on national offer
# day for entry to every state-funded school in England, one row per school,
# phase and admission year, 2014/15 to 2026/27.
OFFERS_CSV = Path(
    "data/school_data/school_applications_and_offers/supporting-files/"
    "AppsandOffers_2026_SchoolLevel29062026.csv"
)
# The admission years a school's places offered are averaged over, as the file
# codes them: the latest five, 2022/23 to 2026/27, so a school that grew or
# shrank before them is sized as it now stands.
OFFER_YEARS = [202223, 202324, 202425, 202526, 202627]

# 2024-2025 is the newest release in the data folder, but it publishes no P8MEA
# for any school in England, so 2023-2024 is the latest usable year.
KS4_CSV = Path("data/school_data/Performancetables_csv/2023-2024/england_ks4final.csv")

# NTS0614a: trips to and from school by trip length, main mode and age, England,
# 2002 onwards. Education trips under 50 miles only; bus is private and local
# bus together, other is rail and everything else.
NTS_MODE_BY_LENGTH_ODS = Path("data/travel_data/national_travel_survey/nts0614.ods")
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
    """Import the mean and standard deviation over the years of every LSOA's
    cohort of each age in COHORT_AGES, the normal each seed draws its students
    from.

    A cohort is the girls and boys of its age, read from every year the
    workbooks in POPULATION_DIR hold, each year weighted equally. A year two
    workbooks hold is read from the later release, which revises it. The
    standard deviation is the sample one over the years. Only five of each
    sheet's 187 columns are used, so the summary is cached and reused until a
    workbook changes.

    Returns:
        pd.DataFrame: One row per LSOA, carrying "LSOA21CD" and, for each age
        a in COHORT_AGES, "Age a Mean" and "Age a SD", with the years read in
        its attrs' "years".

    Raises:
        FileNotFoundError: If POPULATION_DIR holds no workbook named as
        POPULATION_WORKBOOKS.

        ValueError: If fewer than two years are held, so a cohort has no
        spread over them, or a year holds no estimate for an LSOA another
        year holds.
    """
    workbooks = sorted(POPULATION_DIR.glob(POPULATION_WORKBOOKS))
    if not workbooks:
        raise FileNotFoundError(
            f"{POPULATION_DIR} holds no population workbook {POPULATION_WORKBOOKS}."
        )
    source_key = tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in workbooks
    )

    if POPULATION_CACHE.exists():
        cached = pd.read_pickle(POPULATION_CACHE)
        if cached.attrs.get("source_key") == source_key:
            return cached

    # Workbooks are named for the years they span, so name order is release
    # order, and a year a later workbook holds again is its revision.
    sheets: dict[int, tuple[Path, str]] = {}
    for path in workbooks:
        with pd.ExcelFile(path, engine="calamine") as book:
            for sheet in book.sheet_names:
                year = re.fullmatch(r"Mid-(\d{4}) LSOA 2021", str(sheet))
                if year:
                    sheets[int(year[1])] = (path, str(sheet))
    if len(sheets) < 2:
        raise ValueError(
            f"{POPULATION_DIR} holds estimates for fewer than two years, "
            f"{sorted(sheets)}, so a cohort has no spread over the years."
        )

    cohorts: dict[int, dict[int, pd.Series]] = {age: {} for age in COHORT_AGES}
    for year, (path, sheet) in sorted(sheets.items()):
        estimates = pd.read_excel(
            path,
            sheet,
            skiprows=3,
            engine="calamine",
            usecols=[
                "LSOA 2021 Code",
                *(f"{sex}{age}" for age in COHORT_AGES for sex in "FM"),
            ],
        ).set_index("LSOA 2021 Code")
        for age in COHORT_AGES:
            cohorts[age][year] = estimates[f"F{age}"] + estimates[f"M{age}"]

    summary = {}
    for age in COHORT_AGES:
        by_year = pd.concat(cohorts[age], axis=1)
        missing = by_year.isna().any(axis=1)
        if missing.any():
            raise ValueError(
                f"Not every year holds an age {age} estimate for these LSOAs: "
                + ", ".join(by_year.index[missing])
            )
        summary[f"Age {age} Mean"] = by_year.mean(axis=1)
        summary[f"Age {age} SD"] = by_year.std(axis=1, ddof=1)
    population = pd.DataFrame(summary).rename_axis("LSOA21CD").reset_index()
    population.attrs["source_key"] = source_key
    population.attrs["years"] = sorted(sheets)

    POPULATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place whole, so a run started meanwhile
    # never reads half a cache.
    partial = POPULATION_CACHE.with_suffix(f".{os.getpid()}.tmp")
    population.to_pickle(partial)
    partial.replace(POPULATION_CACHE)
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


def load_places_offered(years: list[int] = OFFER_YEARS) -> pd.DataFrame:
    """Import the Year 7 places each secondary school offers, keyed on local
    authority and establishment number.

    A school's figure is the floor of the mean of the places it offered for
    Year 7 entry over `years`, taken over the years among them it made offers
    in. Year 7 is the cohort the matching admits, so the figure sizes a school
    directly, where the register's capacity covers every year group a school
    teaches, sixth form included. The places offered are the offers made on
    national offer day, so they are capped by demand: at an undersubscribed
    school they fall below its published admission number. Offers for entry
    into another year, Year 9 at an upper school, are left out.

    Args:
        years (list[int], optional): Admission years, as the file codes them,
        202627 for 2026/27. Defaults to OFFER_YEARS.

    Returns:
        pd.DataFrame: One row per school that offered Year 7 places in
        `years`, carrying "LA (code)", "EstablishmentNumber" and
        "PlacesOffered".
    """
    offers = pd.read_csv(
        OFFERS_CSV,
        encoding="utf-8-sig",
        usecols=[
            "time_period",
            "school_phase",
            "school_laestab",
            "total_number_places_offered",
            "entry_year",
        ],
        dtype={"entry_year": str},
    )
    missing = sorted(set(years) - set(offers["time_period"]))
    if missing:
        raise ValueError(f"{OFFERS_CSV.name} holds no admission year {missing}.")
    offers = offers[
        (offers["school_phase"] == "Secondary")
        & (offers["entry_year"] == "7")
        & offers["time_period"].isin(years)
    ]
    places = offers.groupby("school_laestab")["total_number_places_offered"].mean()
    # The file keys a school on its local authority's code and establishment
    # number written together, the latter as the last four digits.
    laestab = places.index.to_numpy()
    return pd.DataFrame(
        {
            "LA (code)": laestab // 10000,
            "EstablishmentNumber": laestab % 10000,
            "PlacesOffered": np.floor(places.to_numpy()).astype(int),
        }
    )


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


def authority_of_district(districts: pd.Series) -> pd.Series:
    """The ONS code of the local authority each district falls under.

    A two-tier district, coded E07, falls under its county, read from the ONS
    lookup. Every other district, a unitary authority, metropolitan district
    or London borough, is a local authority of its own. The lookup is read
    for the two-tier districts alone: its April 2025 edition recodes Barnsley
    and Sheffield, which the IoD's 2024 districts and the register both still
    hold under their earlier codes.

    Args:
        districts (pd.Series): ONS district codes.

    Returns:
        pd.Series: The ONS code of each district's local authority, aligned
        with `districts`.

    Raises:
        ValueError: If the lookup holds no county for a two-tier district.
    """
    lookup = pd.read_csv(
        COUNTY_LOOKUP_CSV, encoding="utf-8-sig", usecols=["LAD25CD", "CTYUA25CD"]
    ).set_index("LAD25CD")["CTYUA25CD"]
    two_tier = districts.str.startswith("E07")
    unlisted = two_tier & ~districts.isin(lookup.index)
    if unlisted.any():
        raise ValueError(
            f"{COUNTY_LOOKUP_CSV.name} holds no county for these two-tier "
            "districts: " + ", ".join(sorted(set(districts[unlisted])))
        )
    return districts.where(~two_tier, districts.map(lookup))


def authority_codes(las: list[str]) -> pd.Series:
    """The ONS code of each local authority in `las`, as the register holds it.

    The register names the authority running each school and gives its ONS
    code, so it is the one source that links an authority's name to the
    districts the deprivation files code. Only open schools are read, since a
    closed school can carry the code of an authority since abolished.

    Args:
        las (list[str]): Local authorities, as the register's "LA (name)"
        spells them.

    Returns:
        pd.Series: The ONS code of each authority, indexed by its name in
        `las` order.

    Raises:
        ValueError: If no open school is run by an authority of that name, or
        its schools carry more than one code.
    """
    register = pd.read_csv(
        REGISTER_CSV,
        encoding="latin-1",
        usecols=["LA (name)", "GSSLACode (name)", "EstablishmentStatus (name)"],
    )
    register = register[
        (register["EstablishmentStatus (name)"] == "Open")
        & (register["GSSLACode (name)"] != NO_GSS_CODE)
    ]
    codes = register.groupby("LA (name)")["GSSLACode (name)"].unique()
    unknown = [la for la in las if la not in codes.index]
    if unknown:
        raise ValueError(
            f"No open school in the register is run by a local authority named "
            f"{unknown}."
        )
    ambiguous = {la: list(codes[la]) for la in las if len(codes[la]) != 1}
    if ambiguous:
        raise ValueError(
            f"These local authorities' schools carry more than one ONS code: "
            f"{ambiguous}."
        )
    return pd.Series({la: codes[la][0] for la in las})


def load_areas(las: list[str] = LAS) -> gpd.GeoDataFrame:
    """Import the LSOAs of the local authorities `las` with their population,
    deprivation and geometry.

    An LSOA belongs to the authority its district falls under, as
    `authority_of_district` reads it from the IoD's 2024 districts.

    Args:
        las (list[str], optional): Local authorities, as the register's "LA
        (name)" spells them. Defaults to LAS.

    Returns:
        gpd.GeoDataFrame: One row per LSOA, positionally indexed, carrying
        "LSOA21CD", "LSOA21NM", "LA (name)", "IDACI" (the rank), "IDACI
        Decile", "IDACI Score" (the share of children living in
        income-deprived families), "Age 4 Mean", "Age 4 SD", "Age 11 Mean"
        and "Age 11 SD" (each cohort's mean and standard deviation over the
        years, as `load_population` reads them), a "Centroids" point and a
        "Borders" polygon.

    Raises:
        ValueError: If an authority holds no LSOA the IoD ranks, as a Welsh
        one does not.
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
    codes = authority_codes(las)
    authority = authority_of_district(idaci["Local Authority District code (2024)"])
    selected = authority.isin(codes)
    idaci = idaci[selected].assign(
        **{"LA (name)": authority[selected].map(pd.Series(codes.index, codes))}
    )
    empty = [la for la in las if la not in set(idaci["LA (name)"])]
    if empty:
        raise ValueError(f"{IDACI_CSV.name} ranks no LSOA in {empty}.")

    # Import spacial boundaries for LSOAs. The full-resolution file holds all 35672
    # areas, so the authorities' LSOAs are selected during the read rather than
    # after it.
    geoborders = gpd.read_file(
        BOUNDARIES_DIR,
        where="LSOA21CD IN ({})".format(
            ",".join(f"'{code}'" for code in idaci["LSOA21CD"])
        ),
    )
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
        idaci[["LSOA21CD", "LA (name)", "IDACI", "IDACI Decile"]], "inner", "LSOA21CD"
    )

    # Import the IDACI score, the share of an LSOA's children living in
    # income-deprived families, necessary for drawing which students are
    # disadvantaged. Merged left, so an LSOA the file does not score is named
    # rather than dropped.
    scores = pd.read_csv(
        IDACI_SCORES_CSV,
        usecols=[
            "LSOA code (2021)",
            "Income Deprivation Affecting Children Index (IDACI) Score (rate)",
        ],
    ).rename(
        columns={
            "LSOA code (2021)": "LSOA21CD",
            "Income Deprivation Affecting Children Index (IDACI) Score (rate)": "IDACI Score",
        }
    )
    geomerge = geomerge.merge(scores, "left", "LSOA21CD", validate="one_to_one")
    unscored = ~geomerge["IDACI Score"].between(0, 1)
    if unscored.any():
        raise ValueError(
            "No IDACI score in [0, 1] held, so no students can be drawn "
            "disadvantaged: " + ", ".join(geomerge.loc[unscored, "LSOA21CD"])
        )
    return geomerge[
        [
            "LSOA21CD",
            "LSOA21NM",
            "LA (name)",
            "IDACI",
            "IDACI Decile",
            "IDACI Score",
            *(f"Age {age} {stat}" for age in COHORT_AGES for stat in ("Mean", "SD")),
            "Centroids",
            "Borders",
        ]
    ]


def load_schools(
    las: list[str] = LAS,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Import the open state schools the local authorities `las` run, split by
    phase.

    Every row and plot of a secondary school is keyed on its name, so a name
    two of the authorities' schools share takes the authority's name after it
    in brackets.

    Args:
        las (list[str], optional): Local authorities, as the register's "LA
        (name)" spells them. Defaults to LAS.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: The primary and the
        secondary schools, each carrying the register columns the model uses,
        "P8MEA" and a point geometry, the secondary frame carrying
        "PlacesOffered" as well, as `load_places_offered` reads it. A
        secondary school without a Progress 8 score cannot be ranked on
        performance, and one that offered no Year 7 place takes no part in a
        Year 7 matching, so both are dropped from the secondary frame.

    Raises:
        ValueError: If an authority runs no open state school, or a secondary
        name repeats within one authority.
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
    unknown = [la for la in las if la not in set(schools["LA (name)"])]
    if unknown:
        raise ValueError(
            f"No open state school in the register is run by a local authority "
            f"named {unknown}."
        )
    schools = schools[schools["LA (name)"].isin(las)]

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
    names = secondary_schools["EstablishmentName"]
    shared = names.duplicated(keep=False)
    secondary_schools = secondary_schools.assign(
        EstablishmentName=names.where(
            ~shared, names + " (" + secondary_schools["LA (name)"] + ")"
        )
    )
    repeated = secondary_schools["EstablishmentName"].duplicated(keep=False)
    if repeated.any():
        raise ValueError(
            "These secondary schools share a name within one local authority, "
            "so their rows and plots cannot be told apart: "
            + ", ".join(
                sorted(set(secondary_schools.loc[repeated, "EstablishmentName"]))
            )
        )

    # An all-through school that has never had a KS4 cohort has no Progress 8 score
    # and so cannot be ranked on performance. It still serves the primary phase.
    missing_p8 = secondary_schools[secondary_schools["P8MEA"].isna()]
    if len(missing_p8):
        print(
            "No Progress 8 score, dropped from the secondary choice set: "
            + ", ".join(missing_p8["EstablishmentName"])
        )
        secondary_schools = secondary_schools.dropna(subset=["P8MEA"])

    # The secondary phase is sized on its intake, the places each school offers
    # for Year 7, rather than on the register's capacity across every year
    # group. A school that offered none in OFFER_YEARS, as one admitting at 14
    # or into Year 9 does not, has no Year 7 intake to be matched to.
    secondary_schools = secondary_schools.merge(
        load_places_offered(), "left", ["LA (code)", "EstablishmentNumber"]
    )
    places = secondary_schools["PlacesOffered"]
    no_places = places.isna() | (places == 0)
    if no_places.any():
        print(
            "No Year 7 place offered in admission years "
            + ", ".join(map(str, OFFER_YEARS))
            + ", dropped from the secondary choice set: "
            + ", ".join(secondary_schools.loc[no_places, "EstablishmentName"])
        )
        secondary_schools = secondary_schools[~no_places]

    return primary_schools, secondary_schools.astype({"PlacesOffered": int})
