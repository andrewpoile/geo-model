import os
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

from geo_model.data import sources

# The age of each phase's cohort: the four-year-olds start primary school, the
# eleven-year-olds secondary. Theirs are the cohorts read and sampled.
PHASE_AGE = {"primary": 4, "secondary": 11}
COHORT_AGES = list(PHASE_AGE.values())


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
    workbooks = sorted(sources.POPULATION_DIR.glob(sources.POPULATION_WORKBOOKS))
    if not workbooks:
        raise FileNotFoundError(
            f"{sources.POPULATION_DIR} holds no population workbook "
            f"{sources.POPULATION_WORKBOOKS}."
        )
    source_key = tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in workbooks
    )

    if sources.POPULATION_CACHE.exists():
        cached = pd.read_pickle(sources.POPULATION_CACHE)
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
            f"{sources.POPULATION_DIR} holds estimates for fewer than two years, "
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

    sources.POPULATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place whole, so a run started meanwhile
    # never reads half a cache.
    partial = sources.POPULATION_CACHE.with_suffix(f".{os.getpid()}.tmp")
    population.to_pickle(partial)
    partial.replace(sources.POPULATION_CACHE)
    return population


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
        sources.COUNTY_LOOKUP_CSV,
        encoding="utf-8-sig",
        usecols=["LAD25CD", "CTYUA25CD"],
    ).set_index("LAD25CD")["CTYUA25CD"]
    two_tier = districts.str.startswith("E07")
    unlisted = two_tier & ~districts.isin(lookup.index)
    if unlisted.any():
        raise ValueError(
            f"{sources.COUNTY_LOOKUP_CSV.name} holds no county for these two-tier "
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
        sources.REGISTER_CSV,
        encoding="latin-1",
        usecols=["LA (name)", "GSSLACode (name)", "EstablishmentStatus (name)"],
    )
    register = register[
        (register["EstablishmentStatus (name)"] == "Open")
        & (register["GSSLACode (name)"] != sources.NO_GSS_CODE)
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


def load_areas(las: list[str] = sources.LAS) -> gpd.GeoDataFrame:
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
    idaci = pd.read_csv(sources.IDACI_CSV)
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
        raise ValueError(f"{sources.IDACI_CSV.name} ranks no LSOA in {empty}.")

    # Import spacial boundaries for LSOAs. The full-resolution file holds all 35672
    # areas, so the authorities' LSOAs are selected during the read rather than
    # after it.
    geoborders = gpd.read_file(
        sources.BOUNDARIES_DIR,
        where="LSOA21CD IN ({})".format(
            ",".join(f"'{code}'" for code in idaci["LSOA21CD"])
        ),
    )
    geoborders = geoborders.rename_geometry("Borders")

    # Import the locations of the LSOA population centroids. This layer carries no
    # LSOA21NM, so it is filtered on the codes the boundary read returned.
    geocentroids = gpd.read_file(
        sources.CENTROIDS_DIR,
        where="LSOA21CD IN ({})".format(
            ",".join(f"'{code}'" for code in geoborders["LSOA21CD"])
        ),
    )
    geocentroids = geocentroids.rename_geometry("Centroids")

    # Distances are computed on raw eastings/northings, so both layers must already
    # be on the British National Grid.
    if geoborders.crs != sources.CRS or geocentroids.crs != sources.CRS:
        raise ValueError(
            f"Expected boundary and centroid data in {sources.CRS}, got "
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
        sources.IDACI_SCORES_CSV,
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
