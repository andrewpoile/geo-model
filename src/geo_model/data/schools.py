import geopandas as gpd
import numpy as np
import pandas as pd

from geo_model.data import sources

PRIMARY_PHASES = ["All-through", "Middle deemed primary", "Primary"]
SECONDARY_PHASES = ["All-through", "Middle deemed secondary", "Secondary"]

# The admission years a school's places offered are averaged over, as the file
# codes them: the latest five, 2022/23 to 2026/27, so a school that grew or
# shrank before them is sized as it now stands.
OFFER_YEARS = [202223, 202324, 202425, 202526, 202627]


def load_p8() -> pd.DataFrame:
    """Import Progress 8 scores, keyed on local authority and establishment number.

    P8MEA is centred on the England average by construction: 0 is average,
    below 0 is below average and above 0 is above average.

    Independent schools carry the suppression code "NP" instead of a score.
    They fall outside the model's school types, so they are dropped here.
    """
    ks4 = pd.read_csv(
        sources.KS4_CSV,
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
        sources.OFFERS_CSV,
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
        raise ValueError(
            f"{sources.OFFERS_CSV.name} holds no admission year {missing}."
        )
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


def load_schools(
    las: list[str] = sources.LAS,
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
        sources.REGISTER_CSV,
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
        schools,
        geometry=gpd.points_from_xy(schools.Easting, schools.Northing, crs=sources.CRS),
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
