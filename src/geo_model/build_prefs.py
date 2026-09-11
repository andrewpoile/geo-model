from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from geo_model.utils import sample_students, rank_bundles, district_index, cohort_capacity
from geo_model.build_routes import route_network, save_routes

SEED = 20260907
CRS = "EPSG:27700"  # British National Grid; eastings/northings in metres

PRIMARY_PHASES = ["All-through", "Middle deemed primary", "Primary"]
SECONDARY_PHASES = ["All-through", "Middle deemed secondary", "Secondary"]

POPULATION_XLSX = Path("data/student_data/sapelsoasyoa20222024.xlsx")
POPULATION_CACHE = Path("temp/population_lsoa.pkl")

# 2024-2025 is the newest release in the data folder, but it publishes no P8MEA
# for any school in England, so 2023-2024 is the latest usable year.
KS4_CSV = Path("data/school_data/Performancetables_csv/2023-2024/852_ks4final.csv")
OUTPUT_NPZ = Path("temp/prefprio.npz")

# Share of the secondary preference ranking driven by Progress 8 rather than
# distance. At 0.3 one Progress 8 point is worth roughly 1.4km of extra travel.
PERFORMANCE_WEIGHT = 0.3

# Share of the travel a route takes out of the ranking of the school it serves.
# At 0.5 a routed school ranks as if it stood half as far away, so a route is
# worth taking but does not put every distant school ahead of the local one.
ROUTE_DISCOUNT = 0.5


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
        POPULATION_XLSX, "Mid-2024 LSOA 2021", skiprows=3, engine="calamine",
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
        KS4_CSV, encoding="utf-8-sig", usecols=["RECTYPE", "LEA", "ESTAB", "P8MEA"],
    )
    ks4 = ks4[ks4["RECTYPE"] == 1]  # school rows, not LA or national aggregates
    ks4 = ks4.assign(P8MEA=pd.to_numeric(ks4["P8MEA"], errors="coerce"))
    return ks4[["LEA", "ESTAB", "P8MEA"]].dropna(subset=["P8MEA"])


def main() -> None:
    """Build the matching inputs for both phases and write them out."""
    rng = np.random.default_rng(SEED)

    population = load_population()

    # Import data on deprivation, necessary for determining route eligibility and measuring dissimilarity.
    index_multi_depra = pd.read_csv("data/student_data/File_1_IoD2025 Index of Multiple Deprivation.csv")
    index_multi_depra.rename(
        columns={
            "LSOA code (2021)":"LSOA21CD",
            "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)":"IMD",
            r"Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)":"IMD Decile",
        },
        inplace=True,
    )

    # Import spacial boundaries for LSOAs. The full-resolution file holds all 35672
    # areas, so Southampton is selected during the read rather than after it.
    geoborders = gpd.read_file(
        "data/student_data/LSOA_Boundaries_geospacial_data_2021",
        where="LSOA21NM LIKE 'Southampton%'",
    )
    geoborders = geoborders.rename_geometry('Borders')

    # Import the locations of the LSOA population centroids. This layer carries no
    # LSOA21NM, so it is filtered on the codes the boundary read returned.
    geocentroids = gpd.read_file(
        "data/student_data/LSOA_PopCentroids_geospatial_data_2021",
        where="LSOA21CD IN ({})".format(
            ",".join(f"'{code}'" for code in geoborders["LSOA21CD"])
        ),
    )
    geocentroids = geocentroids.rename_geometry('Centroids')

    # Distances are computed on raw eastings/northings, so both layers must already
    # be on the British National Grid.
    if geoborders.crs != CRS or geocentroids.crs != CRS:
        raise ValueError(
            f"Expected boundary and centroid data in {CRS}, got "
            f"{geoborders.crs} and {geocentroids.crs}."
        )

    # Merges the above spacial data together.
    geomerge = geoborders.merge(geocentroids[["LSOA21CD","Centroids"]],"inner","LSOA21CD")
    # Merges primary and secondary age population estimates with spacial data.
    geomerge = geomerge.merge(population,"inner","LSOA21CD")
    # Merges deprivation data with spacial data.
    geomerge = geomerge.merge(index_multi_depra[["LSOA21CD","IMD","IMD Decile"]],"inner","LSOA21CD")
    geo_soton = geomerge[["LSOA21CD","LSOA21NM",
                          "IMD","IMD Decile",
                          "Total","F4","F11","M4","M11",
                          "Centroids","Borders"]]

    # Simulate student locations within each LSOA, clustered on its population centroid.
    primary_sizes = (geo_soton["F4"].astype(int) + geo_soton["M4"].astype(int)).values
    secondary_sizes = (geo_soton["F11"].astype(int) + geo_soton["M11"].astype(int)).values

    primary_student_xy, primary_student_lsoa = sample_students(
        geo_soton["Borders"], geo_soton["Centroids"], primary_sizes, rng,
    )
    secondary_student_xy, secondary_student_lsoa = sample_students(
        geo_soton["Borders"], geo_soton["Centroids"], secondary_sizes, rng,
    )

    # Only 13 of the register's 135 columns are used, and reading the rest costs
    # more than everything the register is used for.
    schools = pd.read_csv("data/school_data/edubasealldata20260225.csv",
                          encoding="latin-1",
                          usecols=[
                              "LSOA (code)","LA (code)","LA (name)",
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
                              "Easting",
                              "Northing",
                          ])
    schools = schools.rename(columns={"LSOA (code)":"LSOA21CD"})
    schools = schools[schools["EstablishmentStatus (name)"] == "Open"]
    schools = schools[schools["EstablishmentTypeGroup (name)"].isin(
        ["Academies",
         "Free Schools",
         "Local authority maintained schools"]
    )]
    schools = schools[schools["PhaseOfEducation (name)"].isin(
        ["All-through",
         "Middle deemed primary",
         "Middle deemed secondary",
         "Primary",
         "Secondary"]
    )]
    schools = schools[[
        "LSOA21CD","LA (code)","LA (name)",
        "EstablishmentNumber",
        "EstablishmentName",
        "TypeOfEstablishment (name)",
        "EstablishmentTypeGroup (name)",
        "PhaseOfEducation (name)",
        "StatutoryLowAge",
        "StatutoryHighAge",
        "SchoolCapacity",
        "PercentageFSM",
        "Easting",
        "Northing"
    ]]
    schools = gpd.GeoDataFrame(
        schools,
        geometry=gpd.points_from_xy(
            schools.Easting, schools.Northing, crs=CRS
        )
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
    secondary_schools = schools[schools["PhaseOfEducation (name)"].isin(SECONDARY_PHASES)]

    # An all-through school that has never had a KS4 cohort has no Progress 8 score
    # and so cannot be ranked on performance. It still serves the primary phase.
    missing_p8 = secondary_schools[secondary_schools["P8MEA"].isna()]
    if len(missing_p8):
        print(
            "No Progress 8 score, dropped from the secondary choice set: "
            + ", ".join(missing_p8["EstablishmentName"])
        )
        secondary_schools = secondary_schools.dropna(subset=["P8MEA"])

    primary_school_xy = np.column_stack(
        (primary_schools.Easting.values, primary_schools.Northing.values)
    )
    secondary_school_xy = np.column_stack(
        (secondary_schools.Easting.values, secondary_schools.Northing.values)
    )

    # Routes connect the disadvantaged districts to the secondary schools they
    # cannot reach unaided, so they exist for the secondary phase alone.
    secondary_routes = route_network(geo_soton, secondary_school_xy)
    save_routes(secondary_routes)

    # School priorities are the transpose of the same pairwise distances that rank
    # student preferences, so one distance matrix per phase serves both. Progress 8
    # is a KS4 measure with no primary analogue, so it shapes secondary preferences
    # only, and priorities stay on distance and district alone in both phases.
    primary_student_preferences, primary_school_priorities = rank_bundles(
        primary_student_xy, primary_student_lsoa,
        primary_school_xy, district_index(primary_schools, geo_soton),
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
    )
    secondary_student_preferences, secondary_school_priorities = rank_bundles(
        secondary_student_xy, secondary_student_lsoa,
        secondary_school_xy, district_index(secondary_schools, geo_soton),
        secondary_routes["district_idx"].to_numpy(),
        secondary_routes["school_idx"].to_numpy(),
        school_scores=secondary_schools.P8MEA.values,
        performance_weight=PERFORMANCE_WEIGHT,
        route_discount=ROUTE_DISCOUNT,
    )

    # The register publishes capacity across every year group a school teaches,
    # while the matching admits one cohort, so the two are not interchangeable.
    primary_school_capacities = cohort_capacity(primary_schools)
    secondary_school_capacities = cohort_capacity(secondary_schools)

    print(
        f"Primary: {primary_school_capacities.sum()} seats in a cohort for "
        f"{len(primary_student_xy)} students."
    )
    print(
        f"Secondary: {secondary_school_capacities.sum()} seats in a cohort for "
        f"{len(secondary_student_xy)} students."
    )

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    # Compressed, since a priority array is mostly the filler that stands in for
    # the bundles a school never hears from.
    np.savez_compressed(
        OUTPUT_NPZ,
        primary_student_preferences = primary_student_preferences,
        secondary_student_preferences = secondary_student_preferences,
        primary_school_priorities = primary_school_priorities,
        primary_school_capacities = primary_school_capacities,
        secondary_school_priorities = secondary_school_priorities,
        secondary_school_capacities = secondary_school_capacities,
    )


if __name__ == "__main__":
    main()
