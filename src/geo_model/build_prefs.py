from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from utils import sample_students, rank_by_distance

SEED = 20260907
CRS = "EPSG:27700"  # British National Grid; eastings/northings in metres

PRIMARY_PHASES = ["All-through", "Middle deemed primary", "Primary"]
SECONDARY_PHASES = ["All-through", "Middle deemed secondary", "Secondary"]

POPULATION_XLSX = Path("data/student_data/sapelsoasyoa20222024.xlsx")
POPULATION_CACHE = Path("temp/population_lsoa.pkl")

rng = np.random.default_rng(SEED)


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

# Only 11 of the register's 135 columns are used, and reading the rest costs
# more than everything the register is used for.
schools = pd.read_csv("data/school_data/edubasealldata20260225.csv",
                      encoding="latin-1",
                      usecols=[
                          "LSOA (code)","LA (name)",
                          "EstablishmentName",
                          "EstablishmentStatus (name)",
                          "TypeOfEstablishment (name)",
                          "EstablishmentTypeGroup (name)",
                          "PhaseOfEducation (name)",
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
    "LSOA21CD","LA (name)",
    "EstablishmentName",
    "TypeOfEstablishment (name)",
    "EstablishmentTypeGroup (name)",
    "PhaseOfEducation (name)",
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

primary_schools = schools[schools["PhaseOfEducation (name)"].isin(PRIMARY_PHASES)]
secondary_schools = schools[schools["PhaseOfEducation (name)"].isin(SECONDARY_PHASES)]

primary_school_xy = np.column_stack(
    (primary_schools.Easting.values, primary_schools.Northing.values)
)
secondary_school_xy = np.column_stack(
    (secondary_schools.Easting.values, secondary_schools.Northing.values)
)

# School priorities are the transpose of the same pairwise distances that rank
# student preferences, so one distance matrix per phase serves both.
primary_student_preferences, primary_school_priorities = rank_by_distance(
    primary_student_xy, primary_school_xy,
)
secondary_student_preferences, secondary_school_priorities = rank_by_distance(
    secondary_student_xy, secondary_school_xy,
)

primary_school_capacities = np.array(
    primary_schools.SchoolCapacity.values,
    dtype=np.int32,
)
secondary_school_capacities = np.array(
    secondary_schools.SchoolCapacity.values,
    dtype=np.int32,
)

Path("temp").mkdir(parents=True, exist_ok=True)
np.savez(
    "temp/prefprio.npz",
    primary_student_preferences = primary_student_preferences,
    secondary_student_preferences = secondary_student_preferences,
    primary_school_priorities = primary_school_priorities,
    primary_school_capacities = primary_school_capacities,
    secondary_school_priorities = secondary_school_priorities,
    secondary_school_capacities = secondary_school_capacities,
)
