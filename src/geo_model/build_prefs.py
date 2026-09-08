import numpy as np
import pandas as pd
import geopandas as gpd

from utils import sample_with_centroid_primary, sample_with_centroid_secondary,\
    build_student_preferences, build_school_priorities

# Import age-sliced population data, necessary for building primary and secondary age populations.
population = pd.read_excel("data/student_data/sapelsoasyoa20222024.xlsx","Mid-2024 LSOA 2021",skiprows=3)
population.rename(columns={"LSOA 2021 Code":"LSOA21CD"},inplace=True)

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

# Import spacial boundaries for LSOAs.
geoborders = gpd.read_file("data/student_data/LSOA_Boundaries_geospacial_data_2021")
geoborders = geoborders.rename_geometry('Borders')

# Import the locations of the LSOA population centroids.
geocentroids = gpd.read_file("data/student_data/LSOA_PopCentroids_geospatial_data_2021")
geocentroids = geocentroids.rename_geometry('Centroids')

# Merges the above spacial data together.
geomerge = geoborders.merge(geocentroids[["LSOA21CD","Centroids"]],"inner","LSOA21CD")
# Merges primary and secondary age population estimates with spacial data.
geomerge = geomerge.merge(population[["LSOA21CD","Total","F4","F11","M4","M11"]],
                          "inner","LSOA21CD")
# Merges deprivation data with spacial data.
geomerge = geomerge.merge(index_multi_depra[["LSOA21CD","IMD","IMD Decile"]],"inner","LSOA21CD")
geomerge = geomerge[["LSOA21CD","LSOA21NM",
                     "IMD","IMD Decile",
                     "Total","F4","F11","M4","M11",
                     "Centroids","Borders"]]

geo_soton = geomerge[geomerge["LSOA21NM"].str.contains("Southampton")]
geo_soton["Primary Student Locations"] = geo_soton.apply(sample_with_centroid_primary,axis=1)
geo_soton["Secondary Student Locations"] = geo_soton.apply(sample_with_centroid_secondary,axis=1)

schools = pd.read_csv("data/school_data/edubasealldata20260225.csv", encoding="latin-1")
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
        schools.Easting, schools.Northing
    )
)
schools = schools[schools["LA (name)"].isin(["Southampton"])]

primary_school_points = schools[schools["PhaseOfEducation (name)"].isin([
    "All-through",
    "Middle deemed primary",
    # "Middle deemed secondary",
    "Primary",
    # "Secondary",
])].geometry

secondary_school_points = schools[schools["PhaseOfEducation (name)"].isin([
    "All-through",
    # "Middle deemed primary",
    "Middle deemed secondary",
    # "Primary",
    "Secondary",
])].geometry

primary_student_points = geo_soton["Primary Student Locations"].explode()

secondary_student_points = geo_soton["Secondary Student Locations"].explode()

geo_soton["Primary Preferences"] = geo_soton["Primary Student Locations"].apply(
    build_student_preferences,
    school_points=primary_school_points,
)
geo_soton["Secondary Preferences"] = geo_soton["Secondary Student Locations"].apply(
    build_student_preferences,
    school_points=secondary_school_points,
)
schools["Primary Priorities"] = primary_school_points.apply(
    build_school_priorities,
    student_points=geo_soton["Primary Student Locations"],
)
schools["Secondary Priorities"] = secondary_school_points.apply(
    build_school_priorities,
    student_points=geo_soton["Secondary Student Locations"],
)



primary_student_preferences = np.stack(
    geo_soton["Primary Preferences"].explode().dropna().values,
    dtype=np.int32,
)
secondary_student_preferences = np.stack(
    geo_soton["Secondary Preferences"].explode().dropna().values,
    dtype=np.int32,
)

primary_school_priorities = np.vstack(
    schools["Primary Priorities"].dropna().values,
    dtype=np.int32,
)
primary_school_capacities = np.array(
    schools[schools["PhaseOfEducation (name)"].isin([
        "All-through",
        "Middle deemed primary",
        # "Middle deemed secondary",
        "Primary",
        # "Secondary",
    ])].SchoolCapacity.values,
    dtype=np.int32,
)

secondary_school_priorities = np.vstack(
    schools["Secondary Priorities"].dropna().values,
    dtype=np.int32,
)
secondary_school_capacities = np.array(
    schools[schools["PhaseOfEducation (name)"].isin([
        "All-through",
        # "Middle deemed primary",
        "Middle deemed secondary",
        # "Primary",
        "Secondary",
    ])].SchoolCapacity.values,
    dtype=np.int32,
)

np.savez(
    "temp/prefprio.npz",
    primary_student_preferences = primary_student_preferences,
    secondary_student_preferences = secondary_student_preferences,
    primary_school_priorities = primary_school_priorities,
    primary_school_capacities = primary_school_capacities,
    secondary_school_priorities = secondary_school_priorities,
    secondary_school_capacities = secondary_school_capacities,
)