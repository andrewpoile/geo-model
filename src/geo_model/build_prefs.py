import pandas as pd
import geopandas as gpd

from .utils import sample_with_centroid, build_student_prefs, build_school_priorities

population = pd.read_excel("data/student_data/sapelsoabroadage20222024.xlsx","Mid-2024 LSOA 2021",skiprows=3)
population.rename(columns={"Total":"Total Population","LSOA 2021 Code":"LSOA21CD"},inplace=True)
index_multi_depra = pd.read_csv("data/student_data/File_1_IoD2025 Index of Multiple Deprivation.csv")
index_multi_depra.rename(
    columns={
        "LSOA code (2021)":"LSOA21CD",
        "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)":"IMD",
        r"Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)":"IMD Decile",
    },
    inplace=True,
)

geoborders = gpd.read_file("data/student_data/LSOA_Boundaries_geospacial_data_2021")
geoborders = geoborders.rename_geometry('Borders')
geocentroids = gpd.read_file("data/student_data/LSOA_PopCentroids_geospatial_data_2021")
geocentroids = geocentroids.rename_geometry('Centroids')

geomerge = geoborders.merge(geocentroids[["LSOA21CD","Centroids"]],"inner",on="LSOA21CD")
geomerge = geomerge.merge(population[["LSOA21CD","Total Population"]],"inner","LSOA21CD")
geomerge = geomerge.merge(index_multi_depra[["LSOA21CD","IMD","IMD Decile"]],"inner","LSOA21CD")
geomerge = geomerge[["LSOA21CD","LSOA21NM","IMD","IMD Decile","Total Population","Centroids","Borders"]]

geo_soton = geomerge[geomerge["LSOA21NM"].str.contains("Southampton")]
geo_soton["Students"] = geo_soton.apply(sample_with_centroid,axis=1)

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

school_points = list(schools.geometry)
student_points = geo_soton.Students

geo_soton["Preferences"] = geo_soton.Students.apply(build_student_prefs)
schools["Priorities"] = schools.geometry.apply(build_school_priorities)