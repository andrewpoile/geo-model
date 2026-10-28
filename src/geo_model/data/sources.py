"""Where every dataset the model reads lives and how it is coded, the cache
built from the population estimates, and the region read unless others are
named.

Paths are relative to the repository root, which every entry point is run
from.
"""

from pathlib import Path

DATA_DIR = Path("data")

CRS = "EPSG:27700"  # British National Grid; eastings/northings in metres

# The local authorities simulated unless others are named, spelled as the
# register's "LA (name)" spells them.
LAS = ["Southampton"]

# ONS single-year-of-age estimates for 2021 LSOAs, one workbook per run of
# years, mid-2011 to mid-2024, each year a sheet of its own. A workbook is named
# for the years it spans, sapelsoasyoa20222024.xlsx for mid-2022 to mid-2024,
# and matched on that name, so the lock file Excel writes beside a workbook it
# has open is passed over.
POPULATION_DIR = DATA_DIR / "student_data/LSOA_PopEstimates"
POPULATION_WORKBOOKS = "sapelsoasyoa*.xlsx"
POPULATION_CACHE = Path("temp/population_lsoa.pkl")
IDACI_CSV = DATA_DIR / (
    "student_data/IoD2025/File_3_IoD2025 Supplementary Indices_IDACI and IDAOPI.csv"
)
IDACI_SCORES_CSV = DATA_DIR / (
    "student_data/IoD2025/"
    "File_5_IoD2025_Scores_for_the_Indices_of_Deprivation(IoD2025 Scores).csv"
)
BOUNDARIES_DIR = DATA_DIR / "student_data/LSOA_Boundaries_geospacial_data_2021"
CENTROIDS_DIR = DATA_DIR / "student_data/LSOA_PopCentroids_geospatial_data_2021"
REGISTER_CSV = DATA_DIR / "school_data/edubasealldata20260225.csv"
# The register's GSSLACode for some schools, Dorset's and Bournemouth,
# Christchurch and Poole's among them: a placeholder rather than an ONS code.
NO_GSS_CODE = "X999999"
# ONS: every local authority district to the county or unitary authority it
# falls under, April 2025.
COUNTY_LOOKUP_CSV = DATA_DIR / (
    "student_data/Local_Authority_District_to_County_and_Unitary_"
    "Authority_(April_2025)_Lookup_in_UK_v2.csv"
)

# DfE school-level applications and offers: the offers made on national offer
# day for entry to every state-funded school in England, one row per school,
# phase and admission year, 2014/15 to 2026/27.
OFFERS_CSV = DATA_DIR / (
    "school_data/school_applications_and_offers/supporting-files/"
    "AppsandOffers_2026_SchoolLevel29062026.csv"
)

# 2024-2025 is the newest release in the data folder, but it publishes no P8MEA
# for any school in England, so 2023-2024 is the latest usable year.
KS4_CSV = DATA_DIR / "school_data/Performancetables_csv/2023-2024/england_ks4final.csv"

# OS Multi-modal Routing Network, August 2026, all of Great Britain in EPSG:4326:
# every road, path, rail line and ferry as a link between two nodes. A region's
# links are cached in NETWORK_CACHE_DIR, one file per region.
MRN_GPKG = DATA_DIR / (
    "route_data/OS Route Data/mrn-gb_6529457/OSMulti-modalRoutingNetwork.gpkg"
)
MRN_LINKS = "mrn_ntwk_transportlink"
NETWORK_CACHE_DIR = Path("temp/network")

# NTS0614a: trips to and from school by trip length, main mode and age, England,
# 2002 onwards. Education trips under 50 miles only; bus is private and local
# bus together, other is rail and everything else.
NTS_MODE_BY_LENGTH_ODS = DATA_DIR / "travel_data/national_travel_survey/nts0614.ods"
NTS_MODE_BY_LENGTH_SHEET = "NTS0614a_length_by_mode"
