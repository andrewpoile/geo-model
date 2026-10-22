import geopandas as gpd
import pandas as pd


def anonymise(
    areas: gpd.GeoDataFrame, schools: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """`areas` and `schools` with every name they carry numbered instead, so
    no row or plot drawn from them names the region.

    Each district becomes "District n" in the order of `areas`, each
    authority "Authority n" in the order it first appears there, and each
    school "School n" in the order of `schools`, n counting from 1. A school's
    district and authority are renamed as the district and the authority are,
    so `district_index` finds the same districts, and a school outside every
    district stays outside. Nothing else changes, so a matching of the frames
    is the matching of the named ones.

    Args:
        areas (gpd.GeoDataFrame): Districts carrying "LSOA21CD", "LSOA21NM"
        and "LA (name)".

        schools (gpd.GeoDataFrame): Schools carrying "LSOA21CD", "LA (name)"
        and "EstablishmentName".

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: The districts and the
        schools, renamed.
    """
    districts = pd.Series(
        [f"District {n}" for n in range(1, len(areas) + 1)], index=areas["LSOA21CD"]
    )
    authorities = {
        la: f"Authority {n}"
        for n, la in enumerate(areas["LA (name)"].unique(), start=1)
    }
    return (
        areas.assign(
            **{
                "LSOA21CD": districts.to_numpy(),
                "LSOA21NM": districts.to_numpy(),
                "LA (name)": areas["LA (name)"].map(authorities),
            }
        ),
        schools.assign(
            **{
                "LSOA21CD": schools["LSOA21CD"].map(districts),
                "LA (name)": schools["LA (name)"].map(authorities),
                "EstablishmentName": [
                    f"School {n}" for n in range(1, len(schools) + 1)
                ],
            }
        ),
    )
