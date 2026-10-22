import geopandas as gpd
import pandas as pd


def numbered(label: str, count: int) -> list[str]:
    """`count` labels "<label> n", n counting from 1, zero-padded to one width
    so the labels sort in the order of their numbers."""
    width = len(str(count))
    return [f"{label} {n:0{width}d}" for n in range(1, count + 1)]


def anonymise(
    areas: gpd.GeoDataFrame, schools: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """`areas` and `schools` with every name they carry numbered instead, so
    no row or plot drawn from them names the region.

    Each district becomes "District n" in the order of its IDACI rank, the
    most deprived first, rather than of its code, so the districts' scores
    cannot be lined up with the codes' order in the national file. Each
    authority becomes "Authority n" in the order it first appears in `areas`,
    and each school "School n" in the order of `schools`, as `numbered` labels
    them. A school's district and authority are renamed as the district and
    the authority are, so `district_index` finds the same districts, and a
    school outside every district stays outside. Nothing else changes, so a
    matching of the frames is the matching of the named ones.

    Args:
        areas (gpd.GeoDataFrame): Districts carrying "LSOA21CD", "LSOA21NM",
        "LA (name)" and "IDACI" (the rank).

        schools (gpd.GeoDataFrame): Schools carrying "LSOA21CD", "LA (name)"
        and "EstablishmentName".

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: The districts and the
        schools, renamed.
    """
    districts = pd.Series(
        numbered("District", len(areas)),
        index=areas.sort_values("IDACI", kind="stable")["LSOA21CD"],
    )
    las = areas["LA (name)"].unique()
    authorities = dict(zip(las, numbered("Authority", len(las))))
    return (
        areas.assign(
            **{
                "LSOA21CD": areas["LSOA21CD"].map(districts),
                "LSOA21NM": areas["LSOA21CD"].map(districts),
                "LA (name)": areas["LA (name)"].map(authorities),
            }
        ),
        schools.assign(
            **{
                "LSOA21CD": schools["LSOA21CD"].map(districts),
                "LA (name)": schools["LA (name)"].map(authorities),
                "EstablishmentName": numbered("School", len(schools)),
            }
        ),
    )
