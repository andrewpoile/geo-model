import geopandas as gpd
import numpy as np

from geo_model.data.anonymise import anonymise
from geo_model.model.preferences import district_index


def test_anonymise_numbers_every_name_and_keeps_each_school_in_its_district():
    # Ten districts, so the labels are padded to the ten's two digits.
    areas = gpd.GeoDataFrame(
        {
            "LSOA21CD": [f"E0100000{n}" for n in range(10)],
            "LSOA21NM": [f"Here 00{n}A" for n in range(10)],
            "LA (name)": ["Here"] * 5 + ["There"] * 5,
            "IDACI": [30, 10, 20, 90, 80, 70, 60, 50, 40, 100],
        }
    )
    schools = gpd.GeoDataFrame(
        {
            "LSOA21CD": ["E01000002", "E01000000", "E01000099"],
            "LA (name)": ["Here", "Here", "There"],
            "EstablishmentName": ["Beta", "Alpha", "Gamma"],
        }
    )

    anon_areas, anon_schools = anonymise(areas, schools)

    # Numbered by IDACI rank, the most deprived first, not in code order.
    districts = [f"District {n:02d}" for n in (3, 1, 2, 9, 8, 7, 6, 5, 4, 10)]
    assert list(anon_areas["LSOA21CD"]) == districts
    assert list(anon_areas["LSOA21NM"]) == districts
    assert list(anon_areas["LA (name)"]) == ["Authority 1"] * 5 + ["Authority 2"] * 5
    assert list(anon_schools["EstablishmentName"]) == [
        "School 1",
        "School 2",
        "School 3",
    ]
    assert list(anon_schools["LSOA21CD"][:2]) == ["District 02", "District 03"]
    assert list(anon_schools["LA (name)"]) == [
        "Authority 1",
        "Authority 1",
        "Authority 2",
    ]
    # As named: the third district, the first, and outside every district.
    np.testing.assert_array_equal(district_index(anon_schools, anon_areas), [2, 0, -1])
    np.testing.assert_array_equal(district_index(schools, areas), [2, 0, -1])
