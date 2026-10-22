import geopandas as gpd
import numpy as np

from geo_model.data.anonymise import anonymise
from geo_model.model.preferences import district_index


def test_anonymise_numbers_every_name_and_keeps_each_school_in_its_district():
    areas = gpd.GeoDataFrame(
        {
            "LSOA21CD": ["E01000003", "E01000001", "E01000002"],
            "LSOA21NM": ["Here 001A", "There 001A", "Here 001B"],
            "LA (name)": ["Here", "There", "Here"],
        }
    )
    schools = gpd.GeoDataFrame(
        {
            "LSOA21CD": ["E01000002", "E01000003", "E01000009"],
            "LA (name)": ["Here", "Here", "There"],
            "EstablishmentName": ["Beta", "Alpha", "Gamma"],
        }
    )

    anon_areas, anon_schools = anonymise(areas, schools)

    districts = ["District 1", "District 2", "District 3"]
    assert list(anon_areas["LSOA21CD"]) == districts
    assert list(anon_areas["LSOA21NM"]) == districts
    assert list(anon_areas["LA (name)"]) == [
        "Authority 1",
        "Authority 2",
        "Authority 1",
    ]
    assert list(anon_schools["EstablishmentName"]) == [
        "School 1",
        "School 2",
        "School 3",
    ]
    assert list(anon_schools["LA (name)"]) == [
        "Authority 1",
        "Authority 1",
        "Authority 2",
    ]
    # As named: the third district, the first, and outside every district.
    np.testing.assert_array_equal(district_index(anon_schools, anon_areas), [2, 0, -1])
    np.testing.assert_array_equal(district_index(schools, areas), [2, 0, -1])
