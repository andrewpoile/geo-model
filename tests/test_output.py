from pathlib import Path

from geo_model.output import region_dir


def test_region_dir_names_a_region_for_its_authorities_in_order():
    assert region_dir(["Southampton"]) == Path("output/Southampton")
    # The same authorities write to the same folder however they were named.
    assert region_dir(["Southampton", "Hampshire"]) == region_dir(
        ["Hampshire", "Southampton"]
    )
    assert region_dir(["Southampton", "Hampshire"]) == Path(
        "output/Hampshire+Southampton"
    )
    # An anonymised run writes within the region's folder, beside the named.
    assert region_dir(["Southampton"], anon=True) == Path(
        "output/Southampton/anonymised"
    )
