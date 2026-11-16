from dataclasses import fields, replace
from pathlib import Path

from geo_model.model.settings import DEFAULTS, SHORT, Settings
from geo_model.output import region_dir, run_suffix, suffixed


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


def test_run_suffix_names_the_rules_and_every_parameter_off_its_default():
    assert run_suffix(DEFAULTS, "network") == "_dis-score_pri-distance_dst-network"
    settings = replace(
        DEFAULTS,
        disadvantage="score-open",
        priority="bracket",
        round_up_seats=True,
        min_distance=4828,
        capacity_scale=2.5,
        shortest_only=True,
    )
    # The rules lead, and a True field is its name alone.
    assert run_suffix(settings, "euclidean") == (
        "_dis-score-open_pri-bracket_dst-euclidean_rus_mnd-4828_cap-2.5_sho"
    )
    assert suffixed("results.csv", "_dis-score") == "results_dis-score.csv"
    # Every field has a name of its own, so no two settings share a suffix.
    assert set(SHORT) == {field.name for field in fields(Settings)}
    assert len({*SHORT.values(), "dst"}) == len(SHORT) + 1
    assert all(len(short) == 3 for short in SHORT.values())
