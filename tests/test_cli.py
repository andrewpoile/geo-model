import argparse

import pytest

from geo_model.cli import (
    add_disadvantage_argument,
    add_intake_argument,
    add_progressivity_argument,
    add_region_arguments,
    add_seat_arguments,
    add_seeds_argument,
    add_travel_arguments,
    add_workers_argument,
)
from geo_model.data.sources import LAS
from geo_model.data.travel import NTS_YEARS
from geo_model.measures.travel import CIRCUITY
from geo_model.model.settings import DISADVANTAGE, LINEAR_PROGRESSIVITY


def every_shared_option() -> argparse.ArgumentParser:
    """A parser taking every option the scripts share, as each script adds it."""
    parser = argparse.ArgumentParser()
    add_seeds_argument(parser, 7)
    add_workers_argument(parser)
    add_travel_arguments(parser)
    add_seat_arguments(parser, True)
    add_progressivity_argument(parser)
    add_disadvantage_argument(parser)
    add_intake_argument(parser, ["share", "dissimilarity"])
    add_region_arguments(parser)
    return parser


def test_every_shared_option_takes_its_default_and_its_own_value():
    parser = every_shared_option()

    # The defaults the scripts rely on, the intake's being the first measure.
    assert vars(parser.parse_args([])) == {
        "seeds": 7,
        "workers": 1,
        "years": NTS_YEARS,
        "circuity": CIRCUITY,
        "round_up_seats": True,
        "linear_progressivity": LINEAR_PROGRESSIVITY,
        "disadvantage": DISADVANTAGE,
        "intake": "share",
        "la": LAS,
        "merge": False,
    }
    # Every flag, its type and the --no- form of the switches.
    args = parser.parse_args(
        ["--seeds", "3", "--workers", "4", "--years", "2024", "2025"]
        + ["--circuity", "1.3", "--no-round-up-seats", "--linear-progressivity"]
        + ["--disadvantage", "decile", "--intake", "dissimilarity"]
        + ["--la", "Hampshire", "Southampton", "--merge"]
    )
    assert vars(args) == {
        "seeds": 3,
        "workers": 4,
        "years": [2024, 2025],
        "circuity": 1.3,
        "round_up_seats": False,
        "linear_progressivity": True,
        "disadvantage": "decile",
        "intake": "dissimilarity",
        "la": ["Hampshire", "Southampton"],
        "merge": True,
    }


@pytest.mark.parametrize(
    "argv", [["--disadvantage", "nobody"], ["--intake", "nothing"]]
)
def test_a_choice_outside_those_offered_is_rejected(argv):
    with pytest.raises(SystemExit):
        every_shared_option().parse_args(argv)
