import argparse

from geo_model.data.sources import LAS
from geo_model.data.travel import NTS_YEARS
from geo_model.measures.travel import CIRCUITY
from geo_model.model.settings import (
    DISADVANTAGE,
    DISADVANTAGE_CHOICES,
    LINEAR_PROGRESSIVITY,
)
from geo_model.output import OUTPUT_DIR


def add_region_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options choosing the regions an entry point simulates.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.
    """
    parser.add_argument(
        "--la",
        nargs="+",
        default=LAS,
        metavar="NAME",
        help="local authorities to simulate, as the school register names "
        "them (e.g. Southampton, Hampshire); each is simulated on its own, "
        f"one after another, into {OUTPUT_DIR}/<NAME>",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="simulate the named authorities that border each other as one "
        f"region, into {OUTPUT_DIR}/<NAME>+<NAME>; one bordering none of the "
        "others is still simulated on its own",
    )


def add_seeds_argument(parser: argparse.ArgumentParser, default: int) -> None:
    """Add the option of how many student samples an entry point draws.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.

        default (int): Samples drawn unless the option says otherwise.
    """
    parser.add_argument("--seeds", type=int, default=default)


def add_workers_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option of how many processes an entry point scores settings in.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.
    """
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes to score settings in, one setting per task, see `sweep`",
    )


def add_travel_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of the NTS mode shares a matching's travel is read by.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.
    """
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=NTS_YEARS,
        help="NTS years to take mode shares from, pooled when more than one",
    )
    parser.add_argument(
        "--circuity",
        type=float,
        default=CIRCUITY,
        help="road distance per straight-line metre, applied before banding",
    )


def add_seat_arguments(parser: argparse.ArgumentParser, round_up: bool) -> None:
    """Add the option of how each school's route seats are rounded.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.

        round_up (bool): Whether seats are rounded up unless the option says
        otherwise.
    """
    parser.add_argument(
        "--round-up-seats",
        action=argparse.BooleanOptionalAction,
        default=round_up,
        help="round each school's route seats up rather than to the nearest "
        "seat, before they are split between its routes",
    )


def add_progressivity_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option of the profile route seats lean towards deprivation by.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.
    """
    parser.add_argument(
        "--linear-progressivity",
        action=argparse.BooleanOptionalAction,
        default=LINEAR_PROGRESSIVITY,
        help="weight route seats by the linear profile (T + 1 - D) / T of "
        "each district's IDACI decile D rather than by 1 / D",
    )


def add_disadvantage_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option of who is disadvantaged and who may ride a route.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.
    """
    parser.add_argument(
        "--disadvantage",
        choices=DISADVANTAGE_CHOICES,
        default=DISADVANTAGE,
        help="score: each LSOA's students are drawn disadvantaged in proportion "
        "to its IDACI score, and routes carry only them; score-open: drawn the "
        "same way, but routes carry every student of their district; decile: "
        "every student of an LSOA at or below the decile is disadvantaged and "
        "rides, the model as it was before",
    )


def add_intake_argument(parser: argparse.ArgumentParser, measures: list[str]) -> None:
    """Add the option of what the intake figures show of each school.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.

        measures (list[str]): The measures they can show, the keys of
        INTAKE_MEASURES, passed in so cli.py imports no plotting library. The
        first is the default, so the default is always one of the choices.
    """
    parser.add_argument(
        "--intake",
        choices=measures,
        default=measures[0],
        help="what the intake figures show of each school: its term of the "
        "dissimilarity index, or the disadvantaged share of its intake",
    )
