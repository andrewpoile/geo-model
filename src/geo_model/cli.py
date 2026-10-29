import argparse
from dataclasses import fields

from geo_model.data.network import DISTANCE, DISTANCE_CHOICES
from geo_model.data.sources import LAS
from geo_model.data.travel import NTS_YEARS
from geo_model.measures.travel import CIRCUITY
from geo_model.model.settings import (
    DISADVANTAGE,
    DISADVANTAGE_CHOICES,
    LINEAR_PROGRESSIVITY,
    PRIORITY,
    PRIORITY_CHOICES,
    PRIORITY_DISCOUNT,
    Settings,
)
from geo_model.output import ANON_DIR, OUTPUT_DIR


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


def add_anon_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option of writing a region's outputs with nothing naming it.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.
    """
    parser.add_argument(
        "--anon",
        action="store_true",
        help="number the schools, districts and authorities rather than name "
        "them, the districts by IDACI rank, and draw no map, since any map of "
        "the districts' true shapes gives the region away, nor the districts' "
        "IDACI deciles and scores or the register's FSM figures, which the "
        "national files tie to the region, so no plot or table names the "
        "region; the results are the named run's, written to "
        f"{OUTPUT_DIR}/<NAME>/{ANON_DIR}",
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
        help="road distance per metre of trip length, applied before banding",
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


def add_priority_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option of the rule school priorities follow.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.
    """
    parser.add_argument(
        "--priority",
        choices=PRIORITY_CHOICES,
        default=PRIORITY,
        help="distance: each school ranks every student by distance, a routed "
        f"student's taken down by the priority discount, {PRIORITY_DISCOUNT} "
        "unless --priority-discount, the sweep or the search sets it; bracket: "
        "every routed student and every student of the school's own district "
        "first, everyone else after, each by distance, the model as it was "
        f"before, written to {OUTPUT_DIR}/<NAME>/bracket",
    )


def add_distance_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option of how every distance is measured.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.
    """
    parser.add_argument(
        "--distance",
        choices=DISTANCE_CHOICES,
        default=DISTANCE,
        help="network: along the walking network of the OS Multi-modal Routing "
        "Network; euclidean: in a straight line, the model as it was before, "
        f"written to {OUTPUT_DIR}/<NAME>/euclidean",
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


def add_settings_arguments(parser: argparse.ArgumentParser, settings: Settings) -> None:
    """Add an option for every field of Settings carrying help, each
    defaulting to its value in `settings`, so any setting can be run from the
    command line. The rules carry none, their options being the add_*
    functions above.

    Args:
        parser (argparse.ArgumentParser): The entry point's parser.

        settings (Settings): The values the options default to: DEFAULTS, or
        a preset's.
    """
    for field in fields(Settings):
        if "help" not in field.metadata:
            continue
        flag = "--" + field.name.replace("_", "-")
        default = getattr(settings, field.name)
        if field.type is bool:
            parser.add_argument(
                flag,
                action=argparse.BooleanOptionalAction,
                default=default,
                help=field.metadata["help"],
            )
        else:
            parser.add_argument(
                flag,
                type=int if field.type is int else float,
                default=default,
                help=field.metadata["help"],
            )
