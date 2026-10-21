import argparse

from geo_model.data.sources import LAS
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
