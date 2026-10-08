"""Scenario 2: routes leave only the districts of lowest IDACI rank, as many
as there are schools, each school paired with one of them beyond the default
minimum distance so the longest route is as short as it can be, with no
condition on nearby schools, the route holding all of the school's route
seats.

The scenario's students are matched with and without routes over fresh
samples, and every plot `optimise` draws of a setting is drawn of it, as
scenario 1 draws them.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from geo_model.dissimilarity import DEFAULTS
from geo_model.scenario_1 import run

# The scenario's folder within each region's.
OUT_DIR = Path("scenario_2")

# A route only to a school beyond the default minimum distance, two miles, and
# no school scoring above Progress 8 infinity, so no district is served by its
# local schools and the local radius is inert. The districts routed are those
# of lowest IDACI rank, so the decile does not pick them, and each school's
# one route holds all of its route seats, so progressivity is inert too.
SCENARIO = replace(
    DEFAULTS,
    progressivity=0.0,
    max_local_p8=np.inf,
    bottleneck=True,
)


def main() -> None:
    run(
        SCENARIO,
        OUT_DIR,
        "Match fresh samples with and without scenario 2's routes, each "
        "school paired with one of the most deprived districts so the longest "
        "route is shortest, and plot and map the two against each other.",
    )


if __name__ == "__main__":
    main()
