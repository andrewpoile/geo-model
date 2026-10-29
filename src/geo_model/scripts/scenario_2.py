"""Scenario 2: routes leave only the most deprived districts by IDACI rank
that can each have a school of their own beyond the default minimum distance,
as many as there are schools, a district that cannot giving way to the next
in rank. Each school is paired with one of them so the longest route is as
short as it can be, with no condition on nearby schools, the route holding
all of the school's route seats.

The scenario is a preset of `run`: its students are matched with and
without routes over fresh samples, and written, plotted and reported as
`run` does, every option of `run` defaulting to the scenario's value.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from geo_model.model.settings import DEFAULTS
from geo_model.scripts import run

# The scenario's folder within each region's.
OUT_DIR = Path("scenario_2")

N_SEEDS = 100

# A route only to a school beyond the default minimum distance, two miles, and
# no school scoring above Progress 8 infinity, so no district is served by its
# local schools and the local radius is inert. The districts routed are taken
# in IDACI rank order once the minimum distance is applied, so the decile does
# not pick them, and each school's one route holds all of its route seats, so
# progressivity is inert too. Seats are rounded up, as scenario 1 rounds them,
# so a school's one route keeps a seat where the school's route seats fall
# below half of one.
SCENARIO = replace(
    DEFAULTS,
    progressivity=0.0,
    max_local_p8=np.inf,
    bottleneck=True,
    round_up_seats=True,
)


def main() -> None:
    run.main(
        SCENARIO,
        OUT_DIR,
        "Match fresh samples with and without scenario 2's routes, each "
        "school paired with one of the most deprived districts so the longest "
        "route is shortest, and plot and map the two against each other.",
        seeds=N_SEEDS,
    )


if __name__ == "__main__":
    main()
