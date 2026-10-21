"""Scenario 1: every school offers a single route, its shortest, to the
route-eligible districts beyond the default minimum distance, with no
condition on nearby schools, holding all of the school's route seats.

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
OUT_DIR = Path("scenario_1")

N_SEEDS = 100

# A route only to a school beyond the default minimum distance, two miles, and
# no school scoring above Progress 8 infinity, so no district is served by its
# local schools and the local radius is inert. Each school keeps only its route
# to the nearest district beyond the minimum distance, which holds all of the
# school's route seats, so progressivity is inert too.
SCENARIO = replace(
    DEFAULTS,
    progressivity=0.0,
    max_local_p8=np.inf,
    shortest_only=True,
)
# Seats are rounded up, so a school's shortest route keeps a seat where the
# school's route seats fall below half of one.
ROUND_UP = True


def main() -> None:
    run.main(
        SCENARIO,
        OUT_DIR,
        "Match fresh samples with and without scenario 1's routes, each "
        "school's shortest, and plot and map the two against each other.",
        seeds=N_SEEDS,
        round_up=ROUND_UP,
    )


if __name__ == "__main__":
    main()
