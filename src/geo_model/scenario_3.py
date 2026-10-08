"""Scenario 3: routes leave only the districts holding the most disadvantaged
students, as many as there are schools, which take turns from the most
disadvantaged down, each paired with the school offering the most places that
no district before it took, beyond the default minimum distance, with no
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
OUT_DIR = Path("scenario_3")

# A route only to a school beyond the default minimum distance, two miles, and
# no school scoring above Progress 8 infinity, so no district is served by its
# local schools and the local radius is inert. The districts routed are those
# holding the most disadvantaged students, so the decile picks them only where
# it picks who is disadvantaged, and each school's one route holds all of its
# route seats, so progressivity is inert too.
SCENARIO = replace(
    DEFAULTS,
    progressivity=0.0,
    max_local_p8=np.inf,
    largest_first=True,
)


def main() -> None:
    run(
        SCENARIO,
        OUT_DIR,
        "Match fresh samples with and without scenario 3's routes, the "
        "districts holding the most disadvantaged students each paired in turn "
        "with the free school offering the most places, and plot and map the "
        "two against each other.",
    )


if __name__ == "__main__":
    main()
