"""Synthetic score frames shared by the tests of several folders, which
cannot import one another's modules."""

from dataclasses import asdict

import numpy as np
import pandas as pd

from geo_model.analysis.sweep import GRID
from geo_model.measures.travel import MODES
from geo_model.model.settings import DEFAULTS


def synthetic_frames():
    """Frames shaped like the ones `sweep` returns, covering every cell of GRID.

    A school without an FSM figure is included, so the plots are exercised on
    one that draws blank rather than failing.
    """
    rng = np.random.default_rng(0)
    cells = [
        {
            "parameter": parameter,
            "value": getattr(cell, parameter),
            "seed": seed,
            "scenario": scenario,
        }
        for parameter, grid in GRID.items()
        for cell in grid
        for seed in range(2)
        for scenario in ("with routes", "without routes")
    ]
    results = pd.DataFrame(
        {
            **cell,
            "dissimilarity": rng.uniform(0, 1),
            "n_matched": 900,
            "n_unmatched": 100,
            "n_disadvantaged": 300,
            "unassigned_disadvantaged": int(rng.integers(0, 100)),
        }
        for cell in cells
    )
    intake = pd.DataFrame(
        {
            **cell,
            "school": school,
            "disadvantaged": rng.integers(0, 50),
            "other": rng.integers(1, 200),
        }
        for cell in cells
        for school in ("Alpha School", "Beta School", "Gamma School")
    )
    # Each school's term of its matching's index, as `score_sample` gives it.
    totals = intake.groupby(["parameter", "value", "seed", "scenario"])[
        ["disadvantaged", "other"]
    ].transform("sum")
    intake["dissimilarity_term"] = (
        intake["disadvantaged"] / totals["disadvantaged"]
        - intake["other"] / totals["other"]
    )

    modes = pd.DataFrame(
        {**cell, "mode": mode, "students": rng.uniform(0, 500)}
        for cell in cells
        for mode in MODES
    )

    fsm = pd.Series({"Alpha School": 0.4, "Beta School": np.nan, "Gamma School": 0.1})
    return results, intake, modes, fsm


def default_intake(intake: pd.DataFrame) -> pd.DataFrame:
    """The school rows of the first parameter's default cell."""
    parameter = next(iter(GRID))
    return intake[
        (intake["parameter"] == parameter)
        & (intake["value"] == asdict(DEFAULTS)[parameter])
    ]


def route_rows(n_routes: int = 2) -> pd.DataFrame:
    """Route rows over two seeds: route r holds 4 + r seats, and in seed s
    carries r + s disadvantaged riders and 1 other."""
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "route": route,
                "LSOA21CD": f"E{route:08d}",
                "school": f"School {route}",
                "capacity": 4 + route,
                "disadvantaged": route + seed,
                "other": 1,
            }
            for seed in (0, 1)
            for route in range(n_routes)
        ]
    )


def local_rows() -> pd.DataFrame:
    """School rows over two seeds of three schools' local applicants and the
    disadvantaged and other of them seated."""
    counts = {
        # (seed, scenario): applicants, disadvantaged and other seated at
        # Alpha, Beta and Gamma.
        (0, "with routes"): ([4, 8, 0], [1, 3, 0], [2, 4, 0]),
        (1, "with routes"): ([6, 8, 0], [1, 3, 0], [2, 5, 0]),
        (0, "without routes"): ([4, 8, 0], [2, 4, 0], [2, 4, 0]),
        (1, "without routes"): ([6, 8, 0], [2, 4, 0], [2, 4, 0]),
    }
    return pd.DataFrame(
        {
            "seed": seed,
            "scenario": scenario,
            "school": school,
            "local_applicants": n,
            "local_disadvantaged": a,
            "local_other": b,
        }
        for (seed, scenario), columns in counts.items()
        for school, n, a, b in zip(("Alpha", "Beta", "Gamma"), *columns)
    )
