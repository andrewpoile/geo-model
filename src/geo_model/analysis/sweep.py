from dataclasses import asdict, replace

import pandas as pd

from geo_model.measures.statistics import holm, routes_t_test
from geo_model.model.settings import DEFAULTS, Settings

# One-at-a-time grids: a parameter runs over its values while the others hold
# DEFAULTS, so every grid carries its own default. The spans were chosen on
# Southampton, whose figures the comments give; in another region a cell can
# leave no route, which stops the sweep with EmptyRouteSet.
GRID = {
    # The decile sets the districts routes are built from and T of the
    # progressive weight. Under DISADVANTAGE "decile" it also sets the group the
    # index and the unassigned panel measure, so there a step along it moves
    # the treatment and the measurement together, which is what makes decile 2
    # the only setting where routes leave more disadvantaged students
    # unassigned, taken apart in agent_docs/decile_anomaly.md. There decile 10
    # would make every student disadvantaged and the index undefined, so the
    # grid stops at 9 under every choice, and every choice sweeps one grid.
    "decile": [replace(DEFAULTS, decile=d) for d in range(1, 10)],
    # Whole miles in metres. In Southampton every district lies within 9.9km
    # of every school, so the route set stays non-empty throughout.
    "min_distance": [
        replace(DEFAULTS, min_distance=m) for m in (0, 1609, 3218, 4828, 6437, 8047)
    ],
    # At 1 a school's routes hold the disadvantaged students' fair share of
    # the places it offers. In Southampton 1254 of the cohort of 2918 are
    # disadvantaged under DISADVANTAGE "score", so from 2918 / 1254 = 2.33
    # every school's routes hold at least the places it offers.
    "capacity_scale": [
        replace(DEFAULTS, capacity_scale=k)
        for k in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0, 15.0, 20.0, 25.0)
    ],
    # From every district on its fair share alone to decile 1 weighted T^3
    # times the threshold decile T, each school's seats held fixed throughout.
    "progressivity": [replace(DEFAULTS, progressivity=p / 4) for p in range(13)],
    # Quarter steps over the spread of P8MEA, which runs from -0.99 to 0.82 over
    # Southampton's 12 secondary schools: above 0.82 no school is above the
    # threshold, so the condition is inert and the route set is whole. At LOCAL_RADIUS no
    # value empties it, since a district with no school inside the radius at all
    # keeps its routes however low the threshold.
    "max_local_p8": [replace(DEFAULTS, max_local_p8=p / 4) for p in range(-2, 5)],
    # Half miles in metres. Every Southampton district at or below the default
    # decile has a school above MAX_LOCAL_P8 within 5.6km, so there the route
    # set empties from that radius.
    "local_radius": [
        replace(DEFAULTS, local_radius=r)
        for r in (0, 805, 1609, 2414, 3218, 4023, 4828)
    ],
    "performance_weight": [
        replace(DEFAULTS, performance_weight=w / 10) for w in range(11)
    ],
    "disadvantaged_performance_weight": [
        replace(DEFAULTS, disadvantaged_performance_weight=w / 10) for w in range(11)
    ],
    # Inert unless a student who is not disadvantaged rides, so swept only
    # where one does: see `swept`.
    "route_discount": [replace(DEFAULTS, route_discount=d / 10) for d in range(11)],
    "disadvantaged_route_discount": [
        replace(DEFAULTS, disadvantaged_route_discount=d / 10) for d in range(11)
    ],
}


def swept(grid: dict[str, list[Settings]], disadvantage: str) -> list[str]:
    """The parameters of `grid` the sweep runs under `disadvantage`.

    Only under "score-open" does a student who is not disadvantaged ride a
    route: under "score" the disadvantaged alone ride, and under "decile"
    every student of a route's district is disadvantaged. Elsewhere the other
    students' route discount is inert, so it is swept under "score-open" alone.

    Args:
        grid (dict[str, list[Settings]]): The cells of every parameter, as
        GRID holds them.

        disadvantage (str): One of DISADVANTAGE_CHOICES.

    Returns:
        list[str]: Keys of `grid`, in its order.
    """
    return [p for p in grid if p != "route_discount" or disadvantage == "score-open"]


def sweep_tests(results: pd.DataFrame) -> pd.DataFrame:
    """The paired t-test of the index with routes against without in every
    cell, its p-value Holm-corrected over every cell `results` holds.

    The family is the whole frame rather than the parameters drawn, so every
    figure drawn from the same frame marks a cell alike. Every parameter's
    default cell scores the same matchings yet counts as a test of its own,
    which only makes the correction more conservative.

    Args:
        results (pd.DataFrame): The scenario rows `sweep` returns.

    Returns:
        pd.DataFrame: One row per (parameter, value), so indexed, with the
        columns `routes_t_test` returns and "p_holm", the corrected p.
    """
    tests = results.groupby(["parameter", "value"]).apply(routes_t_test)
    return tests.assign(p_holm=holm(tests["p"]))


def default_cell(rows: pd.DataFrame, grid: dict[str, list[Settings]]) -> pd.DataFrame:
    """The rows of the first parameter's default cell.

    Every parameter's default cell scores the same matchings, so the first
    parameter's stands for them all: the rows of the default settings.

    Args:
        rows (pd.DataFrame): Rows `sweep` returns.

        grid (dict[str, list[Settings]]): The cells `rows` were scored
        over, as GRID holds them.

    Returns:
        pd.DataFrame: The rows of that cell.

    Raises:
        ValueError: If the rows hold no default cell of the first parameter.
    """
    parameter = next(iter(grid))
    default = rows[
        (rows["parameter"] == parameter)
        & (rows["value"] == asdict(DEFAULTS)[parameter])
    ]
    if default.empty:
        raise ValueError(f"The rows hold no default cell of {parameter}.")
    return default
