from collections.abc import Callable, Iterable, Iterator
from dataclasses import fields, replace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

from geo_model.analysis.sweep import GRID, swept
from geo_model.measures.travel import mode_change
from geo_model.model.settings import DEFAULTS, Settings

# Every swept parameter. Which of them a search may vary depends on the
# choice of DISADVANTAGE, as `searchable` gives it.
OPTIMISABLE = list(GRID)
# The settings a transport authority holds. The preference weights describe
# the students, so they hold DEFAULTS unless named.
LEVERS = [
    "min_distance",
    "capacity_scale",
    "progressivity",
    "max_local_p8",
    "local_radius",
]

Scored = tuple[list[dict], list[dict], list[dict], list[dict]]


class Objective(NamedTuple):
    """A measure of the frames `score_settings` returns, and which way is better."""

    score: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], float]
    maximise: bool
    label: str


def routed_dissimilarity(
    results: pd.DataFrame, schools: pd.DataFrame, modes: pd.DataFrame
) -> float:
    """The dissimilarity index with routes, the mean over seeds."""
    routed = results["scenario"] == "with routes"
    return float(results.loc[routed, "dissimilarity"].mean())


def car_displacement(
    results: pd.DataFrame, schools: pd.DataFrame, modes: pd.DataFrame
) -> float:
    """The expected students the routes take out of cars, the mean over seeds.

    Car students without routes less those with them, so positive where the
    routes displace car travel.
    """
    change = mode_change(modes)
    return -float(change.loc[change["mode"] == "car", "change"].mean())


OBJECTIVES = {
    "dissimilarity": Objective(
        routed_dissimilarity, False, "Dissimilarity index with routes (lower is better)"
    ),
    "car_displacement": Objective(
        car_displacement, True, "Students displaced from car travel (higher is better)"
    ),
}
DEFAULT_OBJECTIVES = list(OBJECTIVES)


def searchable(disadvantage: str) -> list[str]:
    """The parameters of OPTIMISABLE a search may vary under `disadvantage`.

    Those `swept` runs under it, less the decile under "decile", where it sets
    the group the index measures as well as the districts routes leave, so
    searching it would move the measurement rather than the matching.

    Args:
        disadvantage (str): One of DISADVANTAGE_CHOICES.

    Returns:
        list[str]: Keys of OPTIMISABLE, in its order.
    """
    return [
        parameter
        for parameter in swept(GRID, disadvantage)
        if not (parameter == "decile" and disadvantage == "decile")
    ]


def cast(values: dict[str, Any]) -> dict[str, Any]:
    """`values` as the Settings fields they set: a field annotated int, the
    decile, rounded to the nearest whole number, and every other value a float.

    The search runs over the reals, so a searched decile is rounded here, and
    the deciles at the ends of its span take half the span the others do. The
    annotation is read rather than the default, since some float fields
    default to a whole number of metres.

    Args:
        values (dict[str, Any]): A value for each of some fields of Settings.

    Returns:
        dict[str, Any]: The same fields, each value cast.
    """
    types = {field.name: field.type for field in fields(Settings)}
    return {
        name: round(value) if types[name] is int else float(value)
        for name, value in values.items()
    }


def bounds(parameter: str) -> tuple[float, float]:
    """The span `GRID` sweeps `parameter` over, the range it is searched in.

    The comments on `GRID` say why each end is valid with every other
    parameter at its default. A corner of several spans can still leave no
    route, which `score_feasible` reports as infeasible.

    Args:
        parameter (str): One of OPTIMISABLE.

    Returns:
        tuple[float, float]: The lowest and highest value swept.
    """
    if parameter not in OPTIMISABLE:
        raise ValueError(f"{parameter} is not optimisable; choose from {OPTIMISABLE}.")
    values = [getattr(settings, parameter) for settings in GRID[parameter]]
    return float(min(values)), float(max(values))


def objective_values(scored: Scored, objectives: list[str]) -> dict[str, float]:
    """Each of `objectives`, keys of OBJECTIVES, measured on `scored`, the
    rows `score_settings` returns."""
    results, schools, modes = (pd.DataFrame(rows) for rows in scored[:3])
    return {
        name: OBJECTIVES[name].score(results, schools, modes) for name in objectives
    }


def minimised(frame: pd.DataFrame, objectives: list[str]) -> np.ndarray:
    """The `objectives` columns of `frame` as pymoo minimises them, the
    maximised ones negated, shape (len(frame), len(objectives))."""
    return np.column_stack(
        [
            -frame[name] if OBJECTIVES[name].maximise else frame[name]
            for name in objectives
        ]
    )


class Pipeline(Problem):
    """The pipeline as a pymoo problem, one candidate setting per row.

    pymoo minimises, so a maximised objective is negated on the way in, and a
    setting leaving no route violates the one constraint. Its objectives go in
    as +inf, which pymoo never ranks, since it compares constraint violation
    first. Every candidate scored is kept in `evaluations`, its objectives as
    measured, NaN where infeasible.

    Args:
        parameters (list[str]): Keys of OPTIMISABLE, one per variable.

        objectives (list[str]): Keys of OBJECTIVES, one per objective.

        score (Callable[[Settings], Scored | None]): `score_feasible` with
        every argument but the settings bound.

        mapper (Callable[..., Iterator]): Maps `score` over the candidates of
        a generation, in order: `map`, or a process pool's.
    """

    def __init__(
        self,
        parameters: list[str],
        objectives: list[str],
        score: Callable[[Settings], Scored | None],
        mapper: Callable[[Callable, Iterable], Iterator],
    ) -> None:
        low, high = zip(*(bounds(parameter) for parameter in parameters))
        super().__init__(
            n_var=len(parameters),
            n_obj=len(objectives),
            n_ieq_constr=1,
            xl=np.array(low),
            xu=np.array(high),
        )
        self.parameters = parameters
        self.objectives = objectives
        self.score = score
        self.mapper = mapper
        self.evaluations: list[dict] = []
        self.generation = 0

    def _evaluate(self, x: np.ndarray, out: dict, *args: Any, **kwargs: Any) -> None:
        self.generation += 1
        print(f"Generation {self.generation}: scoring {len(x)} settings.")
        candidates = [cast(dict(zip(self.parameters, row))) for row in x]
        scored = self.mapper(
            self.score, [replace(DEFAULTS, **candidate) for candidate in candidates]
        )
        rows = [
            {
                "generation": self.generation,
                **candidate,
                **(
                    dict.fromkeys(self.objectives, np.nan)
                    if result is None
                    else objective_values(result, self.objectives)
                ),
                "feasible": result is not None,
            }
            for candidate, result in zip(candidates, scored)
        ]
        self.evaluations += rows
        batch = pd.DataFrame(rows)
        feasible = batch["feasible"].to_numpy()
        out["F"] = np.where(
            feasible[:, None], minimised(batch, self.objectives), np.inf
        )
        out["G"] = np.where(feasible, 0.0, 1.0)[:, None]


def compromise(front: pd.DataFrame, objectives: list[str]) -> pd.Series:
    """The front setting nearest the ideal point, every objective rescaled to
    [0, 1] over the front, 0 at its best value and 1 at its worst.

    The ideal holds the best value of every objective at once, which no
    setting reaches while they conflict, so the nearest setting gives up the
    least of each in units of the front's own spread. An objective the whole
    front ties on has no spread and plays no part. A tie in distance goes to
    the earlier row.

    Args:
        front (pd.DataFrame): As returned by `optimise`.

        objectives (list[str]): The keys of OBJECTIVES the front was found for.

    Returns:
        pd.Series: The row of `front` chosen.
    """
    values = minimised(front, objectives)
    low, spread = values.min(axis=0), np.ptp(values, axis=0)
    scaled = np.divide(
        values - low, spread, out=np.zeros_like(values), where=spread > 0
    )
    return front.iloc[int(np.argmin(np.linalg.norm(scaled, axis=1)))]
