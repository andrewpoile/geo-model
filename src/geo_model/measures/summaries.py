import pandas as pd

from geo_model.model.settings import SCENARIOS


def unassigned_rows(results: pd.DataFrame) -> pd.DataFrame:
    """The students each group is left without a place, one row per group.

    A student unseated by the matching is scored by neither the index nor the
    intake panels, both of which count seated students alone, so the burden of
    going unplaced is only visible here.

    Args:
        results (pd.DataFrame): The scenario rows `sweep` returns, carrying
        "n_unmatched" and "unassigned_disadvantaged".

    Returns:
        pd.DataFrame: Two rows per scenario row, carrying "unassigned" (that
        group's students with no place) and "series", naming the group and
        the scenario, as "disadvantaged, with routes".
    """
    unmatched, disadvantaged = (
        results["n_unmatched"],
        results["unassigned_disadvantaged"],
    )
    rows = pd.concat(
        [
            results.assign(group="disadvantaged", unassigned=disadvantaged),
            results.assign(group="other", unassigned=unmatched - disadvantaged),
        ]
    )
    return rows.assign(series=rows["group"] + ", " + rows["scenario"])


def route_utilisation(routes: pd.DataFrame) -> pd.DataFrame:
    """Each route's seats and the riders it carries, the mean over seeds.

    Args:
        routes (pd.DataFrame): The route rows of one setting, as
        `score_settings` returns them with `utilisation`.

    Returns:
        pd.DataFrame: One row per route, with "route", "LSOA21CD", "school"
        and "capacity", the mean "disadvantaged", "other" and "used"
        riders, the "unused" seats (capacity less those used) and
        "utilisation" (used over capacity), the routes with the most seats
        first.
    """
    per_route = (
        routes.assign(used=routes["disadvantaged"] + routes["other"])
        .groupby(["route", "LSOA21CD", "school", "capacity"], as_index=False)[
            ["disadvantaged", "other", "used"]
        ]
        .mean()
    )
    # A route is only built holding a seat, so no capacity is 0.
    return per_route.assign(
        unused=per_route["capacity"] - per_route["used"],
        utilisation=per_route["used"] / per_route["capacity"],
    ).sort_values(["capacity", "route"], ascending=[False, True], ignore_index=True)


def utilisation_summary(routes: pd.DataFrame) -> str:
    """The seats every route provides and the share of them used, as a line
    to print, from the route rows of one setting."""
    per_route = route_utilisation(routes)
    seats, used = per_route["capacity"].sum(), per_route["used"].sum()
    return (
        f"{len(per_route)} routes provide {seats} seats, {used:.1f} of them "
        f"used, the mean over seeds ({used / seats:.1%})."
    )


def local_acceptance(schools: pd.DataFrame) -> pd.DataFrame:
    """Each school's local applicants and those of them it accepts, the mean
    over seeds, with routes and without.

    Args:
        schools (pd.DataFrame): The school rows of one setting, as
        `score_settings` returns them.

    Returns:
        pd.DataFrame: One row per (school, scenario), with "school",
        "scenario", the mean "local_applicants" and "local_disadvantaged"
        and "local_other" accepted, the "accepted" of both groups, the
        "rejected" (applicants less those accepted) and "acceptance"
        (accepted over applicants, NaN where none applied). The schools with
        the most local applicants without routes come first, each with
        routes then without.
    """
    # Counted per seed before the mean, so a school rejecting nobody shows
    # exactly 0 rather than a rounding error of the means.
    accepted = schools["local_disadvantaged"] + schools["local_other"]
    per_school = (
        schools.assign(
            accepted=accepted, rejected=schools["local_applicants"] - accepted
        )
        .groupby(["school", "scenario"])[
            [
                "local_applicants",
                "local_disadvantaged",
                "local_other",
                "accepted",
                "rejected",
            ]
        ]
        .mean()
    )
    per_school = per_school.assign(
        acceptance=per_school["accepted"] / per_school["local_applicants"]
    )
    order = (
        per_school.xs("without routes", level="scenario")
        .reset_index()
        .sort_values(["local_applicants", "school"], ascending=[False, True])
    )
    return per_school.reindex(
        pd.MultiIndex.from_product(
            [order["school"], list(SCENARIOS)], names=["school", "scenario"]
        )
    ).reset_index()


def local_summary(schools: pd.DataFrame) -> str:
    """The local students who applied to their own school and the share of
    them accepted, with routes and without, as lines to print, from the
    school rows of one setting."""
    totals = (
        local_acceptance(schools)
        .groupby("scenario")[["local_applicants", "accepted"]]
        .sum()
        .loc[list(SCENARIOS)]
    )
    return "\n".join(
        f"{scenario.capitalize()}: {applied:.1f} local students applied to "
        f"their own school, {accepted:.1f} of them accepted, the mean over "
        f"seeds ({accepted / applied:.1%})."
        for scenario, applied, accepted in zip(
            SCENARIOS, totals["local_applicants"], totals["accepted"]
        )
    )


def local_difference(schools: pd.DataFrame) -> pd.DataFrame:
    """Each school's local students accepted with routes less those without,
    per seed.

    Both scenarios of a seed match the same students, so the counts pair by
    seed, as `routes_t_test` pairs the index.

    Args:
        schools (pd.DataFrame): The school rows of one setting, as
        `score_settings` returns them.

    Returns:
        pd.DataFrame: One row per (seed, school), with "seed", "school" and
        "difference".

    Raises:
        ValueError: If a seed holds a school's row of one scenario alone.
    """
    paired = schools.assign(
        accepted=schools["local_disadvantaged"] + schools["local_other"]
    ).pivot(index=["seed", "school"], columns="scenario", values="accepted")
    if paired.isna().any(axis=None):
        raise ValueError(
            "Schools scored in one scenario alone have no pair, at (seed, "
            "school): "
            + ", ".join(str(pair) for pair in paired.index[paired.isna().any(axis=1)])
        )
    return (
        (paired["with routes"] - paired["without routes"])
        .rename("difference")
        .reset_index()
    )
