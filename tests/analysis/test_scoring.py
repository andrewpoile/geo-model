import numpy as np
import pandas as pd
import pytest

from geo_model.analysis.scoring import district_rows, score_sample
from geo_model.measures.travel import MODES
from geo_model.model.settings import DEFAULTS, DISADVANTAGE
from geo_model.model.students import student_samples

# --------------------------------------------------------------------------
# score_sample: against the real data folder
# --------------------------------------------------------------------------


def test_score_sample_scores_both_scenarios_of_one_sample(
    areas, secondary, routes, shares, samples
):
    ((_, student_lsoa, disadvantaged, noise_seed, student_distance),) = samples

    rows, school_rows, mode_rows, route_rows = score_sample(
        student_distance,
        student_lsoa,
        areas,
        secondary,
        routes,
        disadvantaged,
        shares,
        noise_seed,
    )
    rows, intake, modes, riders = map(
        pd.DataFrame, (rows, school_rows, mode_rows, route_rows)
    )

    assert list(rows["scenario"]) == ["with routes", "without routes"]
    assert rows["dissimilarity"].between(0, 1).all()
    assert (rows["n_matched"] + rows["n_unmatched"] == len(student_distance)).all()

    # The disadvantaged group is the sample's, not the matching's, so it is
    # the same in both scenarios, and those of it without a place are a
    # subset of both the group and the unmatched.
    assert (rows["n_disadvantaged"] == disadvantaged.sum()).all()
    assert 0 < rows["n_disadvantaged"].iloc[0] < len(student_distance)
    assert (rows["unassigned_disadvantaged"] <= rows["n_unmatched"]).all()
    assert (rows["unassigned_disadvantaged"] <= rows["n_disadvantaged"]).all()

    # Every school of both scenarios has its intake counted, and the counts
    # add up to the students seated in that scenario.
    assert len(intake) == 2 * len(secondary)
    assert set(intake["school"]) == set(secondary["EstablishmentName"])
    seated = intake.groupby("scenario")[["disadvantaged", "other"]].sum().sum(axis=1)
    pd.testing.assert_series_equal(
        seated, rows.set_index("scenario")["n_matched"], check_names=False
    )

    # Every disadvantaged student is either seated somewhere or unassigned.
    placed = intake.groupby("scenario")["disadvantaged"].sum()
    indexed = rows.set_index("scenario")
    pd.testing.assert_series_equal(
        placed + indexed["unassigned_disadvantaged"],
        indexed["n_disadvantaged"],
        check_names=False,
    )

    # A school's local students seated are among its local applicants and
    # its students seated of the same group, and some school seats some.
    seated_locals = intake["local_disadvantaged"] + intake["local_other"]
    assert (seated_locals <= intake["local_applicants"]).all()
    assert (intake["local_disadvantaged"] <= intake["disadvantaged"]).all()
    assert (intake["local_other"] <= intake["other"]).all()
    assert seated_locals.sum() > 0

    # The schools' terms are the index taken apart: they cancel out, and
    # half their absolute sum is the scenario's index.
    terms = intake.groupby("scenario")["dissimilarity_term"]
    assert terms.sum().abs().max() < 1e-12
    pd.testing.assert_series_equal(
        0.5 * terms.apply(lambda t: t.abs().sum()),
        indexed["dissimilarity"],
        check_names=False,
    )

    # Every seated student travels by exactly one expected mode, and only
    # the routed scenario seats anyone on a route.
    assert len(modes) == 2 * len(MODES)
    pd.testing.assert_series_equal(
        modes.groupby("scenario")["students"].sum(),
        rows.set_index("scenario")["n_matched"].astype(float),
        check_names=False,
    )
    on_route = modes[modes["mode"] == "route"].set_index("scenario")["students"]
    assert on_route["with routes"] > 0
    assert on_route["without routes"] == 0

    # Every route of the routed matching has its riders counted, none more
    # than its seats, and together they are the students seated on a route.
    pd.testing.assert_series_equal(
        riders["route"], routes["route_id"], check_names=False, check_dtype=False
    )
    np.testing.assert_array_equal(riders["capacity"], routes["capacity"])
    used = riders["disadvantaged"] + riders["other"]
    assert (used <= riders["capacity"]).all()
    assert used.sum() == on_route["with routes"]
    assert (
        riders["disadvantaged"].sum()
        <= intake.groupby("scenario")["disadvantaged"].sum()["with routes"]
    )


def test_score_sample_ranks_priorities_on_the_rule_and_discount_given(
    areas, secondary, routes, shares, samples
):
    ((_, student_lsoa, disadvantaged, noise_seed, student_distance),) = samples

    def scored(priority, priority_discount):
        return score_sample(
            student_distance,
            student_lsoa,
            areas,
            secondary,
            routes,
            disadvantaged,
            shares,
            noise_seed,
            priority=priority,
            priority_discount=priority_discount,
        )

    # The discount moves routed bundles under "distance" and none under
    # "bracket", so a rule or a discount left behind on the way down to the
    # ranking fails one of these.
    assert scored("distance", 0.0) != scored("distance", 1.0)
    assert scored("bracket", 0.0) == scored("bracket", 1.0)


# --------------------------------------------------------------------------
# district_rows: against the real data folder
# --------------------------------------------------------------------------


def test_district_rows_count_every_seeds_own_students_by_district(areas, to_school):
    samples = list(student_samples(areas, 2, to_school))

    rows = district_rows(samples, areas, DEFAULTS.decile, DISADVANTAGE)

    assert list(rows.columns[:2]) == ["seed", "LSOA21CD"]
    for seed, (_, student_lsoa, drawn, _, _) in enumerate(samples):
        seed_rows = rows[rows["seed"] == seed]
        np.testing.assert_array_equal(seed_rows["LSOA21CD"], areas["LSOA21CD"])
        np.testing.assert_array_equal(
            seed_rows["cohort"], np.bincount(student_lsoa, minlength=len(areas))
        )
        np.testing.assert_array_equal(
            seed_rows["disadvantaged"],
            np.bincount(student_lsoa[drawn], minlength=len(areas)),
        )
        assert seed_rows["share_of_disadvantaged"].sum() == pytest.approx(1)
        assert seed_rows["share_of_advantaged"].sum() == pytest.approx(1)
    # Each seed draws its cohorts afresh, so the seeds' rows differ.
    assert not np.array_equal(
        rows.loc[rows["seed"] == 0, "cohort"], rows.loc[rows["seed"] == 1, "cohort"]
    )
