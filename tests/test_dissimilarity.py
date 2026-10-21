import pandas as pd
import pytest

from geo_model import dissimilarity as ds
from geo_model.analysis.scoring import settings_matchings
from geo_model.measures.segregation import dissimilarity_index
from geo_model.model.settings import DEFAULTS, DISADVANTAGE_CHOICES
from geo_model.model.students import disadvantaged_group

# --------------------------------------------------------------------------
# run: against the real data folder
# --------------------------------------------------------------------------


def test_run_scores_both_scenarios_of_every_seed(areas, secondary, tmp_path):
    results = ds.run(1, areas, secondary, tmp_path)

    assert list(results["scenario"]) == ["with routes", "without routes"]
    assert results["dissimilarity"].between(0, 1).all()
    # Both scenarios match the same student sample.
    assert results["n_matched"].add(results["n_unmatched"]).nunique() == 1
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / ds.RESULTS_CSV), results)


# --------------------------------------------------------------------------
# settings_matchings: against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("disadvantage", DISADVANTAGE_CHOICES)
def test_settings_matchings_are_the_ones_the_sweep_scores_at_the_defaults(
    samples, areas, secondary, shares, disadvantage
):
    matchings = settings_matchings(
        DEFAULTS, samples[0], areas, secondary, disadvantage=disadvantage
    )
    rows, _, _, route_rows = ds.score_settings(
        DEFAULTS,
        samples,
        areas,
        secondary,
        shares,
        disadvantage=disadvantage,
    )
    # The route rows are kept only when asked for.
    assert route_rows == []

    _, student_lsoa, drawn, _ = samples[0]
    disadvantaged = disadvantaged_group(
        student_lsoa, drawn, areas, DEFAULTS.decile, disadvantage
    )
    assert [row["scenario"] for row in rows] == list(matchings)
    for row in rows:
        matched_school = matchings[row["scenario"]][:, 0]
        assert (
            dissimilarity_index(matched_school, disadvantaged, len(secondary))
            == row["dissimilarity"]
        )
        assert (matched_school >= 0).sum() == row["n_matched"]
    assert (matchings["with routes"][:, 1] >= 0).any()
    assert (matchings["without routes"][:, 1] < 0).all()
