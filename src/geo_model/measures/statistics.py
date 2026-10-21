import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def routes_t_test(results: pd.DataFrame) -> pd.Series:
    """Paired t-test of the dissimilarity index with routes against without.

    Both scenarios of a seed match the same students, so the indices pair by
    seed and the test is taken over each seed's difference, with routes less
    without. It is undefined, t and p NaN, with fewer than two seeds, and
    with more where every seed's difference is 0, the only case in which
    `ttest_rel` returns NaN on finite indices.

    Args:
        results (pd.DataFrame): The scenario rows of one setting, carrying
        "seed", "scenario" and "dissimilarity".

    Returns:
        pd.Series: "seeds" (the pairs tested), "difference" (their mean
        difference), "t" and "p" (two-sided).

    Raises:
        ValueError: If a seed holds more than one row of a scenario, or a
        row of one scenario alone.
    """
    paired = results.pivot(index="seed", columns="scenario", values="dissimilarity")
    paired = paired[["with routes", "without routes"]]
    if paired.isna().any(axis=None):
        raise ValueError(
            "Seeds scored in one scenario alone have no pair to test: "
            + ", ".join(str(seed) for seed in paired.index[paired.isna().any(axis=1)])
        )
    if len(paired) < 2:
        t = p = np.nan
    else:
        test = ttest_rel(paired["with routes"], paired["without routes"])
        t, p = float(test.statistic), float(test.pvalue)
    return pd.Series(
        {
            "seeds": len(paired),
            "difference": (paired["with routes"] - paired["without routes"]).mean(),
            "t": t,
            "p": p,
        }
    )


def holm(p: pd.Series) -> pd.Series:
    """Holm's step-down adjustment of the p-values `p` for testing them all.

    With the m p-values defined in ascending order, the i-th is adjusted to
    the largest of (m - j + 1) p_j over j <= i, capped at 1. A NaN p-value,
    a test left undefined, stays NaN and is not counted in m.

    Args:
        p (pd.Series): P-values, uniquely indexed.

    Returns:
        pd.Series: The adjusted p-values, indexed as `p`.
    """
    defined = p.dropna().sort_values()
    adjusted = (defined * np.arange(len(defined), 0, -1)).cummax().clip(upper=1)
    return adjusted.reindex(p.index)


def t_test_label(test: pd.Series) -> str:
    """A test `routes_t_test` returns, in two lines: the seeds it is taken
    over, then its mean difference, t and p, or why it is undefined."""
    seeds = int(test["seeds"])
    if seeds < 2:
        result = "undefined: one seed"
    elif np.isnan(test["t"]):
        result = "undefined: routes change no seed's index"
    else:
        result = (
            f"mean difference {test['difference']:+.4f}, "
            f"t = {test['t']:.2f}, p = {test['p']:.2g}"
        )
    return f"Paired t-test over {seeds} seed{'' if seeds == 1 else 's'}\n{result}"
