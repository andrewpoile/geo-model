import numpy as np
import pandas as pd

from geo_model.data import sources

# NTS0614a's trip-length bands, and its main modes mapped to the model's
# names for them.
NTS_BANDS = [
    "Under 1 mile",
    "1 to under 2 miles",
    "2 to under 5 miles",
    "5 miles and over",
]
NTS_MODES = {
    "Walk (%) [note 2]": "walk",
    "Pedal cycle (%) [note 3]": "cycle",
    "Car or van (%)": "car",
    "Bus (%) [note 4]": "bus",
    "Other transport (%) [note 5]": "other",
}
NTS_TRIPS = "Unweighted sample size: trips (thousands) (number)"
# The latest survey year, the default the shares are taken from.
NTS_YEARS = [2025]


def load_nts_mode_shares(
    years: list[int] = NTS_YEARS, age: str = "11 to 16"
) -> pd.DataFrame:
    """Import the NTS share of school trips by main mode within each trip-length band.

    More than one year is pooled per band as a mean weighted by each year's
    unweighted trip sample, the survey's own measure of how much each year's
    estimate rests on. The 2025 diary moved from paper to digital collection,
    which the survey reports lowered the recorded number of short walks and car
    trips, so shares across that break are not strictly comparable.

    Args:
        years (list[int], optional): Survey years to draw on, 2002 onwards.
        Defaults to NTS_YEARS.

        age (str, optional): Age band as the table labels it, one of "5 to
        16", "5 to 10" and "11 to 16". Defaults to "11 to 16".

    Returns:
        pd.DataFrame: One row per band in NTS_BANDS order, one column per mode
        in NTS_MODES order, each row summing to 1.
    """
    table = pd.read_excel(
        sources.NTS_MODE_BY_LENGTH_ODS,
        sources.NTS_MODE_BY_LENGTH_SHEET,
        header=5,
        engine="calamine",
    )
    table.columns = table.columns.str.strip()
    table = table.rename(columns=NTS_MODES)

    missing = sorted(set(years) - set(table["Year"]))
    if missing:
        raise ValueError(f"NTS0614a holds no year {missing}.")
    if age not in set(table["Age"]):
        raise ValueError(f"NTS0614a holds no age band {age!r}.")
    table = table[table["Year"].isin(years) & (table["Age"] == age)]

    modes = list(NTS_MODES.values())
    shares = (
        pd.DataFrame(
            {
                band: np.average(
                    table.loc[table["Trip length"] == band, modes],
                    axis=0,
                    weights=table.loc[table["Trip length"] == band, NTS_TRIPS],
                )
                for band in NTS_BANDS
            },
            index=modes,
        ).T
        / 100
    )
    # np.average raises on an empty band, so every band is present by here.

    if not np.allclose(shares.sum(axis=1), 1, atol=1e-6):
        raise ValueError(
            "NTS0614a mode shares do not sum to 100 in every band:\n"
            f"{shares.sum(axis=1)}"
        )
    return shares
