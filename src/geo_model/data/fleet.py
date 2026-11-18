import pandas as pd

from geo_model.data import sources

# Passenger seats, the driver's not counted. A vehicle with more is a small
# coach rather than a minibus under GOV.UK's licensing, and needs another
# licence to drive.
MAX_MINIBUS_SEATS = 16

# A model's name ends an output file's name, so it holds nothing a file name
# or the "_" between its blocks would break on.
MODEL_NAME = r"[A-Za-z0-9.-]+"


def load_minibuses() -> pd.DataFrame:
    """Import the minibus models a school can hire, from MINIBUSES_CSV.

    Returns:
        pd.DataFrame: One row per model, indexed by "model", with its "seats"
        and every further column as read.

    Raises:
        ValueError: If a model is named twice or by anything but letters,
        digits, "." and "-", or its seats are not a whole number in
        [1, MAX_MINIBUS_SEATS].
    """
    minibuses = pd.read_csv(sources.MINIBUSES_CSV, dtype={"model": str})
    named = minibuses["model"].str.fullmatch(MODEL_NAME, na=False)
    if not named.all():
        raise ValueError(
            "A minibus model is named by letters, digits, '.' and '-' alone: "
            + ", ".join(map(repr, minibuses.loc[~named, "model"]))
        )
    if minibuses["model"].duplicated().any():
        raise ValueError(
            "Minibus models named twice: "
            + ", ".join(minibuses.loc[minibuses["model"].duplicated(), "model"])
        )
    seats = minibuses["seats"]
    whole = (seats % 1 == 0) & seats.between(1, MAX_MINIBUS_SEATS)
    if not whole.all():
        raise ValueError(
            f"A minibus seats a whole number of passengers in [1, "
            f"{MAX_MINIBUS_SEATS}], more making it a small coach: "
            + ", ".join(minibuses.loc[~whole, "model"])
        )
    return minibuses.astype({"seats": int}).set_index("model")


def minibus_seats(model: str) -> int:
    """The passenger seats of `model`, a model `load_minibuses` reads.

    Raises:
        ValueError: If `model` is not one of them.
    """
    minibuses = load_minibuses()
    if model not in minibuses.index:
        raise ValueError(
            f"No minibus model {model!r} in {sources.MINIBUSES_CSV}; choose from "
            + ", ".join(minibuses.index)
        )
    return int(minibuses["seats"].to_dict()[model])
