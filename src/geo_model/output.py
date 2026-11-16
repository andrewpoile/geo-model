from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from geo_model.model.settings import DEFAULTS, RULES, SHORT, Settings

# Read by the type checker alone, so writing routes needs no plotting library.
if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Every region's outputs are written to a folder of its own in here, so runs
# for different regions never write over each other.
OUTPUT_DIR = Path("output")
# An anonymised run's folder within its region's, so it never writes over the
# named outputs.
ANON_DIR = Path("anonymised")

# Written to each region's folder, `suffixed` as every output file is.
ROUTES_CSV = "secondary_routes.csv"
ROUTES_NPZ = "secondary_routes.npz"


def region_dir(las: list[str], anon: bool = False) -> Path:
    """The folder a region's outputs are written to: its local authorities'
    names, as the register spells them, joined by "+" in alphabetical order.

    Args:
        las (list[str]): The region's local authorities.

        anon (bool, optional): The region's ANON_DIR instead, for an
        anonymised run. Defaults to False.

    Returns:
        Path: A folder in OUTPUT_DIR.
    """
    folder = OUTPUT_DIR / "+".join(sorted(las))
    return folder / ANON_DIR if anon else folder


def run_suffix(settings: Settings, distance: str) -> str:
    """What a run's file names end in, so runs under different rules and
    parameters sit side by side in one folder: blocks joined by "_", each a
    name of SHORT, "-" and the value. The disadvantage, priority and distance
    rules always, then every other rule and then every parameter that is not
    at its default, a True one as its name alone.

    Args:
        settings (Settings): The rules and parameters the run follows.

        distance (str): One of DISTANCE_CHOICES, how it measures distances.

    Returns:
        str: The suffix, led by "_", such as
        "_dis-score_pri-distance_dst-network_rus_mnd-4828".
    """
    blocks = [
        f"{SHORT['disadvantage']}-{settings.disadvantage}",
        f"{SHORT['priority']}-{settings.priority}",
        f"dst-{distance}",
    ]
    names = [field.name for field in fields(Settings)]
    # Stable, so the rules lead and each group keeps the order of Settings.
    for name in sorted(names, key=lambda name: name not in RULES):
        value = getattr(settings, name)
        if isinstance(value, str) or value == getattr(DEFAULTS, name):
            continue
        blocks.append(SHORT[name] if value is True else f"{SHORT[name]}-{value:.12g}")
    return "_" + "_".join(blocks)


def suffixed(name: str, suffix: str) -> str:
    """The file name `name` with `suffix` ahead of its extension."""
    stem, extension = name.rsplit(".", 1)
    return f"{stem}{suffix}.{extension}"


def save_routes(routes: pd.DataFrame, out_dir: Path, suffix: str) -> None:
    """Write the route set out, whole as a CSV and by axis as arrays, to
    ROUTES_CSV and ROUTES_NPZ in `out_dir`, each `suffixed`.

    Args:
        routes (pd.DataFrame): The route set, as returned by `route_network`.

        out_dir (Path): Folder the route set is written to.

        suffix (str): The run's, as `run_suffix` gives it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    routes.to_csv(out_dir / suffixed(ROUTES_CSV, suffix), index=False)
    np.savez(
        out_dir / suffixed(ROUTES_NPZ, suffix),
        route_capacities=routes["capacity"].to_numpy(dtype=np.int32),
        route_school_idx=routes["school_idx"].to_numpy(dtype=np.int32),
        route_district_idx=routes["district_idx"].to_numpy(dtype=np.int32),
        route_minutes=routes["minutes"].to_numpy(dtype=float),
    )


def save(fig: "Figure", path: Path) -> None:
    """Write `fig` to `path`, at the resolution every plot is written at."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
