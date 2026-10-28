from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from geo_model.data.network import DISTANCE
from geo_model.model.settings import PRIORITY

# Read by the type checker alone, so writing routes needs no plotting library.
if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Every region's outputs are written to a folder of its own in here, so runs
# for different regions never write over each other.
OUTPUT_DIR = Path("output")
# An anonymised run's folder within its region's, so it never writes over the
# named outputs.
ANON_DIR = Path("anonymised")

# Written to each region's folder.
ROUTES_CSV = "secondary_routes.csv"
ROUTES_NPZ = "secondary_routes.npz"


def region_dir(
    las: list[str],
    anon: bool = False,
    priority: str = PRIORITY,
    distance: str = DISTANCE,
) -> Path:
    """The folder a region's outputs are written to: its local authorities'
    names, as the register spells them, joined by "+" in alphabetical order.

    Args:
        las (list[str]): The region's local authorities.

        anon (bool, optional): The region's ANON_DIR instead, for an
        anonymised run. Defaults to False.

        priority (str, optional): One of PRIORITY_CHOICES. Any rule but
        PRIORITY writes to a folder of its name within that, so runs under
        two rules never write over each other. Defaults to PRIORITY.

        distance (str, optional): One of DISTANCE_CHOICES. Any rule but
        DISTANCE writes to a folder of its name within that. Defaults to
        DISTANCE.

    Returns:
        Path: A folder in OUTPUT_DIR.
    """
    folder = OUTPUT_DIR / "+".join(sorted(las))
    if anon:
        folder = folder / ANON_DIR
    if priority != PRIORITY:
        folder = folder / priority
    return folder if distance == DISTANCE else folder / distance


def save_routes(routes: pd.DataFrame, out_dir: Path) -> None:
    """Write the route set out, whole as a CSV and by axis as arrays, to
    ROUTES_CSV and ROUTES_NPZ in `out_dir`.

    Args:
        routes (pd.DataFrame): The route set, as returned by `route_network`.

        out_dir (Path): Folder the route set is written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    routes.to_csv(out_dir / ROUTES_CSV, index=False)
    np.savez(
        out_dir / ROUTES_NPZ,
        route_capacities=routes["capacity"].to_numpy(dtype=np.int32),
        route_school_idx=routes["school_idx"].to_numpy(dtype=np.int32),
        route_district_idx=routes["district_idx"].to_numpy(dtype=np.int32),
        route_minutes=routes["minutes"].to_numpy(dtype=float),
    )


def save(fig: "Figure", path: Path) -> None:
    """Write `fig` to `path`, at the resolution every plot is written at."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
