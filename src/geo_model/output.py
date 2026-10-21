from pathlib import Path

import numpy as np
import pandas as pd

# Every region's outputs are written to a folder of its own in here, so runs
# for different regions never write over each other.
OUTPUT_DIR = Path("output")

# Written to each region's folder.
ROUTES_CSV = "secondary_routes.csv"
ROUTES_NPZ = "secondary_routes.npz"


def region_dir(las: list[str]) -> Path:
    """The folder a region's outputs are written to: its local authorities'
    names, as the register spells them, joined by "+" in alphabetical order.

    Args:
        las (list[str]): The region's local authorities.

    Returns:
        Path: A folder in OUTPUT_DIR.
    """
    return OUTPUT_DIR / "+".join(sorted(las))


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
    )
