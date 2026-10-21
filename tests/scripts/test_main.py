import sys
from dataclasses import replace

import pytest

from geo_model import output
from geo_model.data import regions as rg
from geo_model.model.settings import DEFAULTS
from geo_model.output import ROUTES_CSV, ROUTES_NPZ
from geo_model.scripts import build_prefs as bp
from geo_model.scripts import car_displacement as cd
from geo_model.scripts import dissimilarity as ds
from geo_model.scripts import optimise as op
from geo_model.scripts import scenario_1 as s1
from geo_model.scripts import sweep as sw

# The plots `compare` writes for the scenarios an entry point compares.
COMPARE_PNGS = [
    op.DISSIMILARITY_PNG,
    op.UNASSIGNED_PNG,
    op.INTAKE_PNG,
    op.MODES_PNG,
    op.LORENZ_PNG,
    op.MAP_PNG,
    op.ROUTES_PNG,
    op.LOCALS_PNG,
]

# Two parameters of three cells in all, so the sweep scores and plots quickly.
SMALL_GRID = {
    "capacity_scale": [DEFAULTS, replace(DEFAULTS, capacity_scale=25.0)],
    "decile": [DEFAULTS],
}


# --------------------------------------------------------------------------
# main: every entry point, on minimal arguments against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "argv", "folder", "written"),
    [
        (bp, [], "", [bp.OUTPUT_NPZ, ROUTES_CSV, ROUTES_NPZ]),
        (cd, ["--seeds", "2"], "", [cd.RESULTS_CSV, cd.PLOT_PNG]),
        (ds, ["--seeds", "2"], "", [ds.RESULTS_CSV, ds.PLOT_PNG]),
        (
            op,
            ["--seeds", "1", "--held-out-seeds", "1"]
            + ["--pop-size", "4", "--generations", "2"],
            op.OUT_DIR,
            [op.EVALUATIONS_CSV, op.FRONT_CSV, op.FRONT_PNG, *COMPARE_PNGS],
        ),
        (
            s1,
            ["--seeds", "2"],
            s1.OUT_DIR,
            [
                ROUTES_CSV,
                ROUTES_NPZ,
                s1.RESULTS_CSV,
                s1.SCHOOLS_CSV,
                s1.MODES_CSV,
                s1.ROUTE_UTILISATION_CSV,
                s1.DISTRICTS_CSV,
                *COMPARE_PNGS,
            ],
        ),
        (
            sw,
            ["--seeds", "2", "--map", "--lorenz"],
            sw.SWEEP_DIR,
            [
                sw.RESULTS_CSV,
                sw.SCHOOLS_CSV,
                sw.MODES_CSV,
                sw.TESTS_CSV,
                *(f"{parameter}.png" for parameter in SMALL_GRID),
                sw.PLOT_PNG,
                sw.MAP_PNG,
                sw.LORENZ_PNG,
            ],
        ),
    ],
    ids=[
        "build_prefs",
        "car_displacement",
        "dissimilarity",
        "optimise",
        "scenario_1",
        "sweep",
    ],
)
def test_main_writes_its_outputs_and_nothing_else(
    module, argv, folder, written, areas, schools, tmp_path, monkeypatch
):
    # The real output folder holds the user's results, so the region is
    # written to a temporary one, from the frames the session has loaded.
    monkeypatch.setattr(output, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(rg, "load_areas", lambda las: areas)
    monkeypatch.setattr(module, "load_schools", lambda las: schools)
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    if module is sw:
        # Optimise searches the parameters GRID sweeps, so only the sweep's
        # own run takes the small grid.
        monkeypatch.setattr(sw, "GRID", SMALL_GRID)

    module.main()

    out_dir = tmp_path / "Southampton" / folder
    assert sorted(path.name for path in out_dir.iterdir() if path.is_file()) == sorted(
        written
    )
