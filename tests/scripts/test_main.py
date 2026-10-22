import sys
from dataclasses import replace

import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from geo_model import output
from geo_model.data import regions as rg
from geo_model.model.settings import DEFAULTS
from geo_model.output import ANON_DIR, ROUTES_CSV, ROUTES_NPZ
from geo_model.scripts import build_prefs as bp
from geo_model.scripts import optimise as op
from geo_model.scripts import run as rn
from geo_model.scripts import scenario_1 as s1
from geo_model.scripts import scenario_2 as s2
from geo_model.scripts import scenario_3 as s3
from geo_model.scripts import sweep as sw

# What `run` writes of the setting it runs, for every entry point that runs one.
RUN_FILES = [
    ROUTES_CSV,
    ROUTES_NPZ,
    rn.RESULTS_CSV,
    rn.SCHOOLS_CSV,
    rn.MODES_CSV,
    rn.ROUTE_UTILISATION_CSV,
    rn.DISTRICTS_CSV,
    rn.DISSIMILARITY_PNG,
    rn.UNASSIGNED_PNG,
    rn.INTAKE_PNG,
    rn.MODES_PNG,
    rn.LORENZ_PNG,
    rn.MAP_PNG,
    rn.ROUTES_PNG,
    rn.LOCALS_PNG,
]

# Two parameters of three cells in all, so the sweep scores and plots quickly.
SMALL_GRID = {
    "capacity_scale": [DEFAULTS, replace(DEFAULTS, capacity_scale=25.0)],
    "decile": [DEFAULTS],
}
# Minimal arguments for the scripts that search or sweep.
OPTIMISE_ARGV = [
    *["--seeds", "1", "--held-out-seeds", "1"],
    *["--pop-size", "4", "--generations", "2"],
]
SWEEP_ARGV = ["--seeds", "2", "--map", "--lorenz"]
# The columns naming a school, district or authority, the only ones an
# anonymised run writes otherwise than a named one.
NAME_COLUMNS = ["school", "LSOA21CD", "LA (name)"]


def load_region(module, areas, schools, tmp_path, monkeypatch):
    """Have `module.main` simulate the default region from the frames the
    session has loaded, writing to `tmp_path`."""
    # The real output folder holds the user's results, so the region is
    # written to a temporary one.
    monkeypatch.setattr(output, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(rg, "load_areas", lambda las: areas)
    # Every script that loads a region's schools, a preset loading them in run.
    for loader in (bp, op, rn, sw):
        monkeypatch.setattr(loader, "load_schools", lambda las: schools)
    if module is sw:
        # Optimise searches the parameters GRID sweeps, so only the sweep's
        # own run takes the small grid.
        monkeypatch.setattr(sw, "GRID", SMALL_GRID)


# --------------------------------------------------------------------------
# main: every entry point, on minimal arguments against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "argv", "folder", "written"),
    [
        (bp, [], "", [bp.OUTPUT_NPZ, ROUTES_CSV, ROUTES_NPZ]),
        (
            op,
            OPTIMISE_ARGV,
            op.OUT_DIR,
            [op.EVALUATIONS_CSV, op.FRONT_CSV, op.FRONT_PNG, *RUN_FILES],
        ),
        (rn, ["--seeds", "2"], rn.OUT_DIR, RUN_FILES),
        (
            s1,
            ["--seeds", "2"],
            s1.OUT_DIR,
            RUN_FILES,
        ),
        (s2, ["--seeds", "2"], s2.OUT_DIR, RUN_FILES),
        (s3, ["--seeds", "2"], s3.OUT_DIR, RUN_FILES),
        (
            sw,
            SWEEP_ARGV,
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
        "optimise",
        "run",
        "scenario_1",
        "scenario_2",
        "scenario_3",
        "sweep",
    ],
)
def test_main_writes_its_outputs_and_nothing_else(
    module, argv, folder, written, areas, schools, tmp_path, monkeypatch
):
    load_region(module, areas, schools, tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])

    module.main()

    out_dir = tmp_path / "Southampton" / folder
    assert sorted(path.name for path in out_dir.iterdir() if path.is_file()) == sorted(
        written
    )


@pytest.mark.parametrize(
    ("module", "argv", "folder"),
    [
        (op, OPTIMISE_ARGV, op.OUT_DIR),
        (rn, ["--seeds", "2"], rn.OUT_DIR),
        (sw, SWEEP_ARGV, sw.SWEEP_DIR),
    ],
    ids=["optimise", "run", "sweep"],
)
def test_anon_main_writes_the_named_results_naming_nothing(
    module, argv, folder, areas, schools, tmp_path, monkeypatch
):
    load_region(module, areas, schools, tmp_path, monkeypatch)
    figures = []
    savefig = Figure.savefig

    def keep(fig, *args, **kwargs):
        savefig(fig, *args, **kwargs)
        figures.append(fig)

    monkeypatch.setattr(Figure, "savefig", keep)
    for flags in ([], ["--anon"]):
        # Only the anonymised run's figures are kept.
        figures.clear()
        monkeypatch.setattr(sys, "argv", [module.__name__, *argv, *flags])
        module.main()

    named = tmp_path / "Southampton" / folder
    anon = tmp_path / "Southampton" / ANON_DIR / folder
    assert sorted(path.name for path in anon.iterdir()) == sorted(
        path.name for path in named.iterdir()
    )
    # Only the names differ, so the matchings are the named run's.
    for path in named.glob("*.csv"):
        rows, anon_rows = pd.read_csv(path), pd.read_csv(anon / path.name)
        names = [column for column in NAME_COLUMNS if column in rows]
        pd.testing.assert_frame_equal(
            rows.drop(columns=names), anon_rows.drop(columns=names)
        )
    region = [
        "Southampton",
        *schools[1]["EstablishmentName"],
        *areas["LSOA21CD"],
        *areas["LSOA21NM"],
    ]
    # Every figure is read after its save drew it, so its tick labels are set.
    assert figures
    written = [path.read_text() for path in anon.glob("*.csv")] + [
        "\n".join(text.get_text() for text in fig.findobj(Text)) for fig in figures
    ]
    for text in written:
        assert [name for name in region if name in text] == []
