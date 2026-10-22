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
# Minimal arguments for the search, run both named and anonymised.
OPTIMISE_ARGV = [
    *["--seeds", "1", "--held-out-seeds", "1"],
    *["--pop-size", "4", "--generations", "2"],
]
# The columns naming a school, district or authority, the only ones an
# anonymised run writes otherwise than a named one.
NAME_COLUMNS = ["school", "LSOA21CD", "LA (name)"]


# --------------------------------------------------------------------------
# main: every entry point, on minimal arguments against the real data folder
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "argv", "folder", "written", "anon_argv"),
    [
        (bp, [], "", [bp.OUTPUT_NPZ, ROUTES_CSV, ROUTES_NPZ], None),
        (
            op,
            OPTIMISE_ARGV,
            op.OUT_DIR,
            [op.EVALUATIONS_CSV, op.FRONT_CSV, op.FRONT_PNG, *RUN_FILES],
            [*OPTIMISE_ARGV, "--anon"],
        ),
        (rn, ["--seeds", "2"], rn.OUT_DIR, RUN_FILES, ["--seeds", "2", "--anon"]),
        (s1, ["--seeds", "2"], s1.OUT_DIR, RUN_FILES, None),
        (s2, ["--seeds", "2"], s2.OUT_DIR, RUN_FILES, None),
        (s3, ["--seeds", "2"], s3.OUT_DIR, RUN_FILES, None),
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
            ["--seeds", "2", "--lorenz", "--anon"],
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
    module, argv, folder, written, anon_argv, areas, schools, tmp_path, monkeypatch
):
    # The real output folder holds the user's results, so the region is
    # written to a temporary one, from the frames the session has loaded.
    monkeypatch.setattr(output, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(rg, "load_areas", lambda las: areas)
    # Every script that loads a region's schools, the rest loading them in run.
    for loader in (bp, rn):
        monkeypatch.setattr(loader, "load_schools", lambda las: schools)
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
    if anon_argv is None:
        return

    # Anonymised, the same arguments write everything but the map, naming
    # nothing.
    figures = []
    savefig = Figure.savefig

    def keep(fig, *args, **kwargs):
        savefig(fig, *args, **kwargs)
        figures.append(fig)

    monkeypatch.setattr(Figure, "savefig", keep)
    monkeypatch.setattr(sys, "argv", [module.__name__, *anon_argv])

    module.main()

    anon_dir = tmp_path / "Southampton" / ANON_DIR / folder
    assert sorted(path.name for path in anon_dir.iterdir()) == sorted(
        set(written) - {rn.MAP_PNG, sw.MAP_PNG}
    )
    # Only the names differ, and the order of the districts numbered by
    # deprivation, so the matchings are the named run's.
    for path in out_dir.glob("*.csv"):
        rows, anon_rows = pd.read_csv(path), pd.read_csv(anon_dir / path.name)
        kept = list(rows.columns.drop([c for c in NAME_COLUMNS if c in rows]))
        pd.testing.assert_frame_equal(
            rows[kept].sort_values(kept, ignore_index=True),
            anon_rows[kept].sort_values(kept, ignore_index=True),
        )
    region = [
        "Southampton",
        *schools[1]["EstablishmentName"],
        *areas["LSOA21CD"],
        *areas["LSOA21NM"],
    ]
    # Every figure is read after its save drew it, so its tick labels are set.
    assert figures
    texts = [path.read_text() for path in anon_dir.glob("*.csv")] + [
        "\n".join(text.get_text() for text in fig.findobj(Text)) for fig in figures
    ]
    for text in texts:
        assert [name for name in region if name in text] == []
