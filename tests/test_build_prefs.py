import numpy as np
import pandas as pd
import pytest

from geo_model import build_prefs as bp
from geo_model import build_routes as br

DATA = bp.POPULATION_XLSX.parent.parent
POPULATION_COLUMNS = ["LSOA21CD", "Total", "F4", "F11", "M4", "M11"]


def write_population_workbook(path, rows, sheet="Mid-2024 LSOA 2021"):
    """A stand-in for the ONS workbook: three banner rows, then the header.

    The real release carries 187 single-year-of-age columns; only the six the
    model uses are written here, plus one it must drop.
    """
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # startrow=3 leaves the three banner rows read_excel is told to skip.
        frame.to_excel(writer, sheet_name=sheet, startrow=3, index=False)
    return frame


@pytest.fixture
def population_source(tmp_path, monkeypatch):
    """Point the loader at a workbook and a cache of its own."""
    source = tmp_path / "population.xlsx"
    monkeypatch.setattr(bp, "POPULATION_XLSX", source)
    monkeypatch.setattr(bp, "POPULATION_CACHE", tmp_path / "cache" / "population.pkl")
    write_population_workbook(source, {
        "LAD 2023 Code": ["E06000045", "E06000045"],
        "LSOA 2021 Code": ["E01011949", "E01011950"],
        "Total": [1898, 1247],
        "F4": [11, 13],
        "F11": [9, 7],
        "M4": [4, 7],
        "M11": [18, 10],
    })
    return source


# --------------------------------------------------------------------------
# load_population
# --------------------------------------------------------------------------

def test_load_population_keeps_only_the_columns_the_model_uses(population_source):
    population = bp.load_population()

    assert list(population.columns) == POPULATION_COLUMNS
    np.testing.assert_array_equal(population["LSOA21CD"], ["E01011949", "E01011950"])
    np.testing.assert_array_equal(population["F11"], [9, 7])


def test_load_population_writes_a_cache_keyed_on_the_source_file(population_source):
    population = bp.load_population()

    assert bp.POPULATION_CACHE.exists()
    stat = population_source.stat()
    assert population.attrs["source_key"] == (
        population_source.name, stat.st_mtime_ns, stat.st_size
    )


def test_load_population_reuses_the_cache_without_reading_the_workbook(
    population_source, monkeypatch
):
    first = bp.load_population()

    # Any second read of the workbook would now fail, so a returned frame
    # proves the cache alone served it.
    monkeypatch.setattr(
        bp.pd, "read_excel", lambda *a, **k: pytest.fail("re-read a cached workbook")
    )
    pd.testing.assert_frame_equal(bp.load_population(), first)


def test_load_population_rebuilds_a_cache_left_behind_by_another_file(
    population_source, tmp_path
):
    bp.load_population()

    # The cache now holds a key from a workbook that is no longer the source.
    write_population_workbook(population_source, {
        "LAD 2023 Code": ["E06000045"],
        "LSOA 2021 Code": ["E01099999"],
        "Total": [1],
        "F4": [2],
        "F11": [3],
        "M4": [4],
        "M11": [5],
    })

    rebuilt = bp.load_population()
    np.testing.assert_array_equal(rebuilt["LSOA21CD"], ["E01099999"])
    np.testing.assert_array_equal(rebuilt["M11"], [5])


def test_load_population_fails_loudly_when_the_workbook_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(bp, "POPULATION_XLSX", tmp_path / "absent.xlsx")
    monkeypatch.setattr(bp, "POPULATION_CACHE", tmp_path / "cache.pkl")

    with pytest.raises(FileNotFoundError):
        bp.load_population()


# --------------------------------------------------------------------------
# load_p8
# --------------------------------------------------------------------------

KS4_HEADER = "RECTYPE,LEA,ESTAB,URN,SCHNAME,P8MEA\n"


@pytest.fixture
def ks4_source(tmp_path, monkeypatch):
    def write(body):
        path = tmp_path / "ks4final.csv"
        # The release is published with a byte order mark, as the loader expects.
        path.write_text(KS4_HEADER + body, encoding="utf-8-sig")
        monkeypatch.setattr(bp, "KS4_CSV", path)
        return path
    return write


def test_load_p8_keeps_school_rows_with_a_published_score(ks4_source):
    ks4_source(
        "1,852,4278,116458,Bitterne Park School,-0.19\n"
        "1,852,4311,116469,Cantell School,0.21\n"
        "2,852,,,Southampton LA average,-0.35\n"   # local authority aggregate
        "4,,,,England average,0.00\n"              # national aggregate
    )

    p8 = bp.load_p8()

    assert list(p8.columns) == ["LEA", "ESTAB", "P8MEA"]
    np.testing.assert_array_equal(p8["ESTAB"], [4278, 4311])
    np.testing.assert_allclose(p8["P8MEA"], [-0.19, 0.21])


@pytest.mark.parametrize("code", ["NP", "NE", "SUPP"])
def test_load_p8_drops_every_suppression_code_the_release_carries(ks4_source, code):
    ks4_source(
        "1,852,4278,116458,Bitterne Park School,-0.19\n"
        f"1,852,9999,116999,Suppressed School,{code}\n"
    )

    p8 = bp.load_p8()

    np.testing.assert_array_equal(p8["ESTAB"], [4278])
    assert p8["P8MEA"].dtype.kind == "f"  # coerced, not left as text


def test_load_p8_returns_nothing_when_no_school_publishes_a_score(ks4_source):
    ks4_source("1,852,4278,116458,Independent School,NP\n")

    p8 = bp.load_p8()

    assert p8.empty
    assert list(p8.columns) == ["LEA", "ESTAB", "P8MEA"]


def test_load_p8_keys_a_score_on_the_authority_as_well_as_the_school(ks4_source):
    # Establishment numbers are only unique within a local authority, so the
    # pair is what the register is joined on.
    ks4_source(
        "1,852,4278,116458,Southampton School,-0.19\n"
        "1,850,4278,116999,Portsmouth School,1.10\n"
    )

    p8 = bp.load_p8()

    np.testing.assert_array_equal(p8["LEA"], [852, 850])
    assert len(p8.drop_duplicates(["LEA", "ESTAB"])) == 2


def test_load_p8_fails_loudly_when_the_release_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "KS4_CSV", tmp_path / "absent.csv")

    with pytest.raises(FileNotFoundError):
        bp.load_p8()


# --------------------------------------------------------------------------
# main: the whole pipeline, against the real data folder
# --------------------------------------------------------------------------

pytestmark_data = pytest.mark.skipif(
    not DATA.is_dir(), reason=f"the {DATA} folder is not present"
)


@pytest.fixture
def redirected_outputs(tmp_path, monkeypatch):
    """Send every file the pipeline writes to a temporary folder.

    The real temp/ folder holds the artefacts the rest of the model is run
    from, so a test must not overwrite them.
    """
    monkeypatch.setattr(bp, "OUTPUT_NPZ", tmp_path / "prefprio.npz")
    monkeypatch.setattr(bp, "POPULATION_CACHE", tmp_path / "population_lsoa.pkl")
    monkeypatch.setattr(br, "ROUTES_CSV", tmp_path / "secondary_routes.csv")
    monkeypatch.setattr(br, "ROUTES_NPZ", tmp_path / "secondary_routes.npz")
    return tmp_path


@pytestmark_data
def test_main_writes_a_consistent_instance_for_both_phases(redirected_outputs):
    bp.main()

    assert bp.OUTPUT_NPZ.exists()
    assert br.ROUTES_CSV.exists() and br.ROUTES_NPZ.exists()

    with np.load(bp.OUTPUT_NPZ) as built:
        routes = np.load(br.ROUTES_NPZ)
        n_routes = len(routes["route_capacities"])

        for phase, phase_routes in [("primary", 0), ("secondary", n_routes)]:
            preferences = built[f"{phase}_student_preferences"]
            priorities = built[f"{phase}_school_priorities"]
            capacities = built[f"{phase}_school_capacities"]

            n_schools, route_axis, n_students = priorities.shape
            assert preferences.shape == (n_students, preferences.shape[1], 2)
            # The route axis carries every route plus the routeless bundle.
            assert route_axis == phase_routes + 1
            assert capacities.shape == (n_schools,)
            assert n_students > 0 and n_schools > 0
            assert (capacities >= 1).all()

            # Every ranked bundle is a real school, or the padding a short list
            # carries, and every route index falls on the route axis.
            schools, bundle_routes = preferences[..., 0], preferences[..., 1]
            assert ((schools == -1) | (schools < n_schools)).all()
            assert ((bundle_routes == -1) | (bundle_routes < phase_routes)).all()
            assert (preferences[schools == -1] == -1).all()


@pytestmark_data
def test_main_only_routes_the_secondary_phase(redirected_outputs):
    bp.main()

    with np.load(bp.OUTPUT_NPZ) as built:
        # Routes serve the schools the disadvantaged districts cannot reach
        # unaided, which the model builds for the secondary phase alone.
        assert built["primary_school_priorities"].shape[1] == 1
        assert built["secondary_school_priorities"].shape[1] > 1
        assert (built["primary_student_preferences"][..., 1] == -1).all()
        assert (built["secondary_student_preferences"][..., 1] > -1).any()


@pytestmark_data
def test_main_is_reproducible_from_the_seed(redirected_outputs):
    bp.main()
    first = dict(np.load(bp.OUTPUT_NPZ))

    bp.main()
    with np.load(bp.OUTPUT_NPZ) as second:
        for name, array in first.items():
            np.testing.assert_array_equal(array, second[name], err_msg=name)
