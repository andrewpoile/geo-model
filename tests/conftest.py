import pytest

from geo_model import build_routes as br
from geo_model import load_data as ld
from geo_model.build_prefs import cohort_sizes
from geo_model.dissimilarity import student_samples

# The data folder is gitignored, so the tests that read it skip where it is
# absent, as in CI. The frames below are shared by every test of the session,
# so a test must not change them in place.
DATA = ld.POPULATION_DIR.parents[1]


@pytest.fixture(scope="session")
def data():
    """Skip the test unless the real data folder is present."""
    if not DATA.is_dir():
        pytest.skip(f"the {DATA} folder is not present")


@pytest.fixture(scope="session")
def areas(data):
    """The default region's districts, loaded once for the session."""
    return ld.load_areas()


@pytest.fixture(scope="session")
def schools(data):
    """The default region's primary and secondary schools, loaded once."""
    return ld.load_schools()


@pytest.fixture(scope="session")
def secondary(schools):
    """The default region's secondary schools."""
    return schools[1]


@pytest.fixture(scope="session")
def shares(data):
    """The NTS mode shares, loaded once for the session."""
    return ld.load_nts_mode_shares()


@pytest.fixture(scope="session")
def samples(areas):
    """One student sample of the default region."""
    return list(student_samples(areas, 1))


@pytest.fixture(scope="session")
def routes(areas, secondary):
    """The default region's route network at the module defaults."""
    return br.route_network(
        areas,
        secondary[["Easting", "Northing"]].to_numpy(),
        secondary["P8MEA"].to_numpy(),
        secondary["PlacesOffered"].to_numpy(),
        cohort_sizes(areas, "secondary"),
    )
