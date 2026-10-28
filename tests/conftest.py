import pytest

from geo_model.data import areas as ar
from geo_model.data import network as nw
from geo_model.data import schools as sc
from geo_model.data import sources, travel
from geo_model.model import routes as rt
from geo_model.model.students import cohort_sizes, student_samples

# The data folder is gitignored, so the tests that read it skip where it is
# absent, as in CI. It is told by a folder only it holds, since tests/data is a
# data folder too to pytest run from tests/. The frames below are shared by
# every test of the session, so a test must not change them in place.
DATA = sources.POPULATION_DIR


@pytest.fixture(scope="session")
def data():
    """Skip the test unless the real data folder is present."""
    if not DATA.is_dir():
        pytest.skip(f"the {DATA} folder is not present")


@pytest.fixture(scope="session")
def areas(data):
    """The default region's districts, loaded once for the session."""
    return ar.load_areas()


@pytest.fixture(scope="session")
def schools(data):
    """The default region's primary and secondary schools, loaded once."""
    return sc.load_schools()


@pytest.fixture(scope="session")
def secondary(schools):
    """The default region's secondary schools."""
    return schools[1]


@pytest.fixture(scope="session")
def shares(data):
    """The NTS mode shares, loaded once for the session."""
    return travel.load_nts_mode_shares()


@pytest.fixture(scope="session", autouse=True)
def network_cache(tmp_path_factory):
    """A folder every test caches the region's links in, so the session reads
    them from the GeoPackage once and never touches the real cache."""
    cache = tmp_path_factory.mktemp("network")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sources, "NETWORK_CACHE_DIR", cache)
        yield cache


@pytest.fixture(scope="session")
def links(areas):
    """The default region's links, as `region_links` gives them."""
    return nw.region_links(areas, "+".join(sorted(sources.LAS)))


@pytest.fixture(scope="session")
def to_school(secondary):
    """Straight-line distance to the default region's secondary schools, the
    model as it was before the walking network, which the tests read it on."""
    return nw.Euclidean(secondary[["Easting", "Northing"]].to_numpy())


@pytest.fixture(scope="session")
def district_distance(areas, to_school):
    """Distance from every district's population centroid to every school."""
    return to_school(rt.centroid_xy(areas))


@pytest.fixture(scope="session")
def district_minutes(areas, secondary, links):
    """Minutes by bus from every district's population centroid to every
    school."""
    return nw.bus_minutes(
        links,
        secondary[["Easting", "Northing"]].to_numpy(),
        rt.centroid_xy(areas),
    )


@pytest.fixture(scope="session")
def samples(areas, to_school):
    """One student sample of the default region."""
    return list(student_samples(areas, 1, to_school))


@pytest.fixture(scope="session")
def routes(areas, secondary, district_distance, district_minutes):
    """The default region's route network at the module defaults."""
    return rt.route_network(
        areas,
        district_distance,
        district_minutes,
        secondary["P8MEA"].to_numpy(),
        secondary["PlacesOffered"].to_numpy(),
        cohort_sizes(areas, "secondary"),
    )
