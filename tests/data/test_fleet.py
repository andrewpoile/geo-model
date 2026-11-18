import pytest

from geo_model.data import fleet, sources


@pytest.fixture
def minibuses_source(tmp_path, monkeypatch):
    def write(body):
        path = tmp_path / "minibuses.csv"
        path.write_text("model,seats\n" + body)
        monkeypatch.setattr(sources, "MINIBUSES_CSV", path)

    return write


def test_minibus_seats_reads_the_seats_of_the_model_named(minibuses_source):
    minibuses_source("A.1,16\nNA,9\n")
    # Read as written, not as a missing name.
    assert fleet.minibus_seats("NA") == 9
    with pytest.raises(ValueError, match="choose from A.1, NA"):
        fleet.minibus_seats("C")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("A,17\n", "small coach"),
        ("A,0\n", "small coach"),
        ("A,9.5\n", "small coach"),
        ("A,16 seats\n", "small coach"),
        ("A,9\nA,12\n", "named twice"),
        ("A_1,9\n", "letters, digits"),
    ],
)
def test_load_minibuses_rejects_what_is_no_minibus(minibuses_source, body, message):
    minibuses_source(body)
    with pytest.raises(ValueError, match=message):
        fleet.load_minibuses()


def test_load_minibuses_rejects_a_file_with_no_seats(tmp_path, monkeypatch):
    path = tmp_path / "minibuses.csv"
    path.write_text("model\nA\n")
    monkeypatch.setattr(sources, "MINIBUSES_CSV", path)
    with pytest.raises(ValueError, match="no column seats"):
        fleet.load_minibuses()
