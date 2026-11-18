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
    minibuses_source("A.1,16\nB-2,9\n")
    assert fleet.minibus_seats("B-2") == 9
    with pytest.raises(ValueError, match="choose from A.1, B-2"):
        fleet.minibus_seats("C")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("A,17\n", "small coach"),
        ("A,0\n", "small coach"),
        ("A,9.5\n", "small coach"),
        ("A,9\nA,12\n", "named twice"),
        ("A_1,9\n", "letters, digits"),
    ],
)
def test_load_minibuses_rejects_what_is_no_minibus(minibuses_source, body, message):
    minibuses_source(body)
    with pytest.raises(ValueError, match=message):
        fleet.load_minibuses()
