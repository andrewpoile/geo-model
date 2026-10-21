import numpy as np
import pytest

from geo_model.data import schools as sc
from geo_model.data import sources

# --------------------------------------------------------------------------
# load_p8
# --------------------------------------------------------------------------


def test_load_p8_keeps_school_rows_with_a_published_score(ks4_source):
    ks4_source(
        "1,852,4278,116458,Bitterne Park School,-0.19\n"
        "1,852,4311,116469,Cantell School,0.21\n"
        "2,852,,,Southampton LA average,-0.35\n"  # local authority aggregate
        "4,,,,England average,0.00\n"  # national aggregate
    )

    p8 = sc.load_p8()

    assert list(p8.columns) == ["LEA", "ESTAB", "P8MEA"]
    np.testing.assert_array_equal(p8["ESTAB"], [4278, 4311])
    np.testing.assert_allclose(p8["P8MEA"], [-0.19, 0.21])


@pytest.mark.parametrize("code", ["NP", "NE", "SUPP"])
def test_load_p8_drops_every_suppression_code_the_release_carries(ks4_source, code):
    ks4_source(
        "1,852,4278,116458,Bitterne Park School,-0.19\n"
        f"1,852,9999,116999,Suppressed School,{code}\n"
    )

    p8 = sc.load_p8()

    np.testing.assert_array_equal(p8["ESTAB"], [4278])
    assert p8["P8MEA"].dtype.kind == "f"  # coerced, not left as text


def test_load_p8_returns_nothing_when_no_school_publishes_a_score(ks4_source):
    ks4_source("1,852,4278,116458,Independent School,NP\n")

    p8 = sc.load_p8()

    assert p8.empty
    assert list(p8.columns) == ["LEA", "ESTAB", "P8MEA"]


def test_load_p8_keys_a_score_on_the_authority_as_well_as_the_school(ks4_source):
    # Establishment numbers are only unique within a local authority, so the
    # pair is what the register is joined on.
    ks4_source(
        "1,852,4278,116458,Southampton School,-0.19\n"
        "1,850,4278,116999,Portsmouth School,1.10\n"
    )

    p8 = sc.load_p8()

    np.testing.assert_array_equal(p8["LEA"], [852, 850])
    assert len(p8.drop_duplicates(["LEA", "ESTAB"])) == 2


def test_load_p8_fails_loudly_when_the_release_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "KS4_CSV", tmp_path / "absent.csv")

    with pytest.raises(FileNotFoundError):
        sc.load_p8()


# --------------------------------------------------------------------------
# load_places_offered
# --------------------------------------------------------------------------


def test_load_places_offered_floors_the_mean_of_year_7_offers_in_the_years_named(
    offers_source,
):
    offers_source(
        [
            # 150 and 153 in the years named: a mean of 151.5, floored to 151.
            # The year before them is left out.
            (202425, "Secondary", 1014001, "Hill School", 300, 7),
            (202526, "Secondary", 1014001, "Hill School", 150, 7),
            (202627, "Secondary", 1014001, "Hill School", 153, 7),
            # An upper school's offers are for Year 9, and a primary school's
            # for reception, so neither is a Year 7 intake.
            (202526, "Secondary", 1014002, "Upper School", 90, 9),
            (202627, "Secondary", 1014002, "Upper School", 90, 9),
            (202627, "Primary", 1012001, "Low Primary", 30, "R"),
            # A new school is averaged over the one year it offered places.
            (202627, "Secondary", 1024003, "New School", 121, 7),
            # One that offered none in the years named is left out.
            (202425, "Secondary", 1024004, "Closing School", 80, 7),
        ]
    )

    places = sc.load_places_offered([202526, 202627])

    assert list(places.columns) == ["LA (code)", "EstablishmentNumber", "PlacesOffered"]
    assert places.to_dict("list") == {
        "LA (code)": [101, 102],
        "EstablishmentNumber": [4001, 4003],
        "PlacesOffered": [151, 121],
    }


def test_load_places_offered_rejects_an_admission_year_the_file_does_not_hold(
    offers_source,
):
    offers_source([(202627, "Secondary", 1014001, "Hill School", 150, 7)])

    with pytest.raises(ValueError, match="209900"):
        sc.load_places_offered([202627, 209900])


def test_load_places_offered_fails_loudly_when_the_file_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sources, "OFFERS_CSV", tmp_path / "absent.csv")

    with pytest.raises(FileNotFoundError):
        sc.load_places_offered()


# --------------------------------------------------------------------------
# load_schools: against a register of its own
# --------------------------------------------------------------------------


def test_load_schools_keeps_the_authorities_named_alone(register_source):
    primary, secondary = sc.load_schools(["Alpha"])

    assert list(primary["EstablishmentName"]) == ["Low Primary"]
    # No other authority named runs a school of the same name, so the name
    # stands as the register has it.
    assert list(secondary["EstablishmentName"]) == ["Hill School"]
    assert list(secondary["PlacesOffered"]) == [150]


def test_load_schools_tells_apart_a_name_two_authorities_share(register_source):
    _, secondary = sc.load_schools(["Alpha", "Beta"])

    assert list(secondary["EstablishmentName"]) == [
        "Hill School (Alpha)",
        "Hill School (Beta)",
        "Vale School",
    ]
    # Each keeps its own score and places, joined on its own authority.
    assert list(secondary["P8MEA"]) == [0.1, -0.2, 0.3]
    assert list(secondary["PlacesOffered"]) == [150, 180, 210]


def test_load_schools_drops_a_school_that_offered_no_year_7_place(
    register_source, capsys
):
    _, secondary = sc.load_schools(["Beta"])

    # Upper School admits into Year 9 and Empty School offered none, so
    # neither has a Year 7 intake to be matched to.
    assert list(secondary["EstablishmentName"]) == ["Hill School", "Vale School"]
    out = capsys.readouterr().out
    assert "No Year 7 place offered" in out
    assert "Upper School, Empty School" in out


def test_load_schools_rejects_an_authority_running_no_open_state_school(
    register_source,
):
    with pytest.raises(ValueError, match=r"named \['Gamma'\]"):
        sc.load_schools(["Alpha", "Gamma"])
