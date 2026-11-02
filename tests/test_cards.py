import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tps_sim.cards import CardError, format_card, parse_card, parse_hand


def test_parse_card_case_insensitive():
    assert parse_card("Kh") == parse_card("kh") == parse_card("KH") == (13, "h")
    assert parse_card("Ac") == (14, "c")
    assert parse_card("Td") == (10, "d")
    assert parse_card("4s") == (4, "s")


def test_parse_card_invalid():
    with pytest.raises(CardError):
        parse_card("1h")  # no rank '1'
    with pytest.raises(CardError):
        parse_card("Kx")  # bad suit
    with pytest.raises(CardError):
        parse_card("Khh")  # wrong length


def test_parse_hand_runtogether():
    hand = parse_hand("AcKdJd7h7s2c2d")
    assert len(hand) == 7
    assert hand[0] == (14, "c")
    assert hand[-1] == (2, "d")


def test_parse_hand_spaced_and_commas():
    assert parse_hand("Ac Kd Jd 7h 7s 2c 2d") == parse_hand("Ac,Kd,Jd,7h,7s,2c,2d")


def test_parse_hand_wrong_count():
    with pytest.raises(CardError):
        parse_hand("AcKdJd")  # only 3 cards


def test_parse_hand_duplicates():
    with pytest.raises(CardError):
        parse_hand("AcAcKdJd7h7s2c")


def test_format_card_roundtrip():
    assert format_card((13, "h")) == "Kh"
    assert format_card((14, "c")) == "Ac"
    assert format_card((10, "d")) == "Td"
