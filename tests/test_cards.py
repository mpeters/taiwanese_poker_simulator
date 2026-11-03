import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tps_sim.cards import (
    CardError,
    format_card,
    format_cards,
    parse_card,
    parse_hand,
)


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


def test_parse_card_empty_and_single_char():
    with pytest.raises(CardError):
        parse_card("")
    with pytest.raises(CardError):
        parse_card("A")


def test_parse_hand_odd_length_chunk_rejected():
    # A chunk with an odd number of characters cannot be split into 2-char cards.
    with pytest.raises(CardError, match="odd number"):
        parse_hand("AcKdJd7h7s2c2")


def test_parse_hand_custom_expected_count():
    board = parse_hand("2h3h4h5h8c", expected=5)
    assert len(board) == 5
    assert board[0] == (2, "h")


def test_parse_hand_mixed_separators():
    # Mixing run-together groups, spaces, and commas is tolerated.
    hand = parse_hand("AcKd, Jd7h 7s,2c2d")
    assert len(hand) == 7


def test_parse_hand_duplicate_across_groups():
    with pytest.raises(CardError, match="Duplicate"):
        parse_hand("Ac Kd Jd 7h 7s 2c Ac")


def test_format_card_with_symbols():
    assert format_card((13, "h"), symbols=True) == "K♥"
    assert format_card((14, "s"), symbols=True) == "A♠"


def test_format_cards_joins_with_separator():
    cards_list = [(14, "c"), (13, "d")]
    assert format_cards(cards_list) == "Ac Kd"
    assert format_cards(cards_list, sep=",") == "Ac,Kd"
    assert format_cards(cards_list, symbols=True) == "A♣ K♦"


def test_full_deck_is_52_unique_cards():
    from tps_sim.cards import FULL_DECK

    assert len(FULL_DECK) == 52
    assert len(set(FULL_DECK)) == 52
