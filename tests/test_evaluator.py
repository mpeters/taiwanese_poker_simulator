import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itertools import combinations

import pytest

from tps_sim.cards import parse_card
from tps_sim.evaluator import (
    best_holdem,
    best_omaha,
    category_name,
    category_of,
    eval5,
)


def cards(text):
    text = text.replace(" ", "")
    return [parse_card(text[i : i + 2]) for i in range(0, len(text), 2)]


def test_category_ordering():
    # One representative hand per category, strictly increasing strength.
    hands = {
        "high": "Ah Kd 9c 5s 2h",
        "pair": "Ah Ad 9c 5s 2h",
        "two_pair": "Ah Ad 9c 9s 2h",
        "trips": "Ah Ad Ac 9s 2h",
        "straight": "Ah Kd Qc Js Th",
        "flush": "Ah Jh 9h 5h 2h",
        "full_house": "Ah Ad Ac 9s 9h",
        "quads": "Ah Ad Ac As 9h",
        "straight_flush": "Ah Kh Qh Jh Th",
    }
    order = [
        "high", "pair", "two_pair", "trips", "straight",
        "flush", "full_house", "quads", "straight_flush",
    ]
    values = [eval5(cards(hands[name])) for name in order]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_wheel_straight_is_five_high():
    wheel = eval5(cards("5h4d3c2sAh"))
    six_high = eval5(cards("6h5d4c3s2h"))
    assert category_of(wheel) == category_of(six_high) == 4  # both straights
    assert wheel < six_high  # wheel is the lowest straight


def test_wheel_straight_flush_beaten_by_six_high():
    wheel_sf = eval5(cards("5h4h3h2hAh"))
    six_sf = eval5(cards("6h5h4h3h2h"))
    assert category_of(wheel_sf) == 8
    assert wheel_sf < six_sf


def test_kicker_matters():
    ace_king = eval5(cards("AhAdKc5s2h"))
    ace_queen = eval5(cards("AhAdQc5s2h"))
    assert ace_king > ace_queen


def test_flush_compares_by_high_cards():
    higher = eval5(cards("AhJh9h5h2h"))
    lower = eval5(cards("AhTh9h5h2h"))
    assert higher > lower


def test_category_names():
    assert category_name(eval5(cards("AhKhQhJhTh"))) == "Straight Flush"
    assert category_name(eval5(cards("AhAdAcAs9h"))) == "Four of a Kind"


def test_best_holdem_single_card():
    # One hole card + board: best 5 of 6 must find the flush on board+hole.
    board = cards("Kh9h5h2h7d")
    with_hole = best_holdem(cards("Ah") + board)  # Ah completes nut flush
    assert category_of(with_hole) == 5  # flush
    # Brute-force check against all 5-card combos.
    assert with_hole == max(eval5(c) for c in combinations(cards("Ah") + board, 5))


def test_best_omaha_requires_exactly_two_hole():
    # Board is four hearts + an off card. Hold'em would make a flush with ONE
    # heart, but Omaha requires exactly two hole cards, so a single heart hole
    # cannot complete the flush.
    board = cards("Kh9h5h2h7d")
    hole = cards("AhQsJsTs")  # only one heart (Ah)
    value = best_omaha(hole, board)
    # Must use exactly 2 hole + 3 board; cannot make a heart flush here.
    assert category_of(value) != 5 or _flush_possible_with_two(hole, board)


def _flush_possible_with_two(hole, board):
    # Helper: is there any 2-hole + 3-board combo that is a flush?
    for two in combinations(hole, 2):
        for three in combinations(board, 3):
            if category_of(eval5(two + three)) == 5:
                return True
    return False


def test_best_omaha_matches_bruteforce():
    board = cards("Kh9h5h2h7d")
    hole = cards("AhKhQdJc")
    expected = -1
    for two in combinations(hole, 2):
        for three in combinations(board, 3):
            expected = max(expected, eval5(two + three))
    assert best_omaha(hole, board) == expected


def test_category_of_and_name_for_every_category():
    from tps_sim.evaluator import CATEGORY_NAMES

    representatives = {
        0: "Ah Kd 9c 5s 2h",
        1: "Ah Ad 9c 5s 2h",
        2: "Ah Ad 9c 9s 2h",
        3: "Ah Ad Ac 9s 2h",
        4: "Ah Kd Qc Js Th",
        5: "Ah Jh 9h 5h 2h",
        6: "Ah Ad Ac 9s 9h",
        7: "Ah Ad Ac As 9h",
        8: "Ah Kh Qh Jh Th",
    }
    for expected_cat, text in representatives.items():
        packed = eval5(cards(text))
        assert category_of(packed) == expected_cat
        assert category_name(packed) == CATEGORY_NAMES[expected_cat]


def test_best_holdem_exactly_five_cards_delegates_to_eval5():
    five = cards("AhKhQhJhTh")
    assert best_holdem(five) == eval5(five)


def test_best_holdem_seven_cards_matches_bruteforce():
    pool = cards("AhKhQdJc9h5h2h")  # 7 cards (hole pair + board)
    expected = max(eval5(c) for c in combinations(pool, 5))
    assert best_holdem(pool) == expected


def test_best_omaha_pair_matches_bruteforce():
    from tps_sim.evaluator import best_omaha_pair

    board = cards("Kh9h5h2h7d")
    triples = list(combinations(board, 3))
    two = tuple(cards("AhKh"))
    expected = max(eval5(two + t) for t in triples)
    assert best_omaha_pair(two, triples) == expected


def test_full_house_beats_flush():
    full_house = eval5(cards("Ah Ad Ac 9s 9h"))
    flush = eval5(cards("Ah Jh 9h 5h 2h"))
    assert full_house > flush


def test_two_pair_tiebreak_by_kicker():
    higher_kicker = eval5(cards("Ah Ad 9c 9s Kh"))
    lower_kicker = eval5(cards("Ah Ad 9c 9s Qh"))
    assert higher_kicker > lower_kicker
