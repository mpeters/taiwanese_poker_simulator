import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tps_sim.cards import parse_hand
from tps_sim.simulator import (
    HOME_RUN_BONUS,
    SPLITS,
    describe_split_hands,
    greedy_best_strategy,
    make_blind_strategy,
    simulate,
)


def test_there_are_105_splits():
    assert len(SPLITS) == 105
    # Every split must partition all 7 indices into 1 + 2 + 4.
    for top, mid, bottom in SPLITS:
        assert len({top, *mid, *bottom}) == 7
        assert len(mid) == 2 and len(bottom) == 4


def test_simulate_is_deterministic_with_seed():
    hand = parse_hand("AcKdJd7h7s2c2d")
    r1 = simulate(hand, opponents=2, simulations=100, seed=42)
    r2 = simulate(hand, opponents=2, simulations=100, seed=42)
    assert r1.best.split == r2.best.split
    assert r1.best.avg_score == r2.best.avg_score


def test_report_ranked_descending():
    hand = parse_hand("AcKdJd7h7s2c2d")
    report = simulate(hand, opponents=1, simulations=50, seed=1)
    scores = [r.avg_score for r in report.ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(report.ranked) == 105


def test_position_averages_sum_to_total():
    hand = parse_hand("AcKdJd7h7s2c2d")
    report = simulate(hand, opponents=2, simulations=80, seed=7)
    for r in report.ranked:
        total = r.avg_top + r.avg_middle + r.avg_bottom + r.avg_bonus
        assert abs(total - r.avg_score) < 1e-9


def test_best_split_cards_match_player_hand():
    hand = parse_hand("AcKdJd7h7s2c2d")
    report = simulate(hand, opponents=1, simulations=30, seed=3)
    hands = describe_split_hands(hand, report.best.split)
    all_cards = hands["top"] + hands["middle"] + hands["bottom"]
    assert sorted(all_cards) == sorted(hand)
    assert len(hands["top"]) == 1
    assert len(hands["middle"]) == 2
    assert len(hands["bottom"]) == 4


def test_too_many_opponents_rejected():
    hand = parse_hand("AcKdJd7h7s2c2d")
    with pytest.raises(ValueError):
        simulate(hand, opponents=6, simulations=10)


def test_wrong_card_count_rejected():
    with pytest.raises(ValueError):
        simulate([(14, "c")], opponents=1, simulations=10)


def test_home_run_bonus_is_bounded_multiple_of_eight():
    hand = parse_hand("AcKdJd7h7s2c2d")
    sims = 100
    opponents = 3
    report = simulate(hand, opponents=opponents, simulations=sims, seed=11)
    for r in report.ranked:
        # Averages are normalized per hand and per opponent, so the summed raw
        # bonus is avg_bonus * sims * opponents.
        total_bonus = round(r.avg_bonus * sims * opponents)
        # Every sweep contributes +/- HOME_RUN_BONUS per opponent.
        assert total_bonus % HOME_RUN_BONUS == 0
        # At most one (anti-)sweep per opponent per simulation -> per-opponent
        # average bonus cannot exceed the bonus itself.
        assert abs(r.avg_bonus) <= HOME_RUN_BONUS + 1e-9


def test_scores_normalized_per_opponent():
    # Max possible per-opponent, per-hand score is a sweep (12) plus the home-run
    # bonus (8) = 20. With per-opponent normalization this bound holds for any
    # field size; without it, a 3-opponent run could reach ~60.
    hand = parse_hand("AcAdAhAsKdKhKs")
    for opponents in (1, 3, 5):
        report = simulate(hand, opponents=opponents, simulations=60, seed=8)
        for r in report.ranked:
            assert -20 - 1e-9 <= r.avg_score <= 20 + 1e-9


def test_strong_hand_earns_positive_bonus():
    # A monster (quad-friendly, all high) against a single opponent should sweep
    # often enough that the best split's average bonus is positive.
    hand = parse_hand("AcAdAhAsKdKhKs")
    report = simulate(hand, opponents=1, simulations=200, seed=4)
    assert report.best.avg_bonus > 0


def test_blind_strategy_ignores_actual_boards():
    # The blind strategy must choose using only the hand + RNG, never the
    # board_values it is handed. Feeding it different board_values with the same
    # RNG state must yield the same split.
    import random

    from tps_sim.simulator import _position_values

    hand = parse_hand("AcKdJd7h7s2c2d")
    strat = make_blind_strategy(6)

    board_a = parse_hand("2h3h4h5h8c", expected=5)
    board_b = parse_hand("AsQsTs9s6d", expected=5)
    bv_a = [_position_values(hand, board_a), _position_values(hand, board_a)]
    bv_b = [_position_values(hand, board_b), _position_values(hand, board_b)]

    split1 = strat(hand, bv_a, random.Random(99))
    split2 = strat(hand, bv_b, random.Random(99))
    assert split1 == split2  # board_values had no effect


def test_board_aware_strategy_still_works():
    hand = parse_hand("AcKdJd7h7s2c2d")
    report = simulate(
        hand,
        opponents=2,
        simulations=40,
        seed=5,
        opponent_strategy=greedy_best_strategy,
    )
    assert len(report.ranked) == 105


def test_custom_sample_count_runs():
    hand = parse_hand("AcKdJd7h7s2c2d")
    report = simulate(
        hand,
        opponents=1,
        simulations=30,
        seed=5,
        opponent_strategy=make_blind_strategy(3),
    )
    assert report.best.avg_score == report.ranked[0].avg_score
