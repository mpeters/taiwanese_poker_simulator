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


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------

def test_negative_opponents_rejected():
    hand = parse_hand("AcKdJd7h7s2c2d")
    with pytest.raises(ValueError, match="negative"):
        simulate(hand, opponents=-1, simulations=10)


def test_zero_simulations_rejected():
    hand = parse_hand("AcKdJd7h7s2c2d")
    with pytest.raises(ValueError, match="at least one simulation"):
        simulate(hand, opponents=1, simulations=0)


def test_zero_workers_rejected():
    hand = parse_hand("AcKdJd7h7s2c2d")
    with pytest.raises(ValueError, match="at least one worker"):
        simulate(hand, opponents=1, simulations=10, workers=0)


def test_zero_opponents_normalizes_without_division_error():
    # With no opponents every matchup is empty, so all scores are exactly zero,
    # and normalization must fall back to a divisor of 1 (no ZeroDivisionError).
    hand = parse_hand("AcKdJd7h7s2c2d")
    report = simulate(hand, opponents=0, simulations=10, seed=1)
    assert report.opponents == 0
    assert len(report.ranked) == 105
    for r in report.ranked:
        assert r.avg_score == 0.0


def test_max_opponents_allowed():
    hand = parse_hand("AcKdJd7h7s2c2d")
    report = simulate(hand, opponents=5, simulations=10, seed=1)
    assert report.opponents == 5


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def test_progress_callback_reaches_total():
    hand = parse_hand("AcKdJd7h7s2c2d")
    calls = []
    simulate(
        hand,
        opponents=1,
        simulations=15,
        seed=1,
        progress=lambda done, total: calls.append((done, total)),
        workers=1,
    )
    assert calls, "progress callback was never invoked"
    assert calls[-1] == (15, 15)
    # done must be monotonically non-decreasing and never exceed the total.
    for done, total in calls:
        assert 0 < done <= total == 15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_split_counts_distributes_evenly():
    from tps_sim.simulator import _split_counts

    assert _split_counts(10, 4) == [3, 3, 2, 2]
    assert sum(_split_counts(10, 4)) == 10
    assert sum(_split_counts(100, 7)) == 100


def test_split_counts_more_chunks_than_work():
    from tps_sim.simulator import _split_counts

    # More requested chunks than simulations: capped, no empty chunks.
    counts = _split_counts(3, 8)
    assert counts == [1, 1, 1]
    assert all(c > 0 for c in counts)


def test_is_picklable():
    from tps_sim.simulator import _is_picklable, blind_best_strategy

    assert _is_picklable(blind_best_strategy) is True
    assert _is_picklable(make_blind_strategy(5)) is True
    # A lambda cannot be pickled and must be shipped nowhere.
    assert _is_picklable(lambda *a: None) is False


def test_describe_split_hands_maps_indices():
    hand = parse_hand("AcKdJd7h7s2c2d")
    split = (0, (1, 2), (3, 4, 5, 6))
    hands = describe_split_hands(hand, split)
    assert hands["top"] == [hand[0]]
    assert hands["middle"] == [hand[1], hand[2]]
    assert hands["bottom"] == [hand[3], hand[4], hand[5], hand[6]]


# ---------------------------------------------------------------------------
# Parallel execution path
# ---------------------------------------------------------------------------

def test_parallel_path_runs():
    # Any workers > 1 forces the parallel branch regardless of simulation count;
    # confirm it produces a full ranked report.
    import tps_sim.simulator as sim

    hand = parse_hand("AcKdJd7h7s2c2d")
    report = sim.simulate(hand, opponents=1, simulations=8, seed=3, workers=2)
    assert len(report.ranked) == 105
    assert report.simulations == 8


def test_parallel_is_reproducible_for_fixed_workers():
    import tps_sim.simulator as sim

    hand = parse_hand("AcKdJd7h7s2c2d")
    r1 = sim.simulate(hand, opponents=1, simulations=8, seed=3, workers=2)
    r2 = sim.simulate(hand, opponents=1, simulations=8, seed=3, workers=2)
    assert r1.best.split == r2.best.split
    assert r1.best.avg_score == r2.best.avg_score


def test_parallel_progress_reaches_total():
    import tps_sim.simulator as sim

    hand = parse_hand("AcKdJd7h7s2c2d")
    calls = []
    sim.simulate(
        hand,
        opponents=1,
        simulations=8,
        seed=3,
        workers=2,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls
    assert calls[-1] == (8, 8)


def test_auto_stays_single_threaded_below_threshold(monkeypatch):
    # With workers unspecified, a small run must not spawn the parallel branch.
    import tps_sim.simulator as sim

    def boom(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("parallel branch taken below threshold")

    monkeypatch.setattr(sim, "_run_parallel", boom)
    hand = parse_hand("AcKdJd7h7s2c2d")
    report = sim.simulate(hand, opponents=1, simulations=8, seed=3)
    assert report.simulations == 8


def test_auto_parallelizes_above_threshold(monkeypatch):
    # With workers unspecified, a run past the threshold auto-parallelizes using
    # DEFAULT_WORKERS. Lower the threshold so the test stays fast.
    import tps_sim.simulator as sim

    seen = {}

    real = sim._run_parallel

    def spy(*args, **kwargs):
        # Signature: (..., opponent_strategy, workers, progress)
        seen["workers"] = args[6]
        return real(*args, **kwargs)

    monkeypatch.setattr(sim, "PARALLEL_THRESHOLD", 1)
    monkeypatch.setattr(sim, "_run_parallel", spy)
    hand = parse_hand("AcKdJd7h7s2c2d")
    sim.simulate(hand, opponents=1, simulations=8, seed=3)
    assert seen["workers"] == sim.DEFAULT_WORKERS


def test_explicit_workers_override_threshold(monkeypatch):
    # An explicit worker count parallelizes even when below the threshold.
    import tps_sim.simulator as sim

    seen = {}
    real = sim._run_parallel

    def spy(*args, **kwargs):
        seen["workers"] = args[6]
        return real(*args, **kwargs)

    monkeypatch.setattr(sim, "_run_parallel", spy)
    hand = parse_hand("AcKdJd7h7s2c2d")
    sim.simulate(hand, opponents=1, simulations=8, seed=3, workers=3)
    assert seen["workers"] == 3
