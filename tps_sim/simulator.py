"""Monte-Carlo simulation of a Taiwanese Poker hand.

Game model
----------
Each player holds 7 cards and splits them into three hands:

* **top**    - 1 card, scored as a Texas Hold'em hand with the board (weight 1)
* **middle** - 2 cards, scored as a Texas Hold'em hand with the board (weight 2)
* **bottom** - 4 cards, scored as an Omaha hand with the board (weight 3)

Two separate five-card community boards are dealt. Every one of a player's
three hands is scored against *each* board. Against each opponent, on each
board, you earn ``+weight`` for winning a hand, ``-weight`` for losing it, and
0 for a tie. The total for a simulated hand is summed over both boards and all
opponents.

The optimizer tries all 105 ways to split your fixed 7 cards and reports the
split with the best average score across the simulations. Opponents are dealt
random hands each simulation and set themselves with the same split logic,
maximizing their own weighted made-hand strength for that hand's boards.
"""

from __future__ import annotations

import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .cards import Card, FULL_DECK
from .evaluator import best_holdem, best_omaha_pair

# Position weights: top, middle, bottom.
WEIGHTS = (1, 2, 3)

# Parallelism. Runs with more than ``PARALLEL_THRESHOLD`` simulations are split
# across worker processes (each simulation is independent, so the work is
# embarrassingly parallel). ``DEFAULT_WORKERS`` is the default process count.
PARALLEL_THRESHOLD = 1000
DEFAULT_WORKERS = 4

# "Home run" bonus: awarded when a player beats an opponent on all three hands
# on both boards (all 6 matchups). The 12 base points become 20. Applied
# symmetrically: being swept by an opponent costs the same bonus.
HOME_RUN_BONUS = 8

# All 105 ways to partition seven card indices into (top=1, middle=2, bottom=4).
# A split is (top_index, (mid_i, mid_j), (bot_i, bot_j, bot_k, bot_l)).
Split = Tuple[int, Tuple[int, int], Tuple[int, int, int, int]]


def _build_splits() -> List[Split]:
    splits: List[Split] = []
    for top in range(7):
        rest = [i for i in range(7) if i != top]
        for mid in combinations(rest, 2):
            mid_set = set(mid)
            bottom = tuple(i for i in rest if i not in mid_set)
            splits.append((top, mid, bottom))  # type: ignore[arg-type]
    return splits


SPLITS: List[Split] = _build_splits()
# Index pairs used to pick the two card positions for each split component.
_PAIR_INDICES: List[Tuple[int, int]] = list(combinations(range(7), 2))
_QUAD_INDICES: List[Tuple[int, int, int, int]] = list(combinations(range(7), 4))


# Per-board strength tables for one 7-card holding.
@dataclass
class HandBoardValues:
    """Made-hand strength of each possible split component on one board."""

    single: List[int]                                   # indexed by card index
    pair: Dict[Tuple[int, int], int]                    # holdem value of a card pair
    quad: Dict[Tuple[int, int, int, int], int]          # omaha value of a card quad


def _position_values(hand: Sequence[Card], board: Sequence[Card]) -> HandBoardValues:
    """Compute strength of every split component of ``hand`` on one ``board``."""
    board = list(board)
    board_triples = list(combinations(board, 3))

    single = [best_holdem([hand[i], *board]) for i in range(7)]

    pair: Dict[Tuple[int, int], int] = {}
    omaha_pair: Dict[Tuple[int, int], int] = {}
    for pr in _PAIR_INDICES:
        two = (hand[pr[0]], hand[pr[1]])
        pair[pr] = best_holdem([*two, *board])
        omaha_pair[pr] = best_omaha_pair(two, board_triples)

    # A four-card Omaha hand uses exactly two hole cards, so its value is the
    # best of its six constituent pairs (each already maxed over board triples).
    quad: Dict[Tuple[int, int, int, int], int] = {}
    for q in _QUAD_INDICES:
        quad[q] = max(omaha_pair[sp] for sp in combinations(q, 2))

    return HandBoardValues(single=single, pair=pair, quad=quad)


# Number of random boards an opponent samples to estimate expected hand
# strength when setting blind. Higher means stronger, more accurate opponents
# (and slower runs). The best split is stable across this value; mainly the
# score magnitude shifts as the field gets tougher.
BLIND_SAMPLE_BOARDS = 20

# A strategy chooses which split a player uses, given their hand, the per-board
# value tables for the *actual* boards, and the run's RNG. A blind strategy must
# ignore ``board_values`` when deciding (it may only be used by the caller,
# afterwards, to score the chosen split).
Strategy = Callable[
    [Sequence[Card], List["HandBoardValues"], "random.Random"], Split
]


def _best_split_by_weighted_strength(
    single, pair, quad
) -> Split:
    """Pick the split maximizing weighted (top + 2*mid + 3*bottom) strength."""
    w_top, w_mid, w_bot = WEIGHTS
    best_score = None
    best_split: Split = SPLITS[0]
    for top, mid, bottom in SPLITS:
        score = w_top * single[top] + w_mid * pair[mid] + w_bot * quad[bottom]
        if best_score is None or score > best_score:
            best_score = score
            best_split = (top, mid, bottom)
    return best_split


def greedy_best_strategy(
    hand: Sequence[Card],
    board_values: List[HandBoardValues],
    rng: "random.Random",
) -> Split:
    """Board-aware: pick the split strongest on the *actual* boards in play.

    Retained for comparison. This peeks at the real boards, so it is not a
    blind/symmetric model.
    """
    w_top, w_mid, w_bot = WEIGHTS
    best_score = None
    best_split: Split = SPLITS[0]
    for top, mid, bottom in SPLITS:
        score = 0
        for bv in board_values:
            score += w_top * bv.single[top] + w_mid * bv.pair[mid] + w_bot * bv.quad[bottom]
        if best_score is None or score > best_score:
            best_score = score
            best_split = (top, mid, bottom)
    return best_split


class BlindStrategy:
    """A blind strategy: set a split *without* seeing the real boards.

    Each opponent estimates the expected made-hand strength of every split
    component by averaging the evaluator over ``n_samples`` random boards it
    cannot see, then commits to the split maximizing expected weighted strength.
    This mirrors the optimizer's reasoning (same evaluator, same weighting) while
    staying blind to the boards that will actually be dealt.

    Implemented as a plain callable class (rather than a closure) so instances
    are picklable and can be shipped to worker processes for parallel runs.
    """

    def __init__(self, n_samples: int = BLIND_SAMPLE_BOARDS):
        self.n_samples = n_samples

    def __call__(
        self,
        hand: Sequence[Card],
        board_values: List[HandBoardValues],
        rng: "random.Random",
    ) -> Split:
        single_tot = [0] * 7
        pair_tot = {pr: 0 for pr in _PAIR_INDICES}
        quad_tot = {q: 0 for q in _QUAD_INDICES}

        hand_set = set(hand)
        sample_deck = [c for c in FULL_DECK if c not in hand_set]
        for _ in range(self.n_samples):
            board = rng.sample(sample_deck, 5)
            bv = _position_values(hand, board)
            for i in range(7):
                single_tot[i] += bv.single[i]
            for pr in _PAIR_INDICES:
                pair_tot[pr] += bv.pair[pr]
            for q in _QUAD_INDICES:
                quad_tot[q] += bv.quad[q]

        return _best_split_by_weighted_strength(single_tot, pair_tot, quad_tot)


def make_blind_strategy(n_samples: int = BLIND_SAMPLE_BOARDS) -> Strategy:
    """Build a blind strategy that samples ``n_samples`` boards. See
    :class:`BlindStrategy`."""
    return BlindStrategy(n_samples)


# Default field: opponents set blind (symmetric, no board knowledge).
blind_best_strategy: Strategy = make_blind_strategy()


def _cmp(a: int, b: int) -> int:
    return (a > b) - (a < b)


@dataclass
class SplitResult:
    """Aggregated simulation outcome for one of the player's candidate splits.

    All averages are normalized *per hand and per opponent*, so they are
    comparable regardless of how many opponents were in the field.
    """

    split: Split
    avg_score: float
    avg_top: float
    avg_middle: float
    avg_bottom: float
    avg_bonus: float


@dataclass
class SimulationReport:
    """Full result of a simulation run, ranked best split first."""

    player_cards: List[Card]
    opponents: int
    simulations: int
    ranked: List[SplitResult]

    @property
    def best(self) -> SplitResult:
        return self.ranked[0]


# Totals accumulated by a simulation chunk: five parallel arrays indexed by
# split (score, top, middle, bottom, home-run bonus).
ChunkTotals = Tuple[List[int], List[int], List[int], List[int], List[int]]


def _simulate_chunk(
    player_cards: Sequence[Card],
    remaining_deck: Sequence[Card],
    opponents: int,
    n_sims: int,
    seed: Optional[int],
    opponent_strategy: Strategy,
    progress: Optional[Callable[[int, int], None]] = None,
    progress_base: int = 0,
    progress_total: int = 0,
) -> ChunkTotals:
    """Run ``n_sims`` independent simulations and return the summed totals.

    This is the unit of work for both the single-threaded path and each worker
    process in a parallel run. It owns its own RNG (seeded from ``seed``) so a
    chunk's result depends only on its inputs, never on where it runs.

    ``progress`` is only used by the in-process (single-threaded) path; worker
    processes cannot report progress back to the parent and leave it ``None``.
    """
    rng = random.Random(seed)

    n_splits = len(SPLITS)
    total_score = [0] * n_splits
    total_top = [0] * n_splits
    total_mid = [0] * n_splits
    total_bot = [0] * n_splits
    total_bonus = [0] * n_splits

    w_top, w_mid, w_bot = WEIGHTS

    for sim in range(n_sims):
        # Deal two boards and the opponents' hands from the remaining deck.
        deal = rng.sample(remaining_deck, 10 + 7 * opponents)
        board1 = deal[0:5]
        board2 = deal[5:10]
        boards = [board1, board2]

        opp_hands = [
            deal[10 + 7 * k : 10 + 7 * (k + 1)] for k in range(opponents)
        ]

        # Player's strength tables for each board (hand is fixed).
        player_bv = [_position_values(player_cards, b) for b in boards]

        # Each opponent commits a split (blind), then we read off the made
        # values of that split on the actual boards for scoring.
        opp_values = []  # list[opponent] -> list[board] -> (top, mid, bot)
        for hand in opp_hands:
            obv = [_position_values(hand, b) for b in boards]
            o_top, o_mid, o_bot = opponent_strategy(hand, obv, rng)
            opp_values.append(
                [
                    (bv.single[o_top], bv.pair[o_mid], bv.quad[o_bot])
                    for bv in obv
                ]
            )

        # Score every candidate split for the player against the field. The
        # home-run bonus is per opponent, so comparisons are grouped by opponent
        # across both boards (all 6 matchups).
        p0, p1 = player_bv
        for si, (top, mid, bottom) in enumerate(SPLITS):
            pt0, pm0, pb0 = p0.single[top], p0.pair[mid], p0.quad[bottom]
            pt1, pm1, pb1 = p1.single[top], p1.pair[mid], p1.quad[bottom]
            s_top = s_mid = s_bot = bonus = 0
            for ov in opp_values:
                (ot0, om0, ob0), (ot1, om1, ob1) = ov
                ct0, cm0, cb0 = _cmp(pt0, ot0), _cmp(pm0, om0), _cmp(pb0, ob0)
                ct1, cm1, cb1 = _cmp(pt1, ot1), _cmp(pm1, om1), _cmp(pb1, ob1)
                s_top += ct0 + ct1
                s_mid += cm0 + cm1
                s_bot += cb0 + cb1
                if ct0 == ct1 == cm0 == cm1 == cb0 == cb1 == 1:
                    bonus += HOME_RUN_BONUS  # player sweeps this opponent
                elif ct0 == ct1 == cm0 == cm1 == cb0 == cb1 == -1:
                    bonus -= HOME_RUN_BONUS  # opponent sweeps the player
            total_top[si] += s_top
            total_mid[si] += s_mid
            total_bot[si] += s_bot
            total_bonus[si] += bonus
            total_score[si] += (
                w_top * s_top + w_mid * s_mid + w_bot * s_bot + bonus
            )

        if progress is not None:
            progress(progress_base + sim + 1, progress_total)

    return total_score, total_top, total_mid, total_bot, total_bonus


def _split_counts(total: int, n_chunks: int) -> List[int]:
    """Divide ``total`` simulations into ``n_chunks`` sizes as evenly as
    possible. Empty chunks are dropped."""
    n_chunks = min(n_chunks, total)
    base, extra = divmod(total, n_chunks)
    return [base + (1 if i < extra else 0) for i in range(n_chunks)]


def _finalize(
    totals: ChunkTotals,
    player_cards: List[Card],
    opponents: int,
    simulations: int,
) -> SimulationReport:
    """Normalize summed totals into a ranked report."""
    total_score, total_top, total_mid, total_bot, total_bonus = totals
    w_top, w_mid, w_bot = WEIGHTS

    # Normalize per hand *and* per opponent, so scores are comparable across
    # different field sizes (6 vs 3 opponents == 2 vs 1 opponent).
    norm = simulations * (opponents if opponents else 1)
    ranked = [
        SplitResult(
            split=SPLITS[si],
            avg_score=total_score[si] / norm,
            avg_top=w_top * total_top[si] / norm,
            avg_middle=w_mid * total_mid[si] / norm,
            avg_bottom=w_bot * total_bot[si] / norm,
            avg_bonus=total_bonus[si] / norm,
        )
        for si in range(len(SPLITS))
    ]
    ranked.sort(key=lambda r: r.avg_score, reverse=True)

    return SimulationReport(
        player_cards=player_cards,
        opponents=opponents,
        simulations=simulations,
        ranked=ranked,
    )


def _is_picklable(obj: object) -> bool:
    """Whether ``obj`` can be shipped to a worker process."""
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _run_parallel(
    player_cards: List[Card],
    remaining_deck: List[Card],
    opponents: int,
    simulations: int,
    seed: Optional[int],
    opponent_strategy: Strategy,
    workers: int,
    progress: Optional[Callable[[int, int], None]],
) -> ChunkTotals:
    """Run the simulation across ``workers`` processes and sum their totals.

    Simulations are divided into chunks (more than one per worker, for load
    balancing and smoother progress). Each chunk is seeded deterministically
    from the master seed, so a given ``(seed, workers)`` is reproducible and the
    summed result does not depend on the order chunks happen to finish in.
    """
    # Several chunks per worker: better load balancing and finer progress, at a
    # negligible per-chunk dispatch cost.
    chunk_sizes = _split_counts(simulations, workers * 4)

    # Derive a reproducible per-chunk seed stream from the master seed. When
    # ``seed`` is None these are random (and the run is non-reproducible), which
    # matches the single-threaded contract.
    master = random.Random(seed)
    chunk_seeds = [master.randrange(2 ** 63) for _ in chunk_sizes]

    n_splits = len(SPLITS)
    totals: ChunkTotals = (
        [0] * n_splits, [0] * n_splits, [0] * n_splits,
        [0] * n_splits, [0] * n_splits,
    )

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _simulate_chunk,
                player_cards,
                remaining_deck,
                opponents,
                size,
                chunk_seed,
                opponent_strategy,
            ): size
            for size, chunk_seed in zip(chunk_sizes, chunk_seeds)
        }
        for future in as_completed(futures):
            chunk = future.result()
            for acc, part in zip(totals, chunk):
                for si in range(n_splits):
                    acc[si] += part[si]
            done += futures[future]
            if progress is not None:
                progress(done, simulations)

    return totals


def simulate(
    player_cards: Sequence[Card],
    opponents: int = 1,
    simulations: int = 1000,
    seed: Optional[int] = None,
    opponent_strategy: Strategy = blind_best_strategy,
    progress: Optional[Callable[[int, int], None]] = None,
    workers: int = DEFAULT_WORKERS,
) -> SimulationReport:
    """Run the Monte-Carlo simulation and return a ranked report.

    Parameters
    ----------
    player_cards:
        Exactly seven distinct cards held by the player (the split is fixed
        across all simulations and chosen to maximize the average).
    opponents:
        Number of opponents (1-5; limited by the 52-card deck).
    simulations:
        Number of random deals to average over.
    seed:
        Optional RNG seed for reproducible runs.
    opponent_strategy:
        How each opponent sets its hand each deal.
    progress:
        Optional callback invoked as ``progress(done, total)``.
    workers:
        Number of worker processes to use. Runs with more than
        ``PARALLEL_THRESHOLD`` simulations are split across this many processes;
        set to 1 to force single-threaded execution. Defaults to
        ``DEFAULT_WORKERS``.

    Determinism
    -----------
    With a fixed ``seed`` a run is reproducible for a given ``workers`` count.
    The single-threaded path (``workers <= 1`` or ``simulations <=
    PARALLEL_THRESHOLD``) reproduces results bit-for-bit across runs; changing
    ``workers`` changes how the RNG stream is partitioned, so parallel results
    need not match single-threaded ones (but are self-consistent).
    """
    player_cards = list(player_cards)
    if len(player_cards) != 7:
        raise ValueError("Player must hold exactly 7 cards.")
    if opponents < 0:
        raise ValueError("Opponents cannot be negative.")
    if simulations < 1:
        raise ValueError("Need at least one simulation.")
    if workers < 1:
        raise ValueError("Need at least one worker.")

    needed = 7 + 10 + 7 * opponents  # player + two boards + opponents
    if needed > 52:
        max_opp = (52 - 7 - 10) // 7
        raise ValueError(
            f"{opponents} opponents need {needed} cards but the deck holds 52; "
            f"at most {max_opp} opponents are possible."
        )

    player_set = set(player_cards)
    remaining_deck = [c for c in FULL_DECK if c not in player_set]

    # Parallelize only when it is worth the process overhead and the strategy
    # can be shipped to workers; otherwise fall back to a single in-process run.
    if (
        simulations > PARALLEL_THRESHOLD
        and workers > 1
        and _is_picklable(opponent_strategy)
    ):
        totals = _run_parallel(
            player_cards,
            remaining_deck,
            opponents,
            simulations,
            seed,
            opponent_strategy,
            workers,
            progress,
        )
    else:
        totals = _simulate_chunk(
            player_cards,
            remaining_deck,
            opponents,
            simulations,
            seed,
            opponent_strategy,
            progress=progress,
            progress_base=0,
            progress_total=simulations,
        )

    return _finalize(totals, player_cards, opponents, simulations)


def describe_split_hands(player_cards: Sequence[Card], split: Split) -> Dict[str, List[Card]]:
    """Map a split to its actual top/middle/bottom cards."""
    top, mid, bottom = split
    return {
        "top": [player_cards[top]],
        "middle": [player_cards[i] for i in mid],
        "bottom": [player_cards[i] for i in bottom],
    }
