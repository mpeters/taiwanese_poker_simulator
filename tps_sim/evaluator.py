"""Five-card poker hand evaluation plus Hold'em / Omaha best-hand helpers.

:func:`eval5` collapses a five-card hand into a single integer that can be
compared directly: a larger integer always beats a smaller one, with ties
meaning equal hand strength. Packing everything into one int keeps the hot
simulation loop fast.

Hand categories (most significant part of the packed value):

    8 straight flush   5 flush            2 two pair
    7 four of a kind   4 straight         1 one pair
    6 full house       3 three of a kind  0 high card
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

from .cards import Card

# Each hand is packed as: category, then up to 5 tie-break ranks.
# value = ((((category * 15 + t0) * 15 + t1) * 15 + t2) * 15 + t3) * 15 + t4
# Ranks are all < 15, so category always dominates the comparison.


def eval5(cards: Sequence[Card]) -> int:
    """Return a comparable integer strength for exactly five cards.

    Hot path: avoids :class:`collections.Counter` and keyed sorts in favour of
    a single sort of five ranks plus a manual grouping, which is markedly faster
    under the millions of calls the simulator makes.
    """
    c0, c1, c2, c3, c4 = cards
    ranks = [c0[0], c1[0], c2[0], c3[0], c4[0]]
    ranks.sort(reverse=True)
    r0, r1, r2, r3, r4 = ranks

    is_flush = c0[1] == c1[1] == c2[1] == c3[1] == c4[1]

    # Group equal ranks (ranks are sorted, so duplicates are adjacent).
    groups = []  # list of (count, rank), later sorted by (count, rank) desc
    prev = r0
    count = 1
    for x in (r1, r2, r3, r4):
        if x == prev:
            count += 1
        else:
            groups.append((count, prev))
            prev = x
            count = 1
    groups.append((count, prev))
    groups.sort(reverse=True)
    top_count = groups[0][0]

    is_straight = False
    straight_high = 0
    if len(groups) == 5:  # all five ranks distinct
        if r0 - r4 == 4:
            is_straight = True
            straight_high = r0
        elif ranks == [14, 5, 4, 3, 2]:  # wheel: A-2-3-4-5
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        category, tiebreaks = 8, (straight_high, 0, 0, 0, 0)
    elif top_count == 4:
        category, tiebreaks = 7, (groups[0][1], groups[1][1], 0, 0, 0)
    elif top_count == 3 and groups[1][0] == 2:
        category, tiebreaks = 6, (groups[0][1], groups[1][1], 0, 0, 0)
    elif is_flush:
        category, tiebreaks = 5, (r0, r1, r2, r3, r4)
    elif is_straight:
        category, tiebreaks = 4, (straight_high, 0, 0, 0, 0)
    elif top_count == 3:
        category, tiebreaks = 3, (groups[0][1], groups[1][1], groups[2][1], 0, 0)
    elif top_count == 2 and groups[1][0] == 2:
        category, tiebreaks = 2, (groups[0][1], groups[1][1], groups[2][1], 0, 0)
    elif top_count == 2:
        category, tiebreaks = 1, (groups[0][1], groups[1][1], groups[2][1], groups[3][1], 0)
    else:
        category, tiebreaks = 0, (r0, r1, r2, r3, r4)

    value = category
    for t in tiebreaks:
        value = value * 15 + t
    return value


# Category index -> human readable name, for display of a packed value.
CATEGORY_NAMES = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "One Pair",
    0: "High Card",
}


def category_of(packed: int) -> int:
    """Return the hand-category index (0-8) of a packed value from :func:`eval5`."""
    return packed // (15 ** 5)


def category_name(packed: int) -> str:
    """Return the human-readable category name of a packed value."""
    return CATEGORY_NAMES[category_of(packed)]


def best_holdem(cards: Sequence[Card]) -> int:
    """Best five-card value from a pool of cards (Texas Hold'em rules).

    Any five of the supplied cards may be used. Callers pass the hole card(s)
    plus the five community cards, so a one-card hand is "best 5 of 6" and a
    two-card hand is "best 5 of 7".
    """
    if len(cards) == 5:
        return eval5(cards)
    return max(eval5(combo) for combo in combinations(cards, 5))


def best_omaha(hole: Sequence[Card], board: Sequence[Card]) -> int:
    """Best five-card value under Omaha rules.

    Exactly two of the four hole cards must be combined with exactly three of
    the five community cards.
    """
    best = -1
    board_triples = list(combinations(board, 3))
    for two in combinations(hole, 2):
        for three in board_triples:
            value = eval5(two + three)
            if value > best:
                best = value
    return best


def best_omaha_pair(two_hole: Tuple[Card, Card], board_triples: List[Tuple[Card, ...]]) -> int:
    """Best Omaha value using a fixed pair of hole cards over pre-built triples.

    Used by the simulator to cache per-pair results so that four-card hands can
    be scored as the max over their six constituent hole-card pairs.
    """
    best = -1
    for three in board_triples:
        value = eval5(two_hole + three)
        if value > best:
            best = value
    return best
