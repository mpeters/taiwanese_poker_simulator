"""Card parsing and representation for the Taiwanese Poker simulator.

A card is represented as a simple ``(rank, suit)`` tuple:

* ``rank`` is an ``int`` from 2 (two) to 14 (ace high).
* ``suit`` is a lowercase character: ``c`` (clubs), ``d`` (diamonds),
  ``h`` (hearts) or ``s`` (spades).

Tuples are used instead of a class for speed: the simulator evaluates many
millions of five-card hands and tuple access is considerably faster than
attribute access.
"""

from __future__ import annotations

from typing import List, Tuple

Card = Tuple[int, str]

# Rank character -> integer value (ace high).
RANK_TO_VALUE = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "t": 10, "j": 11, "q": 12, "k": 13, "a": 14,
}
VALUE_TO_RANK = {v: k.upper() for k, v in RANK_TO_VALUE.items()}

SUITS = ("c", "d", "h", "s")
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

# The full 52-card deck, built once.
FULL_DECK: List[Card] = [
    (value, suit) for value in range(2, 15) for suit in SUITS
]


class CardError(ValueError):
    """Raised when a card or hand string cannot be parsed."""


def parse_card(token: str) -> Card:
    """Parse a two-character card string such as ``"Kh"`` into a ``Card``.

    Input is case-insensitive, so ``"kh"``, ``"Kh"`` and ``"KH"`` are the same
    card. Raises :class:`CardError` for anything that is not a valid card.
    """
    if len(token) != 2:
        raise CardError(f"Card {token!r} must be exactly 2 characters (value + suit).")
    rank_char, suit_char = token[0].lower(), token[1].lower()
    if rank_char not in RANK_TO_VALUE:
        raise CardError(
            f"Invalid card value {token[0]!r} in {token!r}; "
            f"expected one of 2-9, T, J, Q, K, A."
        )
    if suit_char not in SUIT_SYMBOLS:
        raise CardError(
            f"Invalid suit {token[1]!r} in {token!r}; expected one of c, d, h, s."
        )
    return (RANK_TO_VALUE[rank_char], suit_char)


def parse_hand(text: str, expected: int = 7) -> List[Card]:
    """Parse a run-together card string such as ``"AcKdJd7h7s2c2d"``.

    Whitespace and commas between cards are tolerated. The result is validated
    to contain exactly ``expected`` cards with no duplicates.
    """
    cleaned = text.replace(",", " ").split()
    # Allow either space-separated cards or one long run-together string.
    tokens: List[str] = []
    for chunk in cleaned:
        if len(chunk) % 2 != 0:
            raise CardError(
                f"Card group {chunk!r} has an odd number of characters; "
                f"each card needs 2 characters."
            )
        tokens.extend(chunk[i : i + 2] for i in range(0, len(chunk), 2))

    cards = [parse_card(tok) for tok in tokens]

    if len(cards) != expected:
        raise CardError(
            f"Expected {expected} cards but got {len(cards)}: "
            f"{format_cards(cards)}"
        )

    seen = set()
    for card in cards:
        if card in seen:
            raise CardError(f"Duplicate card {format_card(card)} in hand.")
        seen.add(card)

    return cards


def format_card(card: Card, symbols: bool = False) -> str:
    """Render a card as a short string, e.g. ``(13, 'h')`` -> ``"Kh"``."""
    rank, suit = card
    suit_str = SUIT_SYMBOLS[suit] if symbols else suit
    return f"{VALUE_TO_RANK[rank]}{suit_str}"


def format_cards(cards, symbols: bool = False, sep: str = " ") -> str:
    """Render an iterable of cards as a separated string."""
    return sep.join(format_card(c, symbols) for c in cards)
