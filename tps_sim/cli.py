"""Command-line interface for the Taiwanese Poker simulator (``tps``)."""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

from .cards import CardError, format_cards, parse_hand
from .simulator import (
    BLIND_SAMPLE_BOARDS,
    DEFAULT_WORKERS,
    PARALLEL_THRESHOLD,
    SimulationReport,
    describe_split_hands,
    make_blind_strategy,
    simulate,
)

MAX_OPPONENTS = (52 - 7 - 10) // 7  # 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tps",
        description=(
            "Taiwanese Poker simulator: find the best way to split a 7-card "
            "hand into a 1-card (Hold'em), 2-card (Hold'em) and 4-card (Omaha) "
            "hand, maximizing average points across simulated hands."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hand",
        required=True,
        help='Your 7 cards, e.g. "AcKdJd7h7s2c2d" (case-insensitive).',
    )
    parser.add_argument(
        "--opponents",
        type=int,
        default=1,
        help=f"Number of opponents (1-{MAX_OPPONENTS}).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=1000,
        help="Number of hands to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=(
            f"Worker processes for parallel runs. Runs with more than "
            f"{PARALLEL_THRESHOLD} simulations are split across this many "
            f"processes; use 1 to force single-threaded execution."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many of the best splits to list in the results table.",
    )
    parser.add_argument(
        "--opponent-samples",
        type=int,
        default=BLIND_SAMPLE_BOARDS,
        metavar="N",
        help=(
            "Boards each (blind) opponent samples to set its hand. Higher means "
            "stronger, more accurate opponents but slower runs."
        ),
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help=(
            "In the splits table, show each split's absolute average score "
            "instead of its gap from the best split."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the progress indicator.",
    )
    return parser


def _format_duration(seconds: float) -> str:
    """Render a short, human-friendly duration, e.g. ``9s`` or ``2m03s``."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{int(round(seconds))}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


class _ProgressBar:
    """Minimal stderr progress bar with an estimated time remaining.

    Updates are throttled so writes stay infrequent whether it is called once
    per simulation (single-threaded) or once per completed chunk (parallel,
    where ``done`` arrives in larger jumps).
    """

    def __init__(self, total: int, width: int = 30):
        self.total = total
        self.width = width
        self.start = time.monotonic()
        self.step = max(1, total // 100)
        self.last_done = 0

    def __call__(self, done: int, total: int) -> None:
        # Print on completion, or once progress has advanced by at least a step
        # since the last line (robust to chunk-sized jumps in parallel mode).
        if done != total and done - self.last_done < self.step:
            return
        self.last_done = done

        frac = done / total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self.start

        if done >= total:
            tail = f"{_format_duration(elapsed)} elapsed"
        elif frac > 0:
            remaining = elapsed * (1 - frac) / frac
            tail = (
                f"{_format_duration(elapsed)} elapsed, "
                f"~{_format_duration(remaining)} left"
            )
        else:
            tail = f"{_format_duration(elapsed)} elapsed, ~?s left"

        # Pad to clear any longer previous line (the ETA text shrinks over time).
        line = f"  [{bar}] {done}/{total} ({frac * 100:5.1f}%)  {tail}"
        sys.stderr.write(f"\r{line:<70}")
        sys.stderr.flush()
        if done >= total:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _format_hand_line(label: str, weight: int, cards, kind: str) -> str:
    cards_str = format_cards(cards, symbols=True)
    return f"  {label:<7} (x{weight}, {kind:<7}): {cards_str}"


def _print_report(report: SimulationReport, top_n: int, absolute: bool) -> None:
    best = report.best
    hands = describe_split_hands(report.player_cards, best.split)

    print()
    print("=" * 60)
    print(
        f"Best split  (average score: {best.avg_score:+.3f} "
        f"points per hand, per opponent)"
    )
    print("=" * 60)
    print(_format_hand_line("Top", 1, hands["top"], "Hold'em"))
    print(_format_hand_line("Middle", 2, hands["middle"], "Hold'em"))
    print(_format_hand_line("Bottom", 3, hands["bottom"], "Omaha"))
    print()
    print("  Average points by position:")
    print(f"    Top      : {best.avg_top:+.3f}")
    print(f"    Middle   : {best.avg_middle:+.3f}")
    print(f"    Bottom   : {best.avg_bottom:+.3f}")
    print(f"    Home run : {best.avg_bonus:+.3f}")

    if top_n > 1:
        print()
        best_score = best.avg_score
        if absolute:
            heading = f"Top {min(top_n, len(report.ranked))} splits by average score:"
            col = "score"
        else:
            heading = (
                f"Top {min(top_n, len(report.ranked))} splits "
                f"(gap from best; best = 0):"
            )
            col = "gap"
        print(heading)
        print(f"  {'#':>2}  {col:>8}   {'top':<4} {'middle':<7} {'bottom':<14}")
        print("  " + "-" * 44)
        for rank, result in enumerate(report.ranked[:top_n], start=1):
            h = describe_split_hands(report.player_cards, result.split)
            value = result.avg_score if absolute else result.avg_score - best_score
            print(
                f"  {rank:>2}  {value:>+8.3f}   "
                f"{format_cards(h['top']):<4} "
                f"{format_cards(h['middle']):<7} "
                f"{format_cards(h['bottom']):<14}"
            )
    print()


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return _run(argv)
    except (KeyboardInterrupt, EOFError):
        # Make sure we land on a fresh line past any progress bar.
        sys.stderr.write("\n\nInterrupted — exiting. No results to show.\n")
        sys.stderr.flush()
        return 130  # 128 + SIGINT, the conventional "interrupted" exit code


def _run(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        player_cards = parse_hand(args.hand, expected=7)
    except CardError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover (argparse exits)

    if args.opponents < 1:
        parser.error("--opponents must be at least 1.")
    if args.opponents > MAX_OPPONENTS:
        parser.error(
            f"--opponents must be at most {MAX_OPPONENTS} (deck has only 52 cards)."
        )
    if args.sims < 1:
        parser.error("--sims must be at least 1.")
    if args.opponent_samples < 1:
        parser.error("--opponent-samples must be at least 1.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")

    print("Taiwanese Poker Simulator")
    print(f"  Your hand : {format_cards(player_cards, symbols=True)}")
    print(f"  Opponents : {args.opponents}")
    print(f"  Simulations: {args.sims}")
    if args.seed is not None:
        print(f"  Seed      : {args.seed}")
    parallel = args.sims > PARALLEL_THRESHOLD and args.workers > 1
    if parallel:
        print(f"  Workers   : {args.workers}")
    print()
    mode = f" across {args.workers} workers" if parallel else ""
    print(f"Running {args.sims} simulations against {args.opponents} "
          f"opponent(s){mode}...")

    progress = None if args.quiet else _ProgressBar(args.sims)

    report = simulate(
        player_cards,
        opponents=args.opponents,
        simulations=args.sims,
        seed=args.seed,
        opponent_strategy=make_blind_strategy(args.opponent_samples),
        progress=progress,
        workers=args.workers,
    )

    _print_report(report, args.top, args.absolute)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
