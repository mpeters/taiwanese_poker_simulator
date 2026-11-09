import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import tps_sim.cli as cli
from tps_sim.cli import main

BASE = ["--hand", "AcKdJd7h7s2c2d", "--sims", "40", "--seed", "1", "--quiet"]


def test_default_shows_gap_from_best(capsys):
    rc = main(BASE)
    out = capsys.readouterr().out
    assert rc == 0
    assert "gap from best" in out
    # The top row of the gap table is exactly +0.000.
    assert "+0.000" in out


def test_absolute_flag_shows_scores(capsys):
    rc = main(BASE + ["--absolute"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "by average score" in out
    assert "gap from best" not in out


def test_headline_always_absolute(capsys):
    # The headline reports the best split's absolute EV regardless of mode.
    main(BASE)
    gap_out = capsys.readouterr().out
    main(BASE + ["--absolute"])
    abs_out = capsys.readouterr().out
    for out in (gap_out, abs_out):
        assert "Best split  (average score:" in out
        assert "per hand, per opponent" in out


def test_report_header_lists_inputs(capsys):
    main(BASE)
    out = capsys.readouterr().out
    assert "Taiwanese Poker Simulator" in out
    assert "Opponents : 1" in out
    assert "Seed      : 1" in out  # seed line printed only when a seed is given


def test_seed_line_absent_without_seed(capsys):
    main(["--hand", "AcKdJd7h7s2c2d", "--sims", "10", "--quiet"])
    out = capsys.readouterr().out
    assert "Seed      :" not in out


def test_top_details_adds_position_columns(capsys):
    # Without the flag, the per-position header columns are absent.
    main(BASE)
    plain = capsys.readouterr().out
    assert "pts:top" not in plain

    main(BASE + ["--top-details"])
    out = capsys.readouterr().out
    for header in ("pts:top", "pts:mid", "pts:bot", "pts:HR"):
        assert header in out


def test_top_details_works_with_absolute(capsys):
    main(BASE + ["--top-details", "--absolute"])
    out = capsys.readouterr().out
    assert "by average score" in out
    assert "pts:top" in out


def test_top_one_hides_splits_table(capsys):
    # With --top 1 the multi-row splits table should not be printed.
    main(BASE + ["--top", "1"])
    out = capsys.readouterr().out
    assert "gap from best" not in out
    assert "by average score" not in out


# ---------------------------------------------------------------------------
# Argument validation (argparse errors exit with code 2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "argv",
    [
        ["--hand", "not-a-hand", "--quiet"],          # unparseable hand
        ["--hand", "AcKdJd", "--quiet"],              # wrong card count
        ["--hand", "AcKdJd7h7s2c2d", "--opponents", "0", "--quiet"],
        ["--hand", "AcKdJd7h7s2c2d", "--opponents", "6", "--quiet"],
        ["--hand", "AcKdJd7h7s2c2d", "--sims", "0", "--quiet"],
        ["--hand", "AcKdJd7h7s2c2d", "--opponent-samples", "0", "--quiet"],
        ["--hand", "AcKdJd7h7s2c2d", "--workers", "0", "--quiet"],
    ],
)
def test_invalid_arguments_exit_with_code_2(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_missing_required_hand_exits():
    with pytest.raises(SystemExit):
        main(["--sims", "10"])


# ---------------------------------------------------------------------------
# Workers line in parallel mode
# ---------------------------------------------------------------------------

def test_workers_line_printed_in_parallel_mode(capsys):
    # Any --workers > 1 engages parallel mode, so the "Workers" line prints even
    # with a small simulation count.
    main(["--hand", "AcKdJd7h7s2c2d", "--sims", "10", "--workers", "2",
          "--seed", "1", "--quiet"])
    out = capsys.readouterr().out
    assert "Workers   : 2" in out
    assert "across 2 workers" in out


# ---------------------------------------------------------------------------
# Interrupt handling
# ---------------------------------------------------------------------------

def test_keyboard_interrupt_returns_130(monkeypatch, capsys):
    def boom(argv=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run", boom)
    rc = main([])
    assert rc == 130
    err = capsys.readouterr().err
    assert "Interrupted" in err


def test_eof_error_returns_130(monkeypatch):
    def boom(argv=None):
        raise EOFError

    monkeypatch.setattr(cli, "_run", boom)
    assert main([]) == 130


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [
        (-5.0, "0s"),
        (0.0, "0s"),
        (9.4, "9s"),
        (59.0, "59s"),
        (60.0, "1m00s"),
        (123.0, "2m03s"),
        (3600.0, "1h00m00s"),
        (3661.0, "1h01m01s"),
    ],
)
def test_format_duration(seconds, expected):
    assert cli._format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def test_progress_bar_writes_and_completes(capsys):
    bar = cli._ProgressBar(total=100)
    bar(50, 100)   # mid-run update: shows elapsed + ETA
    bar(100, 100)  # completion: shows final line and newline
    err = capsys.readouterr().err
    assert "50/100" in err
    assert "100/100" in err
    assert "100.0%" in err
    assert err.endswith("\n")


def test_progress_bar_throttles_small_steps(capsys):
    bar = cli._ProgressBar(total=1000)  # step = 10
    bar(1, 1000)  # below the step threshold and not complete -> suppressed
    err = capsys.readouterr().err
    assert err == ""
