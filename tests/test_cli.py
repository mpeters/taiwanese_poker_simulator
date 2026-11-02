import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
