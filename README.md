# Taiwanese Poker Simulator (`tps`)

A command-line Monte-Carlo simulator that takes a 7-card Taiwanese Poker hand
and finds the best way to split it into three hands to maximize the average
number of points scored against a field of opponents.

## The game

Each player is dealt **7 cards** and splits them into three hands:

| Hand       | Cards | Ranking rules   | Point weight |
|------------|-------|-----------------|--------------|
| **Top**    | 1     | Texas Hold'em   | ±1           |
| **Middle** | 2     | Texas Hold'em   | ±2           |
| **Bottom** | 4     | Omaha           | ±3           |

Two separate **5-card community boards** are dealt. Every one of your three
hands is scored against **each board**:

* The **top** hand is the best 5-card hand from your 1 card + the 5 board cards.
* The **middle** hand is the best 5-card hand from your 2 cards + the 5 board
  cards.
* The **bottom** hand follows Omaha rules: exactly **2** of your 4 cards plus
  exactly **3** of the 5 board cards.

Against each opponent, on each board, you earn `+weight` for winning a hand,
`-weight` for losing it, and `0` for a tie. Your total for a simulated hand is
the sum over **both boards** and **all opponents**.

**Home run bonus.** If you beat an opponent on all three hands on *both* boards
(all 6 matchups — 1 + 2 + 3 twice = 12 base points), you earn an extra **+8**
"home run" bonus against that opponent, for **20** points. This is scored
symmetrically: if an opponent sweeps all 6 matchups against you, that costs you
**−8**. The bonus is per opponent, so you can home-run several opponents in one
hand.

## Usage

```bash
./tps --hand "AcKdJd7h7s2c2d" --opponents 3 --sims 500
```

Cards are two characters — value then suit — and are case-insensitive:

* Values: `2 3 4 5 6 7 8 9 T J Q K A`
* Suits: `c` (clubs) `d` (diamonds) `h` (hearts) `s` (spades)

e.g. `Kh` = King of Hearts, `Ac` = Ace of Clubs, `Td` = Ten of Diamonds,
`4s` = Four of Spades.

### Options

| Option            | Default | Description                                      |
|-------------------|---------|--------------------------------------------------|
| `--hand`          | *req.*  | Your 7 cards, e.g. `"AcKdJd7h7s2c2d"`.           |
| `--opponents`     | `1`     | Number of opponents (1–5; limited by the deck).  |
| `--sims`          | `1000`  | Number of hands to simulate.                     |
| `--seed`          | –       | RNG seed for reproducible runs.                  |
| `--workers`       | `4`     | Worker processes for parallel runs. Runs with more than 1000 simulations are split across this many processes; use `1` to force single-threaded. |
| `--top`           | `5`     | How many top splits to list.                     |
| `--opponent-samples` | `20`  | Boards each blind opponent samples to set its hand (higher = tougher, more accurate field, but slower). |
| `--absolute`      | off     | Show each split's absolute score in the table instead of its gap from the best split. |
| `--details`       | off     | Add columns to the splits table breaking down each split's average points by position (top, middle, bottom, home run). |
| `--quiet`         | off     | Suppress the progress indicator.                 |

### Example output

```
Best split  (average score: +2.074 points per hand, per opponent)
============================================================
  Top     (x1, Hold'em): K♦
  Middle  (x2, Hold'em): 7♥ 7♠
  Bottom  (x3, Omaha  ): A♣ J♦ 2♣ 2♦

  Average points by position:
    Top      : +0.758
    Middle   : +0.621
    Bottom   : +0.396
    Home run : +0.299
```

Scores are reported **per hand and per opponent**, so they are comparable
across field sizes: +6 against 3 opponents is shown the same as +2 against 1.

By default the splits table shows each split's **gap from the best** (the best
split is `+0.000`, the rest negative), making the cost of each alternative line
easy to read. Pass `--absolute` to show the raw simulated score for each split
instead. The headline always reports the best split's absolute average EV.

## How it works

* Your 7 cards can be split into exactly **105** distinct (1, 2, 4) partitions.
  Your split is chosen **once** — as if set before the boards are revealed — and
  the simulator reports the partition with the best average score.
* Each simulation deals two random boards and the opponents' hands from the
  remaining deck, then scores all 105 of your candidate splits against the
  field. Averaging over many deals ranks the splits.
* **Opponent model (blind / symmetric):** each opponent commits to a split
  *before* the boards are revealed — the same information you have. It reasons
  the same way the optimizer does (maximize weighted made-hand strength) but,
  since it cannot see the real boards, it estimates each split's expected
  strength by averaging the evaluator over a handful of random boards
  (`--opponent-samples`, default 20). It then commits to that split, which is
  scored on the actual boards. No hand-tuned preflop tables are used — the same
  evaluator drives everything. (The opponent strategy is a pluggable function in
  `tps_sim/simulator.py`; `make_blind_strategy` is the default and the
  board-aware `greedy_best_strategy` is kept for comparison.)

## Development

Pure standard library — no runtime dependencies.

```bash
# run the tests (requires pytest)
python -m pytest
```

Project layout:

```
tps                     # executable CLI launcher
tps_sim/
  cards.py              # card parsing & representation
  evaluator.py          # 5-card hand evaluation, Hold'em / Omaha helpers
  simulator.py          # split enumeration, Monte-Carlo simulation, scoring
  cli.py                # argument parsing, progress bar, output
tests/                  # pytest suite
```
