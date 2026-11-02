#!/usr/bin/env python3
"""Entry point for the Taiwanese Poker simulator CLI.

Run directly from the project directory:

    ./tps --hand "AcKdJd7h7s2c2d" --opponents 3 --simulations 500
"""

import os
import sys

# Make the bundled package importable no matter where tps is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tps_sim.cli import main

if __name__ == "__main__":
    sys.exit(main())
