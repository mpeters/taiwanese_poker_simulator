# Makefile for the Taiwanese Poker Simulator.
#
#   make test   - create a local virtualenv, install dependencies and run the
#                 test suite with a coverage report.
#   make clean  - remove the virtualenv and all build/test artifacts.

VENV := .venv
PYTHON := python3
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Sentinel file: marks that dependencies are installed in the venv. Rebuilt
# whenever pyproject.toml changes so dependency updates are picked up.
STAMP := $(VENV)/.installed

.PHONY: test clean

# Install dependencies into a local virtualenv.
$(STAMP): pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]" pytest-cov
	touch $(STAMP)

# Install (if needed) and run the full test suite with coverage.
test: $(STAMP)
	$(PY) -m pytest --cov=tps_sim --cov-report=term-missing

# Remove the virtualenv and every build/test artifact.
clean:
	rm -rf $(VENV)
	rm -rf .pytest_cache .coverage .coverage.* htmlcov coverage.xml
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
