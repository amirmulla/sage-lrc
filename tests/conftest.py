"""Shared pytest fixtures and deterministic RNG setup for optimal_lrc tests."""

import random
import sys
from pathlib import Path

import matplotlib
import pytest
from sage.misc.randstate import set_random_seed

import optimal_lrc  # noqa: F401 - initializes SageMath runtime before test imports

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def seed_rngs() -> None:
    """Seed both Python and SageMath RNGs before every test for determinism."""
    random.seed(42)
    set_random_seed(42)
