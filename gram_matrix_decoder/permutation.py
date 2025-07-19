"""Permutation generation and evaluation functions."""

import itertools
import math
import numpy as np
from config import ALL_PERMS, N_RANDOM_PERMS, RANDOM_SEED


rng = np.random.default_rng(RANDOM_SEED)


def random_permutation(c: int) -> np.ndarray:
    """Return a random permutation array of 0…c-1."""
    return rng.permutation(c)


def permutation_iterator(C: int):
    """Yield permutations according to ALL_PERMS flag."""
    if ALL_PERMS:
        return itertools.permutations(range(C))
    # else
    return (random_permutation(C) for _ in range(N_RANDOM_PERMS))


def get_permutation_count(C: int) -> int:
    """Get total number of permutations to evaluate."""
    if ALL_PERMS:
        return math.factorial(C)
    return N_RANDOM_PERMS