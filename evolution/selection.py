"""Tournament selection and elitism. All functions work on indices into a fitness list."""

import random


def tournament_select(fitnesses: list, k: int, rng: random.Random) -> int:
    """Sample k contenders with replacement; return the index of the fittest."""
    contenders = [rng.randrange(len(fitnesses)) for _ in range(k)]
    return max(contenders, key=lambda i: fitnesses[i])


def elite_indices(fitnesses: list, fraction: float) -> list:
    n = max(1, round(len(fitnesses) * fraction))
    return sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:n]
