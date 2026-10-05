"""Population initialisation and the generation loop."""

import random
from dataclasses import dataclass

from evolution.crossover import crossover
from evolution.mutation import mutate
from evolution.selection import elite_indices, tournament_select
from genomes.operators import genome_hash
from genomes.schema import random_genome, validate_genome


@dataclass
class EvolutionConfig:
    population_size: int = 20
    generations: int = 15
    tournament_size: int = 3
    elite_fraction: float = 0.1
    crossover_rate: float = 0.5
    mutation_rate: float = 0.8  # chance a crossover child is also mutated; non-crossover children always are
    seed: int = 0


def init_population(n: int, rng: random.Random, max_tries: int = 1000) -> list:
    """N distinct valid random genomes."""
    pop, seen = [], set()
    for _ in range(max_tries):
        if len(pop) == n:
            break
        g = random_genome(rng)
        h = genome_hash(g)
        if h not in seen and validate_genome(g).valid:
            seen.add(h)
            pop.append(g)
    if len(pop) < n:
        raise RuntimeError(f"could only generate {len(pop)} distinct valid genomes")
    return pop


def next_generation(population: list, fitnesses: list, cfg: EvolutionConfig, rng: random.Random) -> list:
    """Elites carried over unchanged; the rest are tournament-selected parents -> crossover -> mutation."""
    new_pop = [population[i] for i in elite_indices(fitnesses, cfg.elite_fraction)]
    while len(new_pop) < len(population):
        pa = population[tournament_select(fitnesses, cfg.tournament_size, rng)]
        crossed = rng.random() < cfg.crossover_rate
        if crossed:
            pb = population[tournament_select(fitnesses, cfg.tournament_size, rng)]
            child = crossover(pa, pb, rng)
        else:
            child = pa
        if not crossed or rng.random() < cfg.mutation_rate:
            child, _ = mutate(child, rng)
        new_pop.append(child)
    return new_pop


def run_evolution(eval_fn, cfg: EvolutionConfig, on_generation=None) -> dict:
    """eval_fn(genome) -> fitness (float). Fitness is cached by genome hash so elites and
    duplicate offspring are never re-trained. Returns history and the best genome."""
    rng = random.Random(cfg.seed)
    cache: dict = {}
    population = init_population(cfg.population_size, rng)
    history, best = [], (float("-inf"), None)

    for gen in range(cfg.generations):
        n_evals = 0
        for g in population:
            h = genome_hash(g)
            if h not in cache:
                cache[h] = eval_fn(g)
                n_evals += 1
        fitnesses = [cache[genome_hash(g)] for g in population]
        i_best = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[i_best] > best[0]:
            best = (fitnesses[i_best], population[i_best])
        stats = {"generation": gen, "best": max(fitnesses), "mean": sum(fitnesses) / len(fitnesses),
                 "min": min(fitnesses), "new_evaluations": n_evals}
        history.append(stats)
        if on_generation:
            on_generation(stats, population, fitnesses)
        if gen < cfg.generations - 1:
            population = next_generation(population, fitnesses, cfg, rng)

    return {"history": history, "best_fitness": best[0], "best_genome": best[1]}
