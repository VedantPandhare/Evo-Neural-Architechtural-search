"""Full search loop: init -> (inherit) -> train -> evaluate -> log -> select -> mutate/crossover.

One call to `step` evaluates one generation and prepares the next, so `SearchState` is always
a consistent checkpoint boundary.

Weights: `state.weights` / `state.genomes` hold, keyed by genome hash, the CPU state_dicts and
genomes of the population the current children were bred from (their parents). Fitness is cached
by genome hash, so an elite or duplicate child is never retrained. A cached genome that is not in
the previous population has no stored weights; if it later becomes a parent, the child simply
falls back to the other parent or to fresh initialisation (counted in the inheritance fraction).
"""

import random
import time
from dataclasses import dataclass, field

from evolution.inheritance import build_offspring
from evolution.population import EvolutionConfig, init_population, next_generation
from genomes.operators import genome_hash


@dataclass
class SearchState:
    generation: int
    population: list
    lineage: list  # per population slot: (primary_parent_hash | None, secondary_parent_hash | None)
    rng: random.Random
    cache: dict = field(default_factory=dict)  # genome_hash -> EvalResult
    weights: dict = field(default_factory=dict)  # parent hash -> cpu state_dict
    genomes: dict = field(default_factory=dict)  # parent hash -> genome
    best_fitness: float = float("-inf")
    best_genome: dict | None = None
    done: bool = False


def init_state(cfg: EvolutionConfig) -> SearchState:
    rng = random.Random(cfg.seed)
    population = init_population(cfg.population_size, rng)
    return SearchState(generation=0, population=population, lineage=[(None, None)] * len(population), rng=rng)


def _parent_models(parent_hashes: tuple, state: SearchState) -> list:
    """Rebuild parent models, in priority order, from stored weights; skip unknown parents."""
    from models.builder import build_model

    models = []
    for h in parent_hashes:
        if h is not None and h in state.weights:
            m = build_model(state.genomes[h])
            m.load_state_dict(state.weights[h])
            models.append(m)
    return models


def make_evaluator(train_cfg, fitness_cfg):
    """Real evaluator: evaluate(genome, parent_models) -> (EvalResult, model, InheritanceReport)."""
    from evaluation.metrics import evaluate_model

    def evaluate(genome, parent_models):
        model, report = build_offspring(genome, parent_models)
        return evaluate_model(model, genome, train_cfg, fitness_cfg), model, report

    return evaluate


def step(state: SearchState, cfg: EvolutionConfig, evaluate, logger=None, inheritance: bool = True) -> dict:
    """Evaluate state.population (generation state.generation), then breed the next one.
    With inheritance=False every child trains from the shared Kaiming init (the baseline arm)."""
    gen = state.generation
    new_weights, new_genomes, fitnesses, fresh_records = {}, {}, [], []
    n_new = 0

    for genome, (pa, pb) in zip(state.population, state.lineage):
        h = genome_hash(genome)
        if h in state.cache:
            fitnesses.append(state.cache[h].fitness)
            if h in state.weights:  # elite carried over: keep its weights for the next breeding step
                new_weights[h], new_genomes[h] = state.weights[h], state.genomes[h]
            continue

        parents = _parent_models((pa, pb), state) if inheritance else []
        start = time.perf_counter()
        result, model, report = evaluate(genome, parents)
        seconds = time.perf_counter() - start

        state.cache[h] = result
        fitnesses.append(result.fitness)
        new_weights[h] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        new_genomes[h] = genome
        n_new += 1
        fresh_records.append({"inherit_fraction": report.fraction if report else 0.0})
        if logger:
            logger.log_candidate(gen, h, genome, result, seconds, inheritance=report,
                                 parents=(pa, pb) if (pa or pb) else None)

    i_best = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    if fitnesses[i_best] > state.best_fitness:
        state.best_fitness, state.best_genome = fitnesses[i_best], state.population[i_best]

    row = logger.log_generation(gen, state.population, fitnesses, n_new, fresh_records) if logger else {
        "generation": gen, "best": max(fitnesses), "min": min(fitnesses), "new_evaluations": n_new,
        "mean": sum(fitnesses) / len(fitnesses)}

    if gen >= cfg.generations - 1:
        state.done = True
    else:
        lineage = []
        state.population = next_generation(state.population, fitnesses, cfg, state.rng, lineage=lineage)
        state.lineage = lineage
        state.generation += 1
    state.weights, state.genomes = new_weights, new_genomes
    return row


def run_search(cfg: EvolutionConfig, evaluate, logger=None, inheritance: bool = True,
               state: SearchState | None = None, on_step=None) -> SearchState:
    """Run (or continue) a search until cfg.generations are done. `on_step(state, row)` is called
    after every generation, which is where checkpoints are written."""
    state = state or init_state(cfg)
    while not state.done:
        row = step(state, cfg, evaluate, logger, inheritance)
        if on_step:
            on_step(state, row)
    if logger:
        logger.log_summary({"best_fitness": state.best_fitness, "best_genome": state.best_genome,
                            "inheritance": inheritance, "generations": cfg.generations})
    return state
