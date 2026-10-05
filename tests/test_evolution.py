import copy
import random

import pytest

from evolution.crossover import crossover
from evolution.mutation import MUTATION_OPERATORS, mutate
from evolution.population import EvolutionConfig, init_population, next_generation, run_evolution
from evolution.selection import elite_indices, tournament_select
from genomes.operators import genome_hash
from genomes.schema import default_genome_space, validate_genome

SPACE = default_genome_space()


def _genomes(n, seed=0):
    return init_population(n, random.Random(seed))


@pytest.mark.parametrize("name", list(MUTATION_OPERATORS))
def test_each_operator_leaves_input_untouched_and_changes_it_when_applicable(name):
    rng = random.Random(1)
    applied = 0
    for g in _genomes(30):
        before = copy.deepcopy(g)
        child = MUTATION_OPERATORS[name](g, rng, SPACE)
        assert g == before
        if child is not None:
            applied += 1
            assert child != g
    assert applied > 0


def test_mutate_always_returns_valid_genome():
    rng = random.Random(2)
    for g in _genomes(40):
        child, _ = mutate(g, rng)
        assert validate_genome(child).valid


def test_mutate_falls_back_to_copy_when_nothing_valid():
    g = _genomes(1)[0]
    child, op = mutate(g, random.Random(0), max_attempts=0)
    assert op is None and child == g and child is not g


def test_crossover_children_valid_and_inherit_from_parents():
    rng = random.Random(3)
    pop = _genomes(20)
    for _ in range(60):
        a, b = rng.sample(pop, 2)
        child = crossover(a, b, rng)
        assert validate_genome(child).valid
        assert child["optimizer"] in (a["optimizer"], b["optimizer"])
        assert child["batch_size"] in (a["batch_size"], b["batch_size"])


def test_tournament_favours_fitter_individuals():
    rng = random.Random(0)
    fit = [0.1, 0.2, 0.3, 0.9]
    wins = sum(tournament_select(fit, 3, rng) == 3 for _ in range(2000))
    assert wins / 2000 > 0.5  # expected ~0.58 vs 0.25 for uniform


def test_tournament_size_one_is_uniform():
    rng = random.Random(0)
    fit = [0.1, 0.2, 0.3, 0.9]
    wins = sum(tournament_select(fit, 1, rng) == 3 for _ in range(4000))
    assert 0.18 < wins / 4000 < 0.32


def test_elite_indices_returns_top_fraction_and_at_least_one():
    assert elite_indices([0.1, 0.9, 0.5, 0.7], 0.5) == [1, 3]
    assert elite_indices([0.1, 0.9], 0.01) == [1]


def test_init_population_distinct_and_valid():
    pop = init_population(25, random.Random(5))
    assert len({genome_hash(g) for g in pop}) == 25
    assert all(validate_genome(g).valid for g in pop)


def test_next_generation_keeps_size_and_elites():
    pop = _genomes(10)
    fit = [float(i) for i in range(10)]
    cfg = EvolutionConfig(population_size=10, elite_fraction=0.2)
    new = next_generation(pop, fit, cfg, random.Random(0))
    assert len(new) == 10
    assert new[0] == pop[9] and new[1] == pop[8]
    assert all(validate_genome(g).valid for g in new)


def test_run_evolution_smoke_caches_and_best_never_regresses():
    calls = []

    def toy_fitness(g):
        calls.append(genome_hash(g))
        return -len(g["layers"]) - g["learning_rate"]

    out = run_evolution(toy_fitness, EvolutionConfig(population_size=8, generations=5, seed=0))
    assert len(out["history"]) == 5
    assert len(calls) == len(set(calls))  # nothing evaluated twice
    bests = [h["best"] for h in out["history"]]
    assert bests == sorted(bests)  # elitism => best fitness is monotone
    assert validate_genome(out["best_genome"]).valid


def test_next_generation_lineage_aligned_with_population():
    pop = _genomes(10)
    fit = [float(i) for i in range(10)]
    cfg = EvolutionConfig(population_size=10, elite_fraction=0.2, crossover_rate=1.0)
    lineage = []
    new = next_generation(pop, fit, cfg, random.Random(0), lineage=lineage)
    hashes = {genome_hash(g) for g in pop}
    assert len(lineage) == len(new)
    assert all(a in hashes and (b is None or b in hashes) for a, b in lineage)
    assert lineage[0] == (genome_hash(pop[9]), None)
    assert all(b is not None for _, b in lineage[2:])  # crossover_rate=1.0
