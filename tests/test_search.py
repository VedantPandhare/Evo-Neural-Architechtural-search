import hashlib

import torch

from evaluation.metrics import EvalResult
from evaluation.run_logger import RunLogger
from evolution.inheritance import build_offspring
from evolution.population import EvolutionConfig
from evolution.search import init_state, run_search, step
from genomes.operators import genome_hash
from genomes.schema import validate_genome

CFG = EvolutionConfig(population_size=6, generations=4, seed=0)


def fake_evaluator(calls=None):
    """Real model build + real inheritance, but no training; fitness is a deterministic hash."""
    def evaluate(genome, parents):
        model, report = build_offspring(genome, parents)
        h = int(hashlib.sha256(genome_hash(genome).encode()).hexdigest()[:6], 16) / 0xFFFFFF
        if calls is not None:
            calls.append((genome_hash(genome), len(parents)))
        return EvalResult(accuracy=h, n_params=sum(p.numel() for p in model.parameters()),
                          latency=0.001, fitness=h), model, report
    return evaluate


def test_generation_zero_has_no_inheritance_then_later_generations_do(tmp_path):
    logger = RunLogger(str(tmp_path))
    run_search(CFG, fake_evaluator(), logger, inheritance=True)
    recs = logger.load_candidates()
    gen0 = [r for r in recs if r["generation"] == 0]
    later = [r for r in recs if r["generation"] > 0]
    assert len(gen0) == CFG.population_size and all(r["inherit_fraction"] == 0.0 for r in gen0)
    assert later and any(r["inherit_fraction"] > 0.0 for r in later)
    assert all(r["parents"] for r in later)


def test_inheritance_off_never_inherits(tmp_path):
    logger = RunLogger(str(tmp_path))
    run_search(CFG, fake_evaluator(), logger, inheritance=False)
    assert all(r["inherit_fraction"] == 0.0 for r in logger.load_candidates())


def test_no_genome_is_evaluated_twice_and_best_never_regresses(tmp_path):
    calls = []
    logger = RunLogger(str(tmp_path))
    state = run_search(CFG, fake_evaluator(calls), logger)
    hashes = [h for h, _ in calls]
    assert len(hashes) == len(set(hashes))
    rows = logger.load_generations()
    assert [r["generation"] for r in rows] == list(range(CFG.generations))
    bests = [r["best"] for r in rows]
    assert bests == sorted(bests)
    assert state.done and validate_genome(state.best_genome).valid
    assert state.best_fitness == max(r["fitness"] for r in logger.load_candidates())


def test_inheritance_arm_passes_parent_models_scratch_arm_passes_none():
    on, off = [], []
    run_search(CFG, fake_evaluator(on), inheritance=True)
    run_search(CFG, fake_evaluator(off), inheritance=False)
    assert any(n > 0 for _, n in on)
    assert all(n == 0 for _, n in off)


def test_same_seed_gives_identical_searches():
    a = run_search(CFG, fake_evaluator())
    b = run_search(CFG, fake_evaluator())
    assert a.best_genome == b.best_genome and a.best_fitness == b.best_fitness


def test_inherited_weights_actually_come_from_stored_parent():
    state = init_state(CFG)
    ev = fake_evaluator()
    step(state, CFG, ev)  # generation 0 evaluated; state now holds the parents' weights
    parent_weights = {h: sd for h, sd in state.weights.items()}
    breeding = {genome_hash(g): lin for g, lin in zip(state.population, state.lineage)
                if genome_hash(g) not in state.cache}
    assert breeding, "expected at least one non-elite child"
    seen = {}

    def spy(genome, parents):
        h = genome_hash(genome)
        seen[h] = [next(p.parameters()).detach().clone() for p in parents]
        return ev(genome, parents)

    step(state, CFG, spy)
    checked = 0
    for h, (pa, _) in breeding.items():
        if pa in parent_weights and seen.get(h):
            first_key = next(iter(parent_weights[pa]))  # first parameter of the primary parent
            assert torch.equal(seen[h][0], parent_weights[pa][first_key])
            checked += 1
    assert checked > 0
