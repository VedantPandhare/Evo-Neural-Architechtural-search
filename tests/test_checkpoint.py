import pytest

from evaluation.run_logger import RunLogger
from evolution.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from evolution.population import EvolutionConfig
from evolution.search import init_state, run_search, step
from tests.test_search import fake_evaluator

CFG = EvolutionConfig(population_size=6, generations=4, seed=0)


class Crash(Exception):
    pass


def _strip(records):
    return [{k: v for k, v in r.items() if k not in ("train_seconds",)} for r in records]


def _run_full(run_dir):
    logger = RunLogger(str(run_dir))
    state = run_search(CFG, fake_evaluator(), logger)
    return state, logger


def test_roundtrip_preserves_state_including_rng(tmp_path):
    state = init_state(CFG)
    step(state, CFG, fake_evaluator())
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(state, path)
    loaded = load_checkpoint(path)
    assert loaded.generation == state.generation and loaded.population == state.population
    assert loaded.lineage == state.lineage and loaded.cache == state.cache
    assert loaded.best_fitness == state.best_fitness and loaded.done == state.done
    assert loaded.rng.random() == state.rng.random()
    assert loaded.weights.keys() == state.weights.keys()


def test_save_is_atomic_no_tmp_left(tmp_path):
    state = init_state(CFG)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(state, path)
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"]


def test_resume_after_clean_interrupt_matches_uninterrupted_run(tmp_path):
    full_state, full_log = _run_full(tmp_path / "full")

    run_dir = tmp_path / "resumed"
    logger = RunLogger(str(run_dir))
    ckpt = checkpoint_path(str(run_dir))

    def on_step(state, row):
        save_checkpoint(state, ckpt)
        if row["generation"] == 1:
            raise Crash

    with pytest.raises(Crash):
        run_search(CFG, fake_evaluator(), logger, on_step=on_step)

    logger2 = RunLogger(str(run_dir))
    state = load_checkpoint(ckpt)
    assert state.generation == 2 and not state.done
    logger2.truncate_from_generation(state.generation)
    final = run_search(CFG, fake_evaluator(), logger2, state=state, on_step=lambda s, r: save_checkpoint(s, ckpt))

    assert final.best_genome == full_state.best_genome and final.best_fitness == full_state.best_fitness
    assert _strip(logger2.load_candidates()) == _strip(full_log.load_candidates())
    assert [r["best"] for r in logger2.load_generations()] == [r["best"] for r in full_log.load_generations()]


def test_resume_after_mid_generation_crash_has_no_duplicate_records(tmp_path):
    full_state, full_log = _run_full(tmp_path / "full")

    run_dir = tmp_path / "crashed"
    logger = RunLogger(str(run_dir))
    ckpt = checkpoint_path(str(run_dir))
    inner = fake_evaluator()
    count = {"n": 0}

    def flaky(genome, parents):
        count["n"] += 1
        if count["n"] == CFG.population_size + 3:  # dies partway through generation 1
            raise Crash
        return inner(genome, parents)

    with pytest.raises(Crash):
        run_search(CFG, flaky, logger, on_step=lambda s, r: save_checkpoint(s, ckpt))

    logger2 = RunLogger(str(run_dir))
    partial = logger2.load_candidates()
    state = load_checkpoint(ckpt)
    logger2.truncate_from_generation(state.generation)
    assert len(logger2.load_candidates()) < len(partial)  # partial records were dropped
    final = run_search(CFG, fake_evaluator(), logger2, state=state)

    hashes = [r["genome_hash"] for r in logger2.load_candidates()]
    assert len(hashes) == len(set(hashes))
    assert final.best_fitness == full_state.best_fitness
    assert _strip(logger2.load_candidates()) == _strip(full_log.load_candidates())


def test_truncate_resets_cumulative_time_base(tmp_path):
    logger = RunLogger(str(tmp_path))
    state = init_state(CFG)
    step(state, CFG, fake_evaluator(), logger)
    step(state, CFG, fake_evaluator(), logger)
    logger.truncate_from_generation(1)
    assert [r["generation"] for r in logger.load_generations()] == [0]
    assert all(r["generation"] < 1 for r in logger.load_candidates())
