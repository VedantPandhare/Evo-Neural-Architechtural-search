import json
import os
import random

from evaluation.metrics import EvalResult
from evaluation.run_logger import (
    GENERATION_FIELDS, RunLogger, genome_distance, layer_signature, population_diversity,
)
from evolution.inheritance import InheritanceReport
from evolution.population import EvolutionConfig, init_population
from genomes.operators import genome_hash


def _pop(n=6, seed=0):
    return init_population(n, random.Random(seed))


def _result(f=0.5):
    return EvalResult(accuracy=0.6, n_params=1234, latency=0.01, fitness=f)


def test_genome_distance_identical_is_zero_and_symmetric_and_bounded():
    a, b = _pop(2)
    assert genome_distance(a, a) == 0.0
    assert genome_distance(a, b) == genome_distance(b, a)
    assert 0.0 < genome_distance(a, b) <= 1.0


def test_diversity_zero_for_clones_positive_for_varied_population():
    g = _pop(1)[0]
    assert population_diversity([g, g, g]) == 0.0
    assert population_diversity(_pop(8)) > 0.0
    assert population_diversity([g]) == 0.0


def test_layer_signature_distinguishes_filter_counts():
    g = _pop(1)[0]
    h = json.loads(json.dumps(g))
    conv = next(l for l in h["layers"] if l["type"] == "conv2d")
    conv["filters"] += 1
    assert layer_signature(g) != layer_signature(h)


def test_config_written_once_and_includes_environment(tmp_path):
    cfg = EvolutionConfig(population_size=7, seed=3)
    RunLogger(str(tmp_path), config=cfg, device="cpu")
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["config"]["population_size"] == 7 and data["config"]["seed"] == 3
    assert data["environment"]["device"] == "cpu" and "python" in data["environment"]
    RunLogger(str(tmp_path), config=EvolutionConfig(population_size=99))  # resume must not overwrite
    assert json.loads((tmp_path / "config.json").read_text())["config"]["population_size"] == 7


def test_candidate_records_roundtrip_with_inheritance_and_lineage(tmp_path):
    log = RunLogger(str(tmp_path))
    g = _pop(1)[0]
    report = InheritanceReport(inherited_params=60, total_params=100, n_exact=2, n_partial=1, n_fresh=1)
    log.log_candidate(2, genome_hash(g), g, _result(0.42), train_seconds=3.5,
                      inheritance=report, parents=("aaa", "bbb"))
    log.log_candidate(0, genome_hash(g), g, _result(0.1), train_seconds=1.0)
    recs = log.load_candidates()
    assert len(recs) == 2
    assert recs[0]["inherit_fraction"] == 0.6 and recs[0]["parents"] == ["aaa", "bbb"]
    assert recs[0]["inherit_exact"] == 2 and recs[0]["inherit_fresh"] == 1
    assert recs[1]["inherit_fraction"] == 0.0 and recs[1]["parents"] is None
    assert recs[0]["genome"] == g


def test_generation_rows_have_expected_stats_and_header_written_once(tmp_path):
    log = RunLogger(str(tmp_path))
    pop = _pop(4)
    fits = [0.1, 0.2, 0.3, 0.4]
    cands = [{"inherit_fraction": 0.5}, {"inherit_fraction": 0.7}]
    row = log.log_generation(0, pop, fits, new_evaluations=2, candidates=cands)
    assert row["best"] == 0.4 and row["min"] == 0.1 and abs(row["mean"] - 0.25) < 1e-12
    assert abs(row["mean_inherit_fraction"] - 0.6) < 1e-12
    log.log_generation(1, pop, fits, new_evaluations=0)
    lines = (tmp_path / "generations.csv").read_text().strip().splitlines()
    assert lines[0].split(",") == GENERATION_FIELDS
    assert len(lines) == 3
    gens = log.load_generations()
    assert [g["generation"] for g in gens] == [0, 1]
    assert gens[1]["cumulative_seconds"] >= gens[0]["cumulative_seconds"]


def test_resume_continues_generation_count_and_cumulative_time(tmp_path):
    log = RunLogger(str(tmp_path))
    pop, fits = _pop(3), [0.1, 0.2, 0.3]
    log.log_generation(0, pop, fits, 3)
    first_total = log.load_generations()[-1]["cumulative_seconds"]

    resumed = RunLogger(str(tmp_path))
    assert resumed.completed_generations() == 1
    resumed.log_generation(1, pop, fits, 0)
    rows = resumed.load_generations()
    assert len(rows) == 2 and rows[1]["cumulative_seconds"] >= first_total
    assert len((tmp_path / "generations.csv").read_text().strip().splitlines()) == 3  # one header only


def test_summary_includes_wallclock_hours(tmp_path):
    log = RunLogger(str(tmp_path))
    log.log_summary({"best_fitness": 0.7})
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["best_fitness"] == 0.7 and s["total_seconds"] >= 0
    assert abs(s["total_gpu_hours_wallclock"] - s["total_seconds"] / 3600) < 1e-12


def test_empty_run_dir_loads_cleanly(tmp_path):
    log = RunLogger(os.path.join(str(tmp_path), "nested", "run"))
    assert log.load_candidates() == [] and log.load_generations() == []
    assert log.completed_generations() == 0
