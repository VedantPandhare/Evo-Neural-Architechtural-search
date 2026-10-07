import json
import os
import random

import pytest

from final_eval import (
    baseline_genome, collect_models, format_final, load_results, main, run_final, summarize_final, train_final,
)
from genomes.schema import random_genome, validate_genome
from models.builder import build_model
from training.trainer import TrainConfig

TINY = TrainConfig(epochs=1, device="cpu", data_dir="./data", use_test_set=True, gpu_data=True,
                   max_train_samples=100, max_val_samples=50)


def _valid_genome(seed=0):
    rng = random.Random(seed)
    return next(g for g in (random_genome(rng, max_conv_layers=2, max_dense_layers=1) for _ in range(50))
                if validate_genome(g).valid)


def _fake_search_run(out_dir, name, genome, fitness, search_acc=None):
    d = os.path.join(out_dir, name)
    os.makedirs(d)
    if search_acc is not None:
        from genomes.operators import genome_hash
        with open(os.path.join(d, "candidates.jsonl"), "w") as f:
            f.write(json.dumps({"genome_hash": genome_hash(genome), "accuracy": search_acc,
                                "generation": 4, "inherit_fraction": 0.7}) + "\n")
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump({"best_genome": genome, "best_fitness": fitness}, f)


def test_baseline_is_valid_builds_and_has_comparable_size():
    g = baseline_genome()
    assert validate_genome(g).valid
    n = sum(p.numel() for p in build_model(g).parameters())
    assert 3e5 < n < 1.5e6


def test_collect_models_reads_finished_runs_skips_unfinished_and_adds_baseline(tmp_path):
    out = str(tmp_path)
    _fake_search_run(out, "search_inherit_seed0", _valid_genome(0), 0.75)
    _fake_search_run(out, "search_scratch_seed0", _valid_genome(1), 0.70)
    _fake_search_run(out, "search_inherit_seed1", _valid_genome(2), 0.72)  # scratch seed 1 unfinished
    models = collect_models(out, [0, 1])
    assert [m["name"] for m in models] == ["on_seed0", "off_seed0", "on_seed1", "baseline"]
    assert models[0]["search_fitness"] == 0.75 and models[-1]["arm"] == "baseline"
    assert [m["name"] for m in collect_models(out, [0], include_baseline=False)] == ["on_seed0", "off_seed0"]


def test_train_final_uses_test_set_and_returns_metrics():
    assert TINY.use_test_set
    res = train_final(_valid_genome(), TINY, train_seed=0)
    assert 0.0 <= res["test_accuracy"] <= 1.0 and res["n_params"] > 0 and res["latency"] > 0


def test_run_final_resumes_without_retraining_finished_models(tmp_path, monkeypatch):
    calls = []

    def fake_train(genome, cfg, train_seed):
        calls.append(train_seed)
        return {"test_accuracy": 0.5 + 0.01 * train_seed, "n_params": 10, "latency": 0.001}

    monkeypatch.setattr("final_eval.train_final", fake_train)
    models = [{"name": "on_seed0", "arm": "on", "seed": 0, "genome": {}, "search_fitness": 0.7}]
    path = str(tmp_path / "final" / "results.jsonl")
    run_final(models, TINY, [0, 1], path)
    assert calls == [0, 1]
    rows = run_final(models, TINY, [0, 1, 2], path)  # resume: only train_seed 2 is new
    assert calls == [0, 1, 2] and len(rows) == 3 and len(load_results(path)) == 3


def _row(name, arm, seed, acc, ts=0):
    return {"name": name, "arm": arm, "seed": seed, "train_seed": ts, "test_accuracy": acc, "n_params": 100,
            "latency": 0.002, "search_fitness": 0.7}


def test_summarize_final_means_over_train_seeds_and_pairs_by_search_seed():
    rows = [_row("on_seed0", "on", 0, 0.80, 0), _row("on_seed0", "on", 0, 0.82, 1),
            _row("off_seed0", "off", 0, 0.78), _row("on_seed1", "on", 1, 0.79), _row("off_seed1", "off", 1, 0.80),
            _row("baseline", "baseline", None, 0.75)]
    s = summarize_final(rows)
    assert abs(s["models"]["on_seed0"]["test_accuracy"] - 0.81) < 1e-9 and s["models"]["on_seed0"]["n_train_seeds"] == 2
    assert [round(p["on_minus_off_test_accuracy"], 4) for p in s["paired"]] == [0.03, -0.01]
    assert s["on_better"] == "1/2" and abs(s["paired_mean_std"][0] - 0.01) < 1e-9
    assert s["arms"]["baseline"]["n"] == 1 and s["arms"]["on"]["n"] == 2
    text = format_final(s)
    assert "baseline" in text and "paired ON-OFF" in text


def test_main_end_to_end_on_tiny_data(tmp_path):
    out = str(tmp_path)
    _fake_search_run(out, "search_inherit_seed0", _valid_genome(0), 0.75)
    _fake_search_run(out, "search_scratch_seed0", _valid_genome(1), 0.70)
    s = main(["--out-dir", out, "--seeds", "0", "--epochs", "1", "--device", "cpu", "--no-baseline",
              "--max-train-samples", "100", "--max-val-samples", "50"])
    assert set(s["models"]) == {"on_seed0", "off_seed0"}
    assert os.path.exists(os.path.join(out, "final", "results.jsonl"))
    assert os.path.exists(os.path.join(out, "final", "summary.txt"))


def test_collect_models_attaches_search_time_accuracy(tmp_path):
    out = str(tmp_path)
    _fake_search_run(out, "search_inherit_seed0", _valid_genome(0), 0.75, search_acc=0.78)
    _fake_search_run(out, "search_scratch_seed0", _valid_genome(1), 0.70)  # no candidates.jsonl
    on, off, base = collect_models(out, [0])
    assert on["search_accuracy"] == 0.78 and on["search_generation"] == 4 and on["search_inherit_fraction"] == 0.7
    assert off["search_accuracy"] is None and base["search_accuracy"] is None


def test_format_final_shows_drop_between_search_and_final_accuracy():
    rows = [dict(_row("on_seed0", "on", 0, 0.65), search_accuracy=0.78, search_inherit_fraction=0.7),
            dict(_row("off_seed0", "off", 0, 0.85), search_accuracy=0.70, search_inherit_fraction=0.0)]
    text = format_final(summarize_final(rows))
    assert "-0.130" in text and "+0.150" in text


def test_val_split_mode_writes_to_separate_folder_and_never_uses_test_set(tmp_path):
    out = str(tmp_path)
    _fake_search_run(out, "search_inherit_seed0", _valid_genome(0), 0.75, search_acc=0.7)
    _fake_search_run(out, "search_scratch_seed0", _valid_genome(1), 0.70, search_acc=0.6)
    main(["--out-dir", out, "--seeds", "0", "--epochs", "1", "--device", "cpu", "--no-baseline",
          "--eval-split", "val", "--max-train-samples", "100", "--max-val-samples", "50"])
    assert os.path.exists(os.path.join(out, "final_val", "results.jsonl"))
    assert not os.path.exists(os.path.join(out, "final"))
    rows = load_results(os.path.join(out, "final_val", "results.jsonl"))
    assert {r["eval_split"] for r in rows} == {"val"}
