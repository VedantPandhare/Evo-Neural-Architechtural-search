import json

import pytest
import yaml

from main import build_run, main
from training.trainer import TrainConfig

BASE = {
    "run_name": "t",
    "inheritance": True,
    "device": "cpu",
    "evolution": {"population_size": 4, "generations": 2, "tournament_size": 2, "elite_fraction": 0.25,
                  "crossover_rate": 0.5, "mutation_rate": 0.8, "seed": 0},
    "training": {"epochs": 1, "max_train_samples": 100, "max_val_samples": 50, "gpu_data": True},
    "fitness": {"lambda_params": 0.1, "lambda_latency": 0.1, "param_reference": 2000000,
                "latency_reference": 0.05},
}


def _write(tmp_path, **over):
    cfg = json.loads(json.dumps(BASE))
    cfg.update(over)
    cfg["output_dir"] = str(tmp_path / "out")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_shipped_configs_all_parse_and_scratch_differs_only_in_inheritance():
    from configs.loader import load_config
    cfgs = {n: load_config(f"configs/{n}.yaml") for n in ("smoke", "search_inherit", "search_scratch")}
    for c in cfgs.values():
        evo, train, fit, inh = build_run(c)
        assert isinstance(train, TrainConfig) and evo.population_size > 0
    assert all("run_name" in c for c in cfgs.values())
    a, b = cfgs["search_inherit"], cfgs["search_scratch"]
    assert a["run_name"] != b["run_name"]
    assert a["inheritance"] is True and b["inheritance"] is False
    strip = lambda c: {k: v for k, v in c.items() if k not in ("inheritance", "run_name")}
    assert strip(a) == strip(b)  # paired comparison: seed, budget, lambdas identical


def test_device_auto_resolves_to_known_device():
    cfg = json.loads(json.dumps(BASE))
    cfg["device"] = "auto"
    assert build_run(cfg)[1].device in ("cpu", "cuda")


def test_end_to_end_run_writes_all_outputs_and_summary(tmp_path):
    state = main(["--config", _write(tmp_path)])
    run = tmp_path / "out" / "t"
    assert state.done
    for name in ("config.json", "candidates.jsonl", "generations.csv", "summary.json", "checkpoint.pt"):
        assert (run / name).exists(), name
    summary = json.loads((run / "summary.json").read_text())
    assert summary["inheritance"] is True and summary["best_fitness"] == state.best_fitness
    rows = (run / "generations.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + BASE["evolution"]["generations"]


def test_rerun_without_resume_is_refused_and_resume_of_finished_run_is_a_noop(tmp_path):
    cfg = _write(tmp_path)
    first = main(["--config", cfg])
    with pytest.raises(SystemExit):
        main(["--config", cfg])
    again = main(["--config", cfg, "--resume"])
    assert again.best_fitness == first.best_fitness


def test_resume_without_checkpoint_errors(tmp_path):
    with pytest.raises(SystemExit):
        main(["--config", _write(tmp_path), "--resume"])


def test_cli_overrides_apply(tmp_path):
    cfg = _write(tmp_path)
    main(["--config", cfg, "--inheritance", "off", "--seed", "3", "--run-name", "o", "--output-dir", str(tmp_path / "x")])
    out = json.loads((tmp_path / "x" / "o" / "config.json").read_text())["config"]
    assert out["inheritance"] is False and out["evolution"]["seed"] == 3
    assert all(r["inherit_fraction"] == 0.0
               for r in map(json.loads, (tmp_path / "x" / "o" / "candidates.jsonl").read_text().splitlines()))
