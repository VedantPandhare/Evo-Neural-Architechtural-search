"""Stage 7: final evaluation. Retrain each search run's best architecture, plus a hand-designed
baseline, from scratch on the full 50k training set and test ONCE on the official 10k test set.

  python final_eval.py --out-dir /content/drive/MyDrive/evonas_results --seeds 0 1 2 3 --epochs 30

Protocol (state this in the paper):
  * every model, both search arms and the baseline, gets the same epochs, augmentation and data;
    each model keeps the optimizer / learning rate / batch size stored in its own genome
  * weights start from the shared Kaiming init: no inheritance, no search-time weights
  * the test set is only evaluated after the last epoch, and never used to pick anything
  * --eval-split val does the same on a held-out validation split (45k train / 5k val): use it while
    developing method changes, and keep --eval-split test for the final confirmatory run
Results are appended to <out-dir>/final/results.jsonl, so an interrupted run resumes where it stopped.
"""

import argparse
import json
import os
import statistics

import torch

from evaluation.metrics import count_parameters, measure_latency
from evolution.inheritance import build_offspring
from genomes.operators import genome_hash
from genomes.schema import validate_genome
from training.trainer import TrainConfig, train_and_evaluate

ARM_DIRS = {"on": "search_inherit_seed{seed}", "off": "search_scratch_seed{seed}"}


def baseline_genome() -> dict:
    """VGG-style small CNN of comparable size (~0.6M parameters): two conv blocks with BatchNorm."""
    def conv(f):
        return {"type": "conv2d", "filters": f, "kernel_size": 3, "stride": 1, "padding": "same",
                "activation": "relu", "batchnorm": True}
    pool = {"type": "maxpool", "kernel_size": 2, "stride": 2}
    genome = {
        "layers": [conv(32), conv(32), dict(pool), conv(64), conv(64), dict(pool), {"type": "flatten"},
                   {"type": "dense", "units": 128, "activation": "relu", "batchnorm": False}],
        "optimizer": "adam", "learning_rate": 1e-3, "batch_size": 128, "dropout": 0.0,
        "input_shape": [3, 32, 32], "num_classes": 10,
    }
    assert validate_genome(genome).valid
    return genome


def _search_record(run_dir: str, genome: dict):
    """The candidates.jsonl record the search logged for this exact genome, or None."""
    path = os.path.join(run_dir, "candidates.jsonl")
    if not os.path.exists(path):
        return None
    h = genome_hash(genome)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["genome_hash"] == h:
                return rec
    return None


def collect_models(out_dir: str, seeds, include_baseline: bool = True) -> list:
    """[{name, arm, seed, genome, search_fitness, search_accuracy, ...}] for every finished search
    run, plus the baseline. search_accuracy is the validation accuracy the search measured for that
    genome (with inherited weights in the ON arm), to compare against its from-scratch accuracy."""
    models = []
    for seed in seeds:
        for arm, pattern in ARM_DIRS.items():
            run_dir = os.path.join(out_dir, pattern.format(seed=seed))
            path = os.path.join(run_dir, "summary.json")
            if not os.path.exists(path):
                print(f"skipping {arm} seed {seed}: no summary.json (search not finished)")
                continue
            with open(path) as f:
                summary = json.load(f)
            rec = _search_record(run_dir, summary["best_genome"])
            models.append({"name": f"{arm}_seed{seed}", "arm": arm, "seed": seed,
                           "genome": summary["best_genome"], "search_fitness": summary["best_fitness"],
                           "search_accuracy": rec["accuracy"] if rec else None,
                           "search_generation": rec["generation"] if rec else None,
                           "search_inherit_fraction": rec["inherit_fraction"] if rec else None})
    if include_baseline:
        models.append({"name": "baseline", "arm": "baseline", "seed": None, "genome": baseline_genome(),
                       "search_fitness": None, "search_accuracy": None, "search_generation": None,
                       "search_inherit_fraction": None})
    return models


def train_final(genome: dict, cfg: TrainConfig, train_seed: int) -> dict:
    torch.manual_seed(train_seed)
    model, _ = build_offspring(genome, None)  # shared Kaiming init, no inheritance
    test_acc = train_and_evaluate(model, genome, cfg)  # cfg.use_test_set -> evaluated once, after the last epoch
    return {
        "test_accuracy": test_acc,
        "n_params": count_parameters(model),
        "latency": measure_latency(model, tuple(genome["input_shape"]), device=cfg.device),
    }


def load_results(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_final(models: list, cfg: TrainConfig, train_seeds, results_path: str) -> list:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    done = {(r["name"], r["train_seed"]) for r in load_results(results_path)}
    for m in models:
        for ts in train_seeds:
            if (m["name"], ts) in done:
                continue
            res = train_final(m["genome"], cfg, ts)
            rec = {"name": m["name"], "arm": m["arm"], "seed": m["seed"], "train_seed": ts,
                   "search_fitness": m["search_fitness"], "search_accuracy": m.get("search_accuracy"),
                   "search_generation": m.get("search_generation"),
                   "search_inherit_fraction": m.get("search_inherit_fraction"),
                   "eval_split": "test" if cfg.use_test_set else "val", "epochs": cfg.epochs,
                   **res, "genome": m["genome"]}
            with open(results_path, "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            print(f"{m['name']:>14} train_seed {ts}: test acc {res['test_accuracy']:.4f}  "
                  f"params {res['n_params']:,}  latency {res['latency'] * 1000:.2f} ms", flush=True)
    return load_results(results_path)


def summarize_final(rows: list) -> dict:
    """Per-model mean over training seeds, then per-arm mean/std over search seeds, and the paired
    ON-OFF difference in test accuracy (matched by search seed)."""
    per_model = {}
    for r in rows:
        per_model.setdefault(r["name"], []).append(r)
    models = {}
    for name, rs in per_model.items():
        models[name] = {
            "arm": rs[0]["arm"], "seed": rs[0]["seed"], "n_train_seeds": len(rs),
            "test_accuracy": statistics.fmean(r["test_accuracy"] for r in rs),
            "n_params": rs[0]["n_params"],
            "latency": statistics.fmean(r["latency"] for r in rs),
            "search_fitness": rs[0]["search_fitness"],
            "search_accuracy": rs[0].get("search_accuracy"),
            "search_inherit_fraction": rs[0].get("search_inherit_fraction"),
        }

    def ms(xs):
        return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (None, None)

    arms = {}
    for arm in ("on", "off", "baseline"):
        ms_ = [m for m in models.values() if m["arm"] == arm]
        arms[arm] = {k: ms([m[k] for m in ms_]) for k in ("test_accuracy", "n_params", "latency")}
        arms[arm]["n"] = len(ms_)

    paired = []
    for seed in {m["seed"] for m in models.values() if m["seed"] is not None}:
        on, off = models.get(f"on_seed{seed}"), models.get(f"off_seed{seed}")
        if on and off:
            paired.append({"seed": seed, "on_minus_off_test_accuracy": on["test_accuracy"] - off["test_accuracy"]})
    diffs = [p["on_minus_off_test_accuracy"] for p in paired]
    return {"models": models, "arms": arms, "paired": sorted(paired, key=lambda p: p["seed"]),
            "paired_mean_std": ms(diffs), "on_better": f"{sum(d > 0 for d in diffs)}/{len(diffs)}"}


def format_final(s: dict) -> str:
    lines = [f"{'model':>14} {'final acc':>9} {'search acc':>10} {'drop':>7} {'params':>10} {'latency ms':>10} "
             f"{'search fit':>10} {'inh.frac':>8}"]
    for name, m in sorted(s["models"].items()):
        fit = "" if m["search_fitness"] is None else f"{m['search_fitness']:.4f}"
        sa = m.get("search_accuracy")
        inh = m.get("search_inherit_fraction")
        sa_txt = "" if sa is None else format(sa, ".4f")
        drop_txt = "" if sa is None else format(m["test_accuracy"] - sa, "+.3f")
        inh_txt = "" if inh is None else format(inh, ".2f")
        lines.append(f"{name:>14} {m['test_accuracy']:>9.4f} {sa_txt:>10} {drop_txt:>7} {m['n_params']:>10,} "
                     f"{m['latency'] * 1000:>10.2f} {fit:>10} {inh_txt:>8}")
    lines.append("('final acc' = accuracy on the evaluation split after the full retrain; 'drop' = final minus the "
                 "accuracy the search measured. A large negative drop for ON but not OFF means the search "
                 "accuracy was inflated by inherited training.)")
    lines.append("")
    for arm, a in s["arms"].items():
        if a["n"]:
            acc, p, lat = a["test_accuracy"], a["n_params"], a["latency"]
            lines.append(f"{arm.upper():>8} (n={a['n']}): test acc {acc[0]:.4f} +/- {acc[1]:.4f} | "
                         f"params {p[0]:,.0f} | latency {lat[0] * 1000:.2f} ms")
    if s["paired"]:
        m, sd = s["paired_mean_std"]
        lines.append(f"paired ON-OFF test accuracy: {[round(p['on_minus_off_test_accuracy'], 4) for p in s['paired']]}  "
                     f"mean {m:+.4f} std {sd:.4f}  ON better in {s['on_better']} seeds")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Final retrain + single test-set evaluation")
    ap.add_argument("--out-dir", required=True, help="folder holding the search_* run directories")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-seeds", type=int, default=1, help="independent trainings per model (mean reported)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--eval-split", choices=["test", "val"], default="test",
                    help="test: train on 50k, evaluate on the official test set (use ONCE, at the very end). "
                         "val: train on 45k, evaluate on the held-out 5k validation split (method development).")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--max-train-samples", type=int, help="testing only: cap the training set")
    ap.add_argument("--max-val-samples", type=int, help="testing only: cap the test set")
    args = ap.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_test = args.eval_split == "test"
    cfg = TrainConfig(epochs=args.epochs, device=device, data_dir=args.data_dir, use_test_set=use_test,
                      gpu_data=True, augment=True, max_train_samples=args.max_train_samples,
                      max_val_samples=args.max_val_samples)
    models = collect_models(args.out_dir, args.seeds, include_baseline=not args.no_baseline)
    sub = "final" if use_test else "final_val"
    results_path = os.path.join(args.out_dir, sub, "results.jsonl")
    rows = run_final(models, cfg, range(args.train_seeds), results_path)

    summary = summarize_final(rows)
    text = format_final(summary)
    print("\n" + text)
    with open(os.path.join(args.out_dir, sub, "summary.txt"), "w") as f:
        f.write(text + "\n")
    return summary


if __name__ == "__main__":
    main()
