"""Per-run experiment logging: plain files, append-only so a resumed run just keeps writing.

Files in run_dir:
  config.json       run configuration + environment, written once (kept on resume)
  candidates.jsonl  one record per evaluated candidate (full genome, metrics, inheritance, lineage)
  generations.csv   one row per generation (fitness stats, diversity, inheritance, time)
  summary.json      final result, written at the end of the run
"""

import csv
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, is_dataclass

GENERATION_FIELDS = [
    "generation", "best", "mean", "min", "std", "new_evaluations", "diversity",
    "mean_inherit_fraction", "gen_seconds", "cumulative_seconds",
]


def layer_signature(genome: dict) -> list:
    """Hashable token per layer, used to measure how different two architectures are."""
    sig = []
    for l in genome["layers"]:
        t = l["type"]
        if t == "conv2d":
            sig.append(("conv2d", l["filters"], l["kernel_size"]))
        elif t == "dense":
            sig.append(("dense", l["units"]))
        elif t in ("maxpool", "avgpool"):
            sig.append((t, l["kernel_size"], l["stride"]))
        elif t == "dropout":
            sig.append(("dropout", l["rate"]))
        else:
            sig.append((t,))
    return sig


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def genome_distance(a: dict, b: dict) -> float:
    """Normalised layer-sequence edit distance in [0, 1]; 0 means identical layer stacks."""
    sa, sb = layer_signature(a), layer_signature(b)
    return _edit_distance(sa, sb) / max(len(sa), len(sb), 1)


def population_diversity(population: list) -> float:
    """Mean pairwise genome_distance. 0 when all architectures are identical."""
    n = len(population)
    if n < 2:
        return 0.0
    total = sum(genome_distance(population[i], population[j]) for i in range(n) for j in range(i + 1, n))
    return total / (n * (n - 1) / 2)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info(device: str | None = None) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "git_commit": _git_commit()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    if device:
        info["device"] = device
    return info


class RunLogger:
    def __init__(self, run_dir: str, config: dict | None = None, device: str | None = None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._paths = {n: os.path.join(run_dir, n) for n in
                       ("config.json", "candidates.jsonl", "generations.csv", "summary.json")}
        self._start = time.perf_counter()
        self._prior_seconds = 0.0
        self._gen_start = self._start

        if not os.path.exists(self._paths["config.json"]):
            cfg = asdict(config) if is_dataclass(config) else dict(config or {})
            self._write_json("config.json", {"config": cfg, "environment": environment_info(device)})
        self._prior_seconds = self._last_cumulative_seconds()

    # ---- writing ----
    def _write_json(self, name: str, obj) -> None:
        with open(self._paths[name], "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    def log_candidate(self, generation: int, genome_hash: str, genome: dict, result,
                      train_seconds: float, inheritance=None, parents: tuple | None = None) -> None:
        """`result` is an evaluation.metrics.EvalResult; `inheritance` an InheritanceReport or None."""
        rec = {
            "generation": generation,
            "genome_hash": genome_hash,
            "fitness": result.fitness,
            "accuracy": result.accuracy,
            "n_params": result.n_params,
            "latency": result.latency,
            "train_seconds": train_seconds,
            "parents": list(parents) if parents else None,
            "inherit_fraction": inheritance.fraction if inheritance else 0.0,
            "inherit_exact": inheritance.n_exact if inheritance else 0,
            "inherit_partial": inheritance.n_partial if inheritance else 0,
            "inherit_fresh": inheritance.n_fresh if inheritance else None,
            "genome": genome,
        }
        with open(self._paths["candidates.jsonl"], "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def log_generation(self, generation: int, population: list, fitnesses: list,
                       new_evaluations: int, candidates: list | None = None) -> dict:
        """`candidates` are this generation's newly evaluated records (as written by log_candidate),
        used for the inheritance fraction. Returns the row that was written."""
        now = time.perf_counter()
        fractions = [c["inherit_fraction"] for c in (candidates or [])]
        row = {
            "generation": generation,
            "best": max(fitnesses),
            "mean": statistics.fmean(fitnesses),
            "min": min(fitnesses),
            "std": statistics.pstdev(fitnesses),
            "new_evaluations": new_evaluations,
            "diversity": population_diversity(population),
            "mean_inherit_fraction": statistics.fmean(fractions) if fractions else 0.0,
            "gen_seconds": now - self._gen_start,
            "cumulative_seconds": self._prior_seconds + (now - self._start),
        }
        self._gen_start = now
        path = self._paths["generations.csv"]
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GENERATION_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        return row

    def log_summary(self, summary: dict) -> None:
        summary = dict(summary)
        summary["total_seconds"] = self._prior_seconds + (time.perf_counter() - self._start)
        summary["total_gpu_hours_wallclock"] = summary["total_seconds"] / 3600
        self._write_json("summary.json", summary)

    # ---- reading / resume ----
    def load_candidates(self) -> list:
        path = self._paths["candidates.jsonl"]
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def load_generations(self) -> list:
        path = self._paths["generations.csv"]
        if not os.path.exists(path):
            return []
        with open(path, newline="") as f:
            return [{k: (int(v) if k in ("generation", "new_evaluations") else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(f)]

    def completed_generations(self) -> int:
        return len(self.load_generations())

    def _last_cumulative_seconds(self) -> float:
        rows = self.load_generations()
        return rows[-1]["cumulative_seconds"] if rows else 0.0
