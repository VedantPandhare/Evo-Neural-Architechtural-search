"""EvoNAS entry point.

  python main.py --config configs/search_inherit.yaml
  python main.py --config configs/search_inherit.yaml --resume      # continue after a disconnect

Each run writes to <output_dir>/<run_name>/ : config.json, candidates.jsonl, generations.csv,
summary.json and checkpoint.pt (see evaluation/run_logger.py and evolution/checkpoint.py).
"""

import argparse
import os
import sys

import torch

from configs.loader import load_config
from evaluation.metrics import FitnessConfig
from evaluation.run_logger import RunLogger
from evolution.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from evolution.population import EvolutionConfig
from evolution.search import init_state, make_evaluator, run_search
from training.trainer import TrainConfig


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def build_run(cfg: dict) -> tuple:
    """YAML dict -> (EvolutionConfig, TrainConfig, FitnessConfig, inheritance flag)."""
    train = dict(cfg["training"])
    train["device"] = resolve_device(cfg.get("device", "auto"))
    return (
        EvolutionConfig(**cfg["evolution"]),
        TrainConfig(**train),
        FitnessConfig(**cfg["fitness"]),
        bool(cfg["inheritance"]),
    )


def run_dir_for(cfg: dict) -> str:
    return os.path.join(cfg.get("output_dir", "results"), cfg["run_name"])


def main(argv: list | None = None):
    ap = argparse.ArgumentParser(description="EvoNAS: evolutionary NAS with weight-inherited offspring")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from the run directory's checkpoint")
    ap.add_argument("--inheritance", choices=["on", "off"], help="override the config's inheritance flag")
    ap.add_argument("--output-dir", help="override the config's output_dir (e.g. a Google Drive folder)")
    ap.add_argument("--seed", type=int, help="override evolution.seed")
    ap.add_argument("--run-name", help="override run_name")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.inheritance:
        cfg["inheritance"] = args.inheritance == "on"
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["evolution"]["seed"] = args.seed
    if args.run_name:
        cfg["run_name"] = args.run_name

    evo_cfg, train_cfg, fit_cfg, inheritance = build_run(cfg)
    run_dir = run_dir_for(cfg)
    ckpt = checkpoint_path(run_dir)

    if os.path.exists(ckpt) and not args.resume:
        sys.exit(f"{run_dir} already has a checkpoint. Use --resume to continue it, or change --run-name.")
    if args.resume and not os.path.exists(ckpt):
        sys.exit(f"--resume given but {ckpt} does not exist.")

    logger = RunLogger(run_dir, config=cfg, device=train_cfg.device)
    if args.resume:
        state = load_checkpoint(ckpt)
        if state.done:
            print(f"{run_dir} is already complete (best fitness {state.best_fitness:.4f}).")
            return state
        logger.truncate_from_generation(state.generation)
        print(f"Resuming {cfg['run_name']} at generation {state.generation}/{evo_cfg.generations}")
    else:
        state = init_state(evo_cfg)

    evaluate = make_evaluator(train_cfg, fit_cfg)

    def on_step(st, row):
        save_checkpoint(st, ckpt)
        print(f"gen {row['generation']:>3}  best {row['best']:.4f}  mean {row['mean']:.4f}  "
              f"new_evals {row['new_evaluations']}  inherit {row.get('mean_inherit_fraction', 0.0):.2f}  "
              f"t {row.get('cumulative_seconds', 0.0) / 60:.1f} min", flush=True)

    state = run_search(evo_cfg, evaluate, logger, inheritance=inheritance, state=state, on_step=on_step)
    print(f"Done. Best fitness {state.best_fitness:.4f}. Results in {run_dir}")
    return state


if __name__ == "__main__":
    main()
