"""Checkpoint / resume for SearchState. Written after every generation, atomically, so an
interrupted write can never leave a corrupt checkpoint behind."""

import os

import torch

from evolution.search import SearchState
import random

CHECKPOINT_NAME = "checkpoint.pt"
FORMAT_VERSION = 1


def save_checkpoint(state: SearchState, path: str) -> None:
    payload = {
        "version": FORMAT_VERSION,
        "generation": state.generation,
        "population": state.population,
        "lineage": state.lineage,
        "rng_state": state.rng.getstate(),
        "cache": state.cache,
        "weights": state.weights,
        "genomes": state.genomes,
        "best_fitness": state.best_fitness,
        "best_genome": state.best_genome,
        "done": state.done,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> SearchState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {payload.get('version')}")
    rng = random.Random()
    rng.setstate(payload["rng_state"])
    return SearchState(
        generation=payload["generation"],
        population=payload["population"],
        lineage=payload["lineage"],
        rng=rng,
        cache=payload["cache"],
        weights=payload["weights"],
        genomes=payload["genomes"],
        best_fitness=payload["best_fitness"],
        best_genome=payload["best_genome"],
        done=payload["done"],
    )


def checkpoint_path(run_dir: str) -> str:
    return os.path.join(run_dir, CHECKPOINT_NAME)
