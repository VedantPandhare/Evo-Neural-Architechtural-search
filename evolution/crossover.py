"""Layer-level crossover.

Child = conv block (layers before flatten) spliced at a random cut point from both parents,
dense block (flatten onward) taken whole from one random parent, scalar hyperparameters
(optimizer, lr, batch size, dropout) each taken from a random parent.

Repair policy (documented for the paper): invalid children are rejected and new cut points are
resampled up to `max_attempts` times; if none is valid the child is a copy of parent A."""

import copy
import random

from genomes.schema import default_genome_space, validate_genome

HYPERPARAMS = ("optimizer", "learning_rate", "batch_size", "dropout")


def _split(genome: dict) -> tuple:
    layers = genome["layers"]
    f = next(i for i, l in enumerate(layers) if l["type"] == "flatten")
    return layers[:f], layers[f:]


def crossover(a: dict, b: dict, rng: random.Random, space: dict | None = None, max_attempts: int = 10) -> dict:
    space = space or default_genome_space()
    conv_a, tail_a = _split(a)
    conv_b, tail_b = _split(b)
    for _ in range(max_attempts):
        cut_a = rng.randint(0, len(conv_a))
        cut_b = rng.randint(0, len(conv_b))
        conv = conv_a[:cut_a] + conv_b[cut_b:]
        tail = tail_a if rng.random() < 0.5 else tail_b
        child = copy.deepcopy(a)
        child["layers"] = copy.deepcopy(conv + tail)
        for key in HYPERPARAMS:
            child[key] = (a if rng.random() < 0.5 else b)[key]
        n_conv = sum(1 for l in child["layers"] if l["type"] == "conv2d")
        if n_conv < 1 or n_conv > space["max_conv_layers"]:
            continue
        if validate_genome(child).valid:
            return child
    return copy.deepcopy(a)
