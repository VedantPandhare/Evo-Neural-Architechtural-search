"""Mutation operators. Each operator takes (genome, rng, space) and returns a new genome,
or None when it is not applicable. Operators never modify their input.

Repair policy (documented for the paper): `mutate` validates the result and rejects invalid
offspring, resampling a different operator up to `max_attempts` times; if none succeeds the
offspring is an unmodified copy of the parent."""

import copy
import random

from genomes.schema import default_genome_space, random_layer_spec, validate_genome


def _flatten_index(genome: dict) -> int:
    for i, layer in enumerate(genome["layers"]):
        if layer["type"] == "flatten":
            return i
    raise ValueError("genome has no flatten layer")


def _indices(genome: dict, types: tuple) -> list:
    return [i for i, l in enumerate(genome["layers"]) if l["type"] in types]


def add_layer(genome, rng, space):
    g = copy.deepcopy(genome)
    f = _flatten_index(g)
    kinds = []
    if len(_indices(g, ("conv2d",))) < space["max_conv_layers"]:
        kinds.append("conv2d")
    if len(_indices(g, ("dense",))) < space["max_dense_layers"]:
        kinds.append("dense")
    if not kinds:
        return None
    kind = rng.choice(kinds)
    layer = random_layer_spec(kind, rng, space)
    if kind == "conv2d":
        g["layers"].insert(rng.randint(0, f), layer)
    else:
        g["layers"].insert(rng.randint(f + 1, len(g["layers"])), layer)
    return g


def remove_layer(genome, rng, space):
    g = copy.deepcopy(genome)
    n_conv = len(_indices(g, ("conv2d",)))
    n_dense = len(_indices(g, ("dense",)))
    candidates = _indices(g, ("maxpool", "avgpool", "dropout"))
    candidates += [i for i in _indices(g, ("conv2d",)) if n_conv > 1]
    candidates += [i for i in _indices(g, ("dense",)) if n_dense > 1]
    if not candidates:
        return None
    del g["layers"][rng.choice(candidates)]
    return g


def _change_field(genome, rng, ltype, field, choices):
    g = copy.deepcopy(genome)
    idx = _indices(g, (ltype,))
    if not idx:
        return None
    i = rng.choice(idx)
    options = [c for c in choices if c != g["layers"][i][field]]
    if not options:
        return None
    g["layers"][i][field] = rng.choice(options)
    return g


def change_filters(genome, rng, space):
    return _change_field(genome, rng, "conv2d", "filters", space["conv_filters"])


def change_kernel_size(genome, rng, space):
    return _change_field(genome, rng, "conv2d", "kernel_size", space["kernel_sizes"])


def change_dense_units(genome, rng, space):
    return _change_field(genome, rng, "dense", "units", space["dense_units"])


def change_activation(genome, rng, space):
    ltype = rng.choice(["conv2d", "dense"])
    return _change_field(genome, rng, ltype, "activation", space["activations"])


def change_dropout(genome, rng, space):
    g = copy.deepcopy(genome)
    idx = _indices(g, ("dropout",))
    if idx and rng.random() < 0.5:
        layer = g["layers"][rng.choice(idx)]
        layer["rate"] = rng.choice([r for r in space["dropout_rates"] if r != layer["rate"]])
    else:
        g["dropout"] = rng.choice([r for r in space["dropout_rates"] if r != g["dropout"]])
    return g


def change_lr_optimizer(genome, rng, space):
    g = copy.deepcopy(genome)
    if rng.random() < 0.5:
        g["learning_rate"] = rng.choice([lr for lr in space["learning_rates"] if lr != g["learning_rate"]])
    else:
        g["optimizer"] = rng.choice([o for o in space["optimizers"] if o != g["optimizer"]])
    return g


def replace_layer_type(genome, rng, space):
    """Swap a pooling layer's kind (max <-> avg), keeping kernel size and stride."""
    g = copy.deepcopy(genome)
    idx = _indices(g, ("maxpool", "avgpool"))
    if not idx:
        return None
    layer = g["layers"][rng.choice(idx)]
    layer["type"] = "avgpool" if layer["type"] == "maxpool" else "maxpool"
    return g


MUTATION_OPERATORS = {
    "add_layer": add_layer,
    "remove_layer": remove_layer,
    "change_filters": change_filters,
    "change_kernel_size": change_kernel_size,
    "change_dense_units": change_dense_units,
    "change_activation": change_activation,
    "change_dropout": change_dropout,
    "change_lr_optimizer": change_lr_optimizer,
    "replace_layer_type": replace_layer_type,
}


def mutate(genome: dict, rng: random.Random, space: dict | None = None, max_attempts: int = 10) -> tuple:
    """Apply one randomly chosen operator. Returns (child, operator_name); operator_name is
    None when no valid mutation was found and the child is an unmodified copy."""
    space = space or default_genome_space()
    names = list(MUTATION_OPERATORS)
    for _ in range(max_attempts):
        name = rng.choice(names)
        child = MUTATION_OPERATORS[name](genome, rng, space)
        if child is not None and validate_genome(child).valid:
            return child, name
    return copy.deepcopy(genome), None
