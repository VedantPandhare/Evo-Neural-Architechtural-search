"""Genome data structure, random sampling, and structural validation."""

import copy
import random
from dataclasses import dataclass

LAYER_TYPES = {"conv2d", "maxpool", "avgpool", "flatten", "dense", "dropout"}
CONV_LAYER_TYPES = {"conv2d", "maxpool", "avgpool"}
ACTIVATIONS = {"relu", "leaky_relu", "gelu", "tanh"}
DENSE_ACTIVATIONS = ACTIVATIONS | {"none"}
OPTIMIZERS = {"adam", "sgd", "rmsprop"}
BATCH_SIZES = [32, 64, 128, 256]


@dataclass
class ValidationResult:
    valid: bool
    reason: str | None = None


def default_genome_space() -> dict:
    """Single source of sampling ranges/choices, shared with mutation.py."""
    return {
        "conv_filters": [16, 32, 48, 64, 96, 128],
        "kernel_sizes": [3, 5],
        "pool_kernel_sizes": [2, 3],
        "dense_units": [32, 64, 128, 256, 512],
        "activations": sorted(ACTIVATIONS),
        "dropout_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "optimizers": sorted(OPTIMIZERS),
        "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "batch_sizes": BATCH_SIZES,
        "max_conv_layers": 5,
        "max_dense_layers": 3,
    }


def random_layer_spec(kind: str, rng: random.Random, space: dict | None = None) -> dict:
    space = space or default_genome_space()
    if kind == "conv2d":
        return {
            "type": "conv2d",
            "filters": rng.choice(space["conv_filters"]),
            "kernel_size": rng.choice(space["kernel_sizes"]),
            "stride": 1,
            "padding": "same",
            "activation": rng.choice(space["activations"]),
            "batchnorm": rng.random() < 0.5,
        }
    if kind in ("maxpool", "avgpool"):
        return {
            "type": kind,
            "kernel_size": rng.choice(space["pool_kernel_sizes"]),
            "stride": rng.choice(space["pool_kernel_sizes"]),
        }
    if kind == "flatten":
        return {"type": "flatten"}
    if kind == "dense":
        return {
            "type": "dense",
            "units": rng.choice(space["dense_units"]),
            "activation": rng.choice(space["activations"]),
            "batchnorm": rng.random() < 0.5,
        }
    if kind == "dropout":
        return {"type": "dropout", "rate": rng.choice(space["dropout_rates"])}
    raise ValueError(f"unknown layer kind: {kind}")


def random_genome(
    rng: random.Random | None = None,
    input_shape=(3, 32, 32),
    num_classes: int = 10,
    max_conv_layers: int | None = None,
    max_dense_layers: int | None = None,
) -> dict:
    """Generates a structurally-random genome that is valid by construction:
    conv/pool block first, flatten, then dense block."""
    rng = rng or random.Random()
    space = default_genome_space()
    max_conv_layers = max_conv_layers or space["max_conv_layers"]
    max_dense_layers = max_dense_layers or space["max_dense_layers"]

    layers = []
    h, w = input_shape[1], input_shape[2]
    n_conv = rng.randint(1, max_conv_layers)
    for _ in range(n_conv):
        layers.append(random_layer_spec("conv2d", rng, space))
        if rng.random() < 0.5:
            pool = random_layer_spec(rng.choice(["maxpool", "avgpool"]), rng, space)
            k, s = pool["kernel_size"], pool["stride"]
            new_h, new_w = (h - k) // s + 1, (w - k) // s + 1
            if new_h > 0 and new_w > 0:
                layers.append(pool)
                h, w = new_h, new_w

    layers.append({"type": "flatten"})

    n_dense = rng.randint(1, max_dense_layers)
    for _ in range(n_dense):
        layers.append(random_layer_spec("dense", rng, space))
        if rng.random() < 0.3:
            layers.append(random_layer_spec("dropout", rng, space))

    return {
        "layers": layers,
        "optimizer": rng.choice(space["optimizers"]),
        "learning_rate": rng.choice(space["learning_rates"]),
        "batch_size": rng.choice(space["batch_sizes"]),
        "dropout": rng.choice(space["dropout_rates"]),
        "input_shape": list(input_shape),
        "num_classes": num_classes,
    }


def _simulate_spatial_dims(layers: list[dict], input_shape: tuple[int, int, int]) -> tuple[int, int, int, str | None]:
    """Walks conv/pool layers up to (not including) flatten, tracking (channels, h, w).
    Returns (channels, h, w, error_reason). error_reason is None if all dims stayed positive."""
    channels, h, w = input_shape
    for layer in layers:
        if layer["type"] == "flatten":
            break
        if layer["type"] == "conv2d":
            channels = layer["filters"]
            # padding="same", stride=1 assumed to keep spatial dims (builder enforces this)
        elif layer["type"] in ("maxpool", "avgpool"):
            k, s = layer["kernel_size"], layer["stride"]
            h = (h - k) // s + 1
            w = (w - k) // s + 1
            if h <= 0 or w <= 0:
                return channels, h, w, f"spatial dims collapsed to non-positive size ({h}x{w})"
    return channels, h, w, None


def validate_genome(genome: dict) -> ValidationResult:
    layers = genome.get("layers")
    if not layers:
        return ValidationResult(False, "layers list is empty")

    seen_flatten_or_dense = False
    for i, layer in enumerate(layers):
        ltype = layer.get("type")
        if ltype not in LAYER_TYPES:
            return ValidationResult(False, f"unknown layer type at index {i}: {ltype}")
        if ltype in CONV_LAYER_TYPES and seen_flatten_or_dense:
            return ValidationResult(False, f"conv/pool layer at index {i} appears after flatten/dense")
        if ltype in ("flatten", "dense", "dropout"):
            seen_flatten_or_dense = True

    # dense must not directly follow a conv/pool layer without an intervening flatten
    saw_conv = False
    saw_flatten = False
    for i, layer in enumerate(layers):
        ltype = layer["type"]
        if ltype in CONV_LAYER_TYPES:
            saw_conv = True
        elif ltype == "flatten":
            saw_flatten = True
        elif ltype == "dense":
            if saw_conv and not saw_flatten:
                return ValidationResult(False, f"dense layer at index {i} appears before flatten")

    _, h, w, reason = _simulate_spatial_dims(layers, tuple(genome.get("input_shape", (3, 32, 32))))
    if reason:
        return ValidationResult(False, reason)

    if genome.get("optimizer") not in OPTIMIZERS:
        return ValidationResult(False, f"invalid optimizer: {genome.get('optimizer')}")
    lr = genome.get("learning_rate")
    if lr is None or lr <= 0:
        return ValidationResult(False, f"invalid learning_rate: {lr}")
    if genome.get("batch_size") not in BATCH_SIZES:
        return ValidationResult(False, f"invalid batch_size: {genome.get('batch_size')}")
    dropout = genome.get("dropout")
    if dropout is None or not (0.0 <= dropout < 1.0):
        return ValidationResult(False, f"invalid dropout: {dropout}")

    return ValidationResult(True)
