"""Genome <-> build spec conversion and (de)serialization."""

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class BuildSpec:
    layers: list
    optimizer: str
    learning_rate: float
    batch_size: int
    dropout: float
    input_shape: tuple
    num_classes: int


def genome_to_build_spec(genome: dict) -> BuildSpec:
    """Assumes genome already passed validate_genome(). Inserts a flatten layer at the
    conv->dense boundary if missing, and appends the output dense layer."""
    layers = [dict(l) for l in genome["layers"]]

    has_conv = any(l["type"] in ("conv2d", "maxpool", "avgpool") for l in layers)
    has_flatten = any(l["type"] == "flatten" for l in layers)
    if has_conv and not has_flatten:
        # insert flatten right before the first dense/dropout layer, or at the end
        insert_at = len(layers)
        for i, l in enumerate(layers):
            if l["type"] in ("dense", "dropout"):
                insert_at = i
                break
        layers.insert(insert_at, {"type": "flatten"})

    layers.append({
        "type": "dense",
        "units": genome["num_classes"],
        "activation": "none",
        "batchnorm": False,
    })

    return BuildSpec(
        layers=layers,
        optimizer=genome["optimizer"],
        learning_rate=genome["learning_rate"],
        batch_size=genome["batch_size"],
        dropout=genome["dropout"],
        input_shape=tuple(genome["input_shape"]),
        num_classes=genome["num_classes"],
    )


def genome_to_json(genome: dict) -> str:
    return json.dumps(genome, sort_keys=True)


def genome_from_json(s: str) -> dict:
    return json.loads(s)


def genome_hash(genome: dict) -> str:
    canonical = json.dumps(genome, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
