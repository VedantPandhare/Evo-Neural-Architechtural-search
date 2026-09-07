import random

from genomes.operators import genome_to_build_spec, genome_to_json, genome_from_json, genome_hash
from genomes.schema import random_genome


def test_build_spec_appends_output_dense_layer():
    genome = random_genome(random.Random(0))
    spec = genome_to_build_spec(genome)
    last = spec.layers[-1]
    assert last["type"] == "dense"
    assert last["units"] == genome["num_classes"]
    assert last["activation"] == "none"


def test_build_spec_inserts_flatten_if_missing():
    genome = random_genome(random.Random(0))
    genome["layers"] = [l for l in genome["layers"] if l["type"] != "flatten"]
    genome["layers"].insert(0, {"type": "conv2d", "filters": 8, "kernel_size": 3, "stride": 1,
                                 "padding": "same", "activation": "relu", "batchnorm": False})
    spec = genome_to_build_spec(genome)
    types = [l["type"] for l in spec.layers]
    assert "flatten" in types


def test_build_spec_preserves_original_layers_order_before_output():
    genome = random_genome(random.Random(1))
    spec = genome_to_build_spec(genome)
    assert spec.layers[:len(genome["layers"])] == genome["layers"]


def test_genome_json_roundtrip_preserves_equality():
    genome = random_genome(random.Random(2))
    s = genome_to_json(genome)
    restored = genome_from_json(s)
    assert restored == genome


def test_genome_hash_stable_and_sensitive_to_change():
    genome = random_genome(random.Random(3))
    h1 = genome_hash(genome)
    h2 = genome_hash(genome)
    assert h1 == h2
    genome2 = random_genome(random.Random(4))
    assert genome_hash(genome2) != h1
