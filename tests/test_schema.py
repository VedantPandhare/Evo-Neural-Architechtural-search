import random

from genomes.schema import random_genome, validate_genome


def test_fifty_random_genomes_validate():
    rng = random.Random(0)
    for _ in range(50):
        genome = random_genome(rng)
        result = validate_genome(genome)
        assert result.valid, result.reason


def test_validate_rejects_dense_immediately_after_conv_without_flatten():
    genome = random_genome(random.Random(1))
    genome["layers"] = [
        {"type": "conv2d", "filters": 16, "kernel_size": 3, "stride": 1, "padding": "same",
         "activation": "relu", "batchnorm": False},
        {"type": "dense", "units": 64, "activation": "relu", "batchnorm": False},
    ]
    result = validate_genome(genome)
    assert not result.valid


def test_validate_rejects_conv_after_dense():
    genome = random_genome(random.Random(2))
    genome["layers"] = [
        {"type": "dense", "units": 64, "activation": "relu", "batchnorm": False},
        {"type": "conv2d", "filters": 16, "kernel_size": 3, "stride": 1, "padding": "same",
         "activation": "relu", "batchnorm": False},
    ]
    result = validate_genome(genome)
    assert not result.valid


def test_validate_rejects_out_of_range_dropout():
    genome = random_genome(random.Random(3))
    genome["dropout"] = 1.5
    result = validate_genome(genome)
    assert not result.valid


def test_validate_rejects_non_positive_learning_rate():
    genome = random_genome(random.Random(4))
    genome["learning_rate"] = 0.0
    result = validate_genome(genome)
    assert not result.valid


def test_validate_rejects_bad_batch_size():
    genome = random_genome(random.Random(5))
    genome["batch_size"] = 17
    result = validate_genome(genome)
    assert not result.valid


def test_validate_rejects_shape_collapsing_pool_stack():
    genome = random_genome(random.Random(6))
    genome["input_shape"] = [3, 32, 32]
    genome["layers"] = [
        {"type": "maxpool", "kernel_size": 4, "stride": 4},
        {"type": "maxpool", "kernel_size": 4, "stride": 4},
        {"type": "maxpool", "kernel_size": 4, "stride": 4},
        {"type": "flatten"},
        {"type": "dense", "units": 64, "activation": "relu", "batchnorm": False},
    ]
    result = validate_genome(genome)
    assert not result.valid
