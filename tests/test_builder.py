import random

import torch

from genomes.schema import random_genome, validate_genome
from models.builder import build_model, make_optimizer


def test_twenty_random_valid_genomes_build_and_forward_without_shape_errors():
    rng = random.Random(1)
    built = 0
    attempts = 0
    while built < 20 and attempts < 200:
        attempts += 1
        genome = random_genome(rng)
        if not validate_genome(genome).valid:
            continue
        model = build_model(genome)
        x = torch.randn(4, *genome["input_shape"])
        out = model(x)
        assert out.shape == (4, genome["num_classes"])
        built += 1
    assert built == 20


def test_build_model_raises_on_invalid_genome():
    genome = random_genome(random.Random(2))
    genome["dropout"] = 5.0
    try:
        build_model(genome)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_batchnorm_and_dropout_layers_present_when_specified():
    genome = {
        "layers": [
            {"type": "conv2d", "filters": 8, "kernel_size": 3, "stride": 1, "padding": "same",
             "activation": "relu", "batchnorm": True},
            {"type": "flatten"},
            {"type": "dropout", "rate": 0.3},
            {"type": "dense", "units": 16, "activation": "relu", "batchnorm": False},
        ],
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "dropout": 0.2,
        "input_shape": [3, 16, 16],
        "num_classes": 10,
    }
    model = build_model(genome)
    modules = list(model.modules())
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in modules)
    assert any(isinstance(m, torch.nn.Dropout) for m in modules)


def test_make_optimizer_returns_correct_type():
    genome = random_genome(random.Random(3))
    model = build_model(genome)
    opt = make_optimizer(model, genome)
    assert isinstance(opt, torch.optim.Optimizer)
