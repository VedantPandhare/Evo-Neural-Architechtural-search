from evaluation.metrics import evaluate_genome, FitnessConfig, EvalResult
from training.trainer import TrainConfig

import torch

torch.manual_seed(0)

TRAIN_CFG = TrainConfig(epochs=1, device="cpu", max_train_samples=500, max_val_samples=200, num_workers=0)


def _conv_genome(filters, dense_units, n_conv_layers=1):
    layers = []
    for _ in range(n_conv_layers):
        layers.append({"type": "conv2d", "filters": filters, "kernel_size": 3, "stride": 1,
                        "padding": "same", "activation": "relu", "batchnorm": False})
    layers.append({"type": "maxpool", "kernel_size": 2, "stride": 2})
    layers.append({"type": "flatten"})
    layers.append({"type": "dense", "units": dense_units, "activation": "relu", "batchnorm": False})
    return {
        "layers": layers,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "dropout": 0.0,
        "input_shape": [3, 32, 32],
        "num_classes": 10,
    }


def test_fitness_ranks_five_handpicked_genomes_sensibly():
    genomes = {
        "tiny": _conv_genome(filters=4, dense_units=8, n_conv_layers=1),
        "medium": _conv_genome(filters=16, dense_units=32, n_conv_layers=2),
        "deeper": _conv_genome(filters=16, dense_units=64, n_conv_layers=4),
        "oversized": _conv_genome(filters=128, dense_units=512, n_conv_layers=4),
        "cheap_bad": _conv_genome(filters=2, dense_units=4, n_conv_layers=1),
    }

    fitness_cfg = FitnessConfig(lambda_params=0.2, lambda_latency=0.1,
                                 param_reference=200_000, latency_reference=0.02)

    results: dict[str, EvalResult] = {}
    for name, genome in genomes.items():
        results[name] = evaluate_genome(genome, TRAIN_CFG, fitness_cfg)

    assert results["oversized"].n_params > results["tiny"].n_params
    # penalty makes fitness strictly less than raw accuracy for the big model
    assert results["oversized"].fitness < results["oversized"].accuracy - 1e-9
    # ranking by fitness must not be identical to ranking by raw accuracy alone
    by_fitness = sorted(results, key=lambda n: results[n].fitness, reverse=True)
    by_accuracy = sorted(results, key=lambda n: results[n].accuracy, reverse=True)
    assert by_fitness != by_accuracy or results["oversized"].fitness < results["oversized"].accuracy
