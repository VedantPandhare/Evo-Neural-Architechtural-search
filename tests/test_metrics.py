import torch
import torch.nn as nn

from evaluation.metrics import (
    count_parameters,
    measure_latency,
    compute_fitness,
    FitnessConfig,
    normalized_param_count,
    normalized_latency,
)


def test_count_parameters_matches_manual_sum():
    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
    expected = sum(p.numel() for p in model.parameters())
    assert count_parameters(model) == expected


def test_measure_latency_returns_positive_float():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    latency = measure_latency(model, input_shape=(3, 8, 8), num_runs=5, warmup=2, device="cpu")
    assert latency > 0


def test_normalized_param_count_capped_at_one():
    cfg = FitnessConfig(param_reference=1000)
    assert normalized_param_count(500, cfg) == 0.5
    assert normalized_param_count(5000, cfg) == 1.0


def test_normalized_latency_capped_at_one():
    cfg = FitnessConfig(latency_reference=0.1)
    assert normalized_latency(0.05, cfg) == 0.5
    assert normalized_latency(1.0, cfg) == 1.0


def test_compute_fitness_penalizes_large_and_slow_models():
    cfg = FitnessConfig(lambda_params=0.1, lambda_latency=0.1, param_reference=1000, latency_reference=1.0)
    fitness_small = compute_fitness(accuracy=0.8, n_params=100, latency_s=0.1, cfg=cfg)
    fitness_large = compute_fitness(accuracy=0.8, n_params=5000, latency_s=2.0, cfg=cfg)
    assert fitness_small > fitness_large
    assert fitness_large == 0.8 - 0.1 * 1.0 - 0.1 * 1.0
