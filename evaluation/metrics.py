"""Parameter counting, latency measurement, and fitness computation."""

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency(
    model: nn.Module,
    input_shape: tuple,
    batch_size: int = 1,
    num_runs: int = 50,
    warmup: int = 10,
    device: str = "cpu",
) -> float:
    """Mean seconds per forward pass, averaged over num_runs after warmup."""
    model = model.to(device)
    model.eval()
    x = torch.randn(batch_size, *input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed / num_runs


@dataclass
class FitnessConfig:
    lambda_params: float = 0.1
    lambda_latency: float = 0.1
    param_reference: int = 2_000_000
    latency_reference: float = 0.05


def normalized_param_count(n_params: int, cfg: FitnessConfig) -> float:
    return min(n_params / cfg.param_reference, 1.0)


def normalized_latency(latency_s: float, cfg: FitnessConfig) -> float:
    return min(latency_s / cfg.latency_reference, 1.0)


def compute_fitness(accuracy: float, n_params: int, latency_s: float, cfg: FitnessConfig) -> float:
    return (
        accuracy
        - cfg.lambda_params * normalized_param_count(n_params, cfg)
        - cfg.lambda_latency * normalized_latency(latency_s, cfg)
    )
