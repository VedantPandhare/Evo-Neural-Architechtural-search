import random

import torch

from genomes.schema import random_genome, validate_genome
from models.builder import build_model
from training.trainer import (
    GPULoader, TrainConfig, get_cifar10_loaders, get_gpu_loaders, train_and_evaluate,
)


def _all(loader):
    xs, ys = zip(*list(loader))
    return torch.cat(xs), torch.cat(ys)


def _sorted_by_sum(x, y):
    order = torch.argsort(x.flatten(1).sum(1))
    return x[order], y[order]


def test_val_split_matches_cpu_loader_exactly():
    _, cpu_v = get_cifar10_loaders(32, max_train_samples=200, max_val_samples=100, augment=False)
    _, gpu_v = get_gpu_loaders(32, "cpu", max_train_samples=200, max_val_samples=100, augment=False)
    cx, cy = _all(cpu_v)
    gx, gy = _all(gpu_v)
    assert torch.equal(cy, gy) and torch.allclose(cx, gx, atol=1e-5)  # val is unshuffled -> same order


def test_train_subset_has_same_images_as_cpu_loader():
    cpu_t, _ = get_cifar10_loaders(32, max_train_samples=200, augment=False)
    gpu_t, _ = get_gpu_loaders(32, "cpu", max_train_samples=200, augment=False)
    cx, cy = _sorted_by_sum(*_all(cpu_t))
    gx, gy = _sorted_by_sum(*_all(gpu_t))
    assert torch.equal(cy, gy) and torch.allclose(cx, gx, atol=1e-5)


def test_default_split_sizes_and_test_set_mode():
    t, v = get_gpu_loaders(256, "cpu")
    assert len(t.x) == 45000 and len(v.x) == 5000
    t2, v2 = get_gpu_loaders(256, "cpu", use_test_set=True)
    assert len(t2.x) == 50000 and len(v2.x) == 10000


def test_batches_cover_every_sample_once_with_partial_last_batch():
    t, _ = get_gpu_loaders(64, "cpu", max_train_samples=150, augment=False)
    assert [len(y) for _, y in t] == [64, 64, 22] and len(t) == 3


def test_augmentation_keeps_shape_changes_pixels_and_preserves_labels():
    x = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.uint8)
    y = torch.arange(64)
    px, _ = next(iter(GPULoader(x, y, 64, shuffle=False, augment=False)))
    ax, ay = next(iter(GPULoader(x, y, 64, shuffle=False, augment=True)))
    assert ax.shape == px.shape == (64, 3, 32, 32)
    assert not torch.allclose(ax, px)
    assert torch.equal(ay, y)


def test_augmentation_is_a_pure_shift_or_flip_of_the_image():
    # an image with a single bright pixel must still contain exactly that one pixel (or lose it off the edge)
    x = torch.zeros(256, 3, 32, 32, dtype=torch.uint8)
    x[:, :, 16, 16] = 255
    loader = GPULoader(x, torch.zeros(256, dtype=torch.long), 256, shuffle=False, augment=True)
    ax, _ = next(iter(loader))
    raw = ax * loader.std + loader.mean  # undo normalisation
    bright = (raw > 0.99).flatten(1).sum(1)
    assert bright.max().item() == 3 * 1 and bright.min().item() == 3  # one pixel x 3 channels, never duplicated


def test_gpu_path_trains_and_returns_valid_accuracy():
    rng = random.Random(0)
    genome = next(g for g in (random_genome(rng, max_conv_layers=2, max_dense_layers=1) for _ in range(50))
                  if validate_genome(g).valid)
    cfg = TrainConfig(epochs=1, device="cpu", max_train_samples=100, max_val_samples=50, gpu_data=True)
    assert 0.0 <= train_and_evaluate(build_model(genome), genome, cfg) <= 1.0
    cfg_amp = TrainConfig(epochs=1, device="cpu", max_train_samples=100, max_val_samples=50, gpu_data=True, amp=True)
    assert 0.0 <= train_and_evaluate(build_model(genome), genome, cfg_amp) <= 1.0
