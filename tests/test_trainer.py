import random

from genomes.schema import random_genome, validate_genome
from models.builder import build_model
from training.trainer import get_cifar10_loaders, train_and_evaluate, TrainConfig


def _tiny_valid_genome():
    rng = random.Random(0)
    for _ in range(50):
        genome = random_genome(rng, max_conv_layers=2, max_dense_layers=1)
        if validate_genome(genome).valid:
            return genome
    raise RuntimeError("could not find valid genome")


def test_cifar10_loader_respects_max_samples_subset_size():
    train_loader, val_loader = get_cifar10_loaders(
        genome_batch_size=32, max_train_samples=50, max_val_samples=20, num_workers=0
    )
    n_train = sum(len(batch[0]) for batch in train_loader)
    n_val = sum(len(batch[0]) for batch in val_loader)
    assert n_train == 50
    assert n_val == 20


def test_cifar10_loader_subset_is_deterministic_across_calls():
    t1, _ = get_cifar10_loaders(genome_batch_size=32, max_train_samples=10, num_workers=0)
    t2, _ = get_cifar10_loaders(genome_batch_size=32, max_train_samples=10, num_workers=0)
    idx1 = t1.dataset.indices
    idx2 = t2.dataset.indices
    assert list(idx1) == list(idx2)


def test_train_and_evaluate_returns_accuracy_in_valid_range():
    genome = _tiny_valid_genome()
    model = build_model(genome)
    cfg = TrainConfig(epochs=1, device="cpu", max_train_samples=100, max_val_samples=50, num_workers=0)
    acc = train_and_evaluate(model, genome, cfg)
    assert 0.0 <= acc <= 1.0
