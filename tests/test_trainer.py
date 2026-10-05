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


def _root_indices(subset):
    """Resolve nested Subsets to indices into the underlying 50k training set."""
    idx = list(range(len(subset.dataset))) if not hasattr(subset.dataset, "indices") else list(subset.dataset.indices)
    return [idx[i] for i in subset.indices] if hasattr(subset.dataset, "indices") else list(subset.indices)


def test_val_split_is_disjoint_from_train_and_never_uses_test_set():
    train_loader, val_loader = get_cifar10_loaders(genome_batch_size=32, num_workers=0, augment=False)
    assert len(train_loader.dataset) == 45000
    assert len(val_loader.dataset) == 5000
    assert not set(train_loader.dataset.indices) & set(val_loader.dataset.indices)
    assert len(val_loader.dataset.dataset) == 50000  # drawn from the training files, not the 10k test set


def test_use_test_set_evaluates_on_official_test_set():
    train_loader, val_loader = get_cifar10_loaders(genome_batch_size=32, num_workers=0, use_test_set=True)
    assert len(val_loader.dataset) == 10000
    assert len(train_loader.dataset) == 50000


def test_subsets_stay_inside_their_split():
    train_loader, val_loader = get_cifar10_loaders(
        genome_batch_size=32, max_train_samples=100, max_val_samples=50, num_workers=0)
    assert not set(_root_indices(train_loader.dataset)) & set(_root_indices(val_loader.dataset))
