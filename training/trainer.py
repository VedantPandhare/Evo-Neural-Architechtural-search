"""Fixed-budget training and CIFAR-10 data loading."""

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class TrainConfig:
    epochs: int
    device: str = "cpu"
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    num_workers: int = 0
    data_dir: str = "./data"
    augment: bool = True
    subset_seed: int = 42
    val_size: int = 5000
    use_test_set: bool = False


def _subset_indices(dataset_len: int, n: int, seed: int) -> list:
    rng = random.Random(seed)
    perm = list(range(dataset_len))
    rng.shuffle(perm)
    return perm[:n]


def get_cifar10_loaders(
    genome_batch_size: int,
    data_dir: str = "./data",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    num_workers: int = 0,
    augment: bool = True,
    subset_seed: int = 42,
    val_size: int = 5000,
    split_seed: int = 0,
    use_test_set: bool = False,
) -> tuple:
    """By default the second loader is a validation split held out of the 50k training
    images (used for fitness during search). Set use_test_set=True only for the final
    evaluation: it trains on all 50k and evaluates on the official 10k test set."""
    if augment and max_train_samples is None:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    val_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

    if use_test_set:
        val_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
    else:
        # same images, eval transform; the split is fixed by split_seed so it never changes between genomes
        val_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=val_transform)
        perm = _subset_indices(len(train_set), len(train_set), split_seed)
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        train_set = Subset(train_set, train_idx)
        val_set = Subset(val_set, val_idx)

    if max_train_samples is not None:
        idx = _subset_indices(len(train_set), max_train_samples, subset_seed)
        train_set = Subset(train_set, idx)
    if max_val_samples is not None:
        idx = _subset_indices(len(val_set), max_val_samples, subset_seed)
        val_set = Subset(val_set, idx)

    train_loader = DataLoader(train_set, batch_size=genome_batch_size, shuffle=True, num_workers=num_workers,
                               pin_memory=(num_workers > 0))
    val_loader = DataLoader(val_set, batch_size=genome_batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=(num_workers > 0))
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train_and_evaluate(model: nn.Module, genome: dict, cfg: TrainConfig) -> float:
    from models.builder import make_optimizer

    model = model.to(cfg.device)
    optimizer = make_optimizer(model, genome)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = get_cifar10_loaders(
        genome_batch_size=genome["batch_size"],
        data_dir=cfg.data_dir,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        num_workers=cfg.num_workers,
        augment=cfg.augment,
        subset_seed=cfg.subset_seed,
        val_size=cfg.val_size,
        use_test_set=cfg.use_test_set,
    )

    for _ in range(cfg.epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)

    return evaluate_accuracy(model, val_loader, cfg.device)
