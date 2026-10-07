"""Fixed-budget training and CIFAR-10 data loading."""

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    gpu_data: bool = False  # keep CIFAR-10 on the device and augment there (much faster on a GPU)
    amp: bool = False  # bfloat16 autocast; changes numerics slightly, so keep it fixed within an experiment


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


_GPU_CACHE: dict = {}


def _cifar_uint8(data_dir: str, train: bool, device: str) -> tuple:
    """CIFAR-10 as uint8 NCHW tensors resident on `device`, loaded once per process."""
    key = (data_dir, train, device)
    if key not in _GPU_CACHE:
        ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous().to(device)
        y = torch.tensor(ds.targets, dtype=torch.long, device=device)
        _GPU_CACHE[key] = (x, y)
    return _GPU_CACHE[key]


def _split_indices(n_train_total: int, n_test_total: int, max_train_samples, max_val_samples,
                   subset_seed: int, val_size: int, split_seed: int, use_test_set: bool) -> tuple:
    """Same split and subsetting rules as get_cifar10_loaders, as plain index lists."""
    if use_test_set:
        train_pool, val_pool = list(range(n_train_total)), list(range(n_test_total))
    else:
        perm = _subset_indices(n_train_total, n_train_total, split_seed)
        val_pool, train_pool = perm[:val_size], perm[val_size:]
    if max_train_samples is not None:
        train_pool = [train_pool[i] for i in _subset_indices(len(train_pool), max_train_samples, subset_seed)]
    if max_val_samples is not None:
        val_pool = [val_pool[i] for i in _subset_indices(len(val_pool), max_val_samples, subset_seed)]
    return train_pool, val_pool


class GPULoader:
    """Iterates device-resident uint8 images; normalises (and augments) each batch on the device."""

    def __init__(self, x_uint8, y, batch_size: int, shuffle: bool, augment: bool):
        self.x, self.y, self.batch_size, self.shuffle, self.augment = x_uint8, y, batch_size, shuffle, augment
        dev = x_uint8.device
        self.mean = torch.tensor(CIFAR10_MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, device=dev).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return (len(self.x) + self.batch_size - 1) // self.batch_size

    def _augment(self, xb: torch.Tensor) -> torch.Tensor:
        """RandomCrop(32, padding=4) + horizontal flip, per sample, on [0,1] images."""
        b, dev = xb.shape[0], xb.device
        padded = F.pad(xb, (4, 4, 4, 4))
        i = torch.randint(0, 9, (b,), device=dev)
        j = torch.randint(0, 9, (b,), device=dev)
        ar = torch.arange(32, device=dev)
        rows = (i[:, None] + ar)[:, :, None]
        cols = (j[:, None] + ar)[:, None, :]
        out = padded[torch.arange(b, device=dev)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(b, device=dev) < 0.5
        return torch.where(flip[:, None, None, None], out.flip(3), out)

    def __iter__(self):
        n = len(self.x)
        order = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for s in range(0, n, self.batch_size):
            idx = order[s:s + self.batch_size]
            xb = self.x[idx].float() / 255.0
            if self.augment:
                xb = self._augment(xb)
            yield (xb - self.mean) / self.std, self.y[idx]


def get_gpu_loaders(genome_batch_size: int, device: str, data_dir: str = "./data",
                    max_train_samples=None, max_val_samples=None, augment: bool = True,
                    subset_seed: int = 42, val_size: int = 5000, split_seed: int = 0,
                    use_test_set: bool = False) -> tuple:
    """Drop-in for get_cifar10_loaders with identical splits, but everything lives on `device`."""
    x_tr, y_tr = _cifar_uint8(data_dir, True, device)
    x_te, y_te = _cifar_uint8(data_dir, False, device) if use_test_set else (None, None)
    train_idx, val_idx = _split_indices(len(x_tr), 10000, max_train_samples, max_val_samples,
                                        subset_seed, val_size, split_seed, use_test_set)
    ti = torch.tensor(train_idx, device=device)
    vi = torch.tensor(val_idx, device=device)
    vx, vy = (x_te[vi], y_te[vi]) if use_test_set else (x_tr[vi], y_tr[vi])
    # same rule as the CPU path: no augmentation when training on a capped subset
    do_aug = augment and max_train_samples is None
    train = GPULoader(x_tr[ti], y_tr[ti], genome_batch_size, shuffle=True, augment=do_aug)
    val = GPULoader(vx, vy, genome_batch_size, shuffle=False, augment=False)
    return train, val


def _autocast(device: str, amp: bool):
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=amp)


def train_one_epoch(model, loader, optimizer, criterion, device, amp: bool = False) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with _autocast(device, amp):
            out = model(x)
            loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def evaluate_accuracy(model, loader, device, amp: bool = False) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with _autocast(device, amp):
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

    common = dict(
        data_dir=cfg.data_dir,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
        augment=cfg.augment,
        subset_seed=cfg.subset_seed,
        val_size=cfg.val_size,
        use_test_set=cfg.use_test_set,
    )
    if cfg.gpu_data:
        train_loader, val_loader = get_gpu_loaders(genome["batch_size"], cfg.device, **common)
    else:
        train_loader, val_loader = get_cifar10_loaders(genome["batch_size"], num_workers=cfg.num_workers, **common)

    for _ in range(cfg.epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, cfg.device, cfg.amp)

    return evaluate_accuracy(model, val_loader, cfg.device, cfg.amp)
