"""Weight inheritance protocol (the MethodsX contribution).

Protocol, stated explicitly so it can be reported in the paper:

1. Units. A model is reduced to an ordered list of weight units: each Conv2d or Linear layer
   plus its BatchNorm (if any). Pooling, dropout and activations carry no weights.
2. Matching. For one parent, child units are aligned to parent units with a monotone
   (order-preserving) weighted alignment, so an inserted or removed layer does not shift every
   later match. Two units are compatible if they are the same kind and, for convolutions, have the
   same kernel size. Identical weight shapes ("exact") score 3, other compatible pairs ("partial") 1.
3. Transfer. Matched units copy the parent's weights *before* training. Exact matches copy
   everything. Partial matches (e.g. 32 -> 48 filters, or a different input width) copy the
   overlapping leading slice [:min_out, :min_in] and keep the fresh initialisation elsewhere.
   BatchNorm weight, bias and running statistics are copied the same way when both units have one.
   Exception: the first dense layer after flatten, when the flattened spatial size changed; the
   feature order is then scrambled, so the layer keeps its fresh initialisation.
4. Fresh initialisation. Every weight not inherited keeps Kaiming-normal initialisation
   (`kaiming_init`, fan-in, ReLU gain, zero bias). The from-scratch baseline MUST use the same
   function so the two arms differ only in inheritance.
5. Crossover. Parents are given in priority order. The primary parent is the one that supplied the
   child's conv prefix (parent A in `evolution.crossover`). It claims its matches first; the
   secondary parent only fills child units the primary could not match, and never overwrites.
6. Logging. `InheritanceReport` records, per child, the fraction of parameters inherited, and
   how many units were exact / partial / fresh.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

EXACT_SCORE = 3
PARTIAL_SCORE = 1


def kaiming_init(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@dataclass
class Unit:
    kind: str  # "conv" | "dense"
    layer: nn.Module
    bn: nn.Module | None = None
    first_dense: bool = False
    flat_hw: int | None = None  # spatial size of the flattened input, first dense only

    @property
    def out_dim(self) -> int:
        return self.layer.weight.shape[0]

    @property
    def in_dim(self) -> int:
        return self.layer.weight.shape[1]

    @property
    def kernel(self):
        return self.layer.kernel_size if self.kind == "conv" else None


def extract_units(model: nn.Module) -> list:
    units: list = []
    last_conv_out = None
    seen_dense = False
    for layer in list(model.feature_layers) + list(model.classifier_layers):
        if isinstance(layer, nn.Conv2d):
            units.append(Unit("conv", layer))
            last_conv_out = layer.out_channels
        elif isinstance(layer, nn.Linear):
            u = Unit("dense", layer)
            if not seen_dense:
                u.first_dense = True
                if last_conv_out:
                    u.flat_hw = layer.in_features // last_conv_out
                seen_dense = True
            units.append(u)
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)) and units:
            units[-1].bn = layer
    return units


def _is_exact(p: Unit, c: Unit) -> bool:
    return (p.layer.weight.shape == c.layer.weight.shape
            and (p.bn is None) == (c.bn is None))


def _compatibility(p: Unit, c: Unit):
    """Returns 'exact', 'partial' or None."""
    if p.kind != c.kind or p.kernel != c.kernel:
        return None
    if _is_exact(p, c):
        return "exact"
    if p.first_dense and c.first_dense and p.in_dim != c.in_dim and p.flat_hw != c.flat_hw:
        return None  # flattened feature order would be scrambled
    return "partial"


def match_layers(parent_units: list, child_units: list) -> list:
    """Monotone weighted alignment. Returns [(child_idx, parent_idx, 'exact'|'partial'), ...]
    sorted by child_idx. Unmatched child units are simply absent."""
    n, m = len(parent_units), len(child_units)
    compat = [[_compatibility(parent_units[i], child_units[j]) for j in range(m)] for i in range(n)]
    score = {"exact": EXACT_SCORE, "partial": PARTIAL_SCORE}
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(dp[i - 1][j], dp[i][j - 1])
            c = compat[i - 1][j - 1]
            if c:
                best = max(best, dp[i - 1][j - 1] + score[c])
            dp[i][j] = best

    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        c = compat[i - 1][j - 1]
        if c and dp[i][j] == dp[i - 1][j - 1] + score[c]:
            pairs.append((j - 1, i - 1, c))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return sorted(pairs)


def _copy_slice(src: torch.Tensor, dst: torch.Tensor) -> int:
    """Copy the overlapping leading slice of src into dst; returns elements copied."""
    idx = tuple(slice(0, min(s, d)) for s, d in zip(src.shape, dst.shape))
    dst[idx] = src[idx]
    return src[idx].numel()


def _transfer(p: Unit, c: Unit) -> int:
    """Copy parent unit weights into child unit; returns number of *parameters* inherited."""
    n = _copy_slice(p.layer.weight.data, c.layer.weight.data)
    if p.layer.bias is not None and c.layer.bias is not None:
        n += _copy_slice(p.layer.bias.data, c.layer.bias.data)
    if p.bn is not None and c.bn is not None:
        n += _copy_slice(p.bn.weight.data, c.bn.weight.data)
        n += _copy_slice(p.bn.bias.data, c.bn.bias.data)
        _copy_slice(p.bn.running_mean, c.bn.running_mean)  # buffers: copied but not counted as parameters
        _copy_slice(p.bn.running_var, c.bn.running_var)
    return n


@dataclass
class InheritanceReport:
    inherited_params: int = 0
    total_params: int = 0
    n_exact: int = 0
    n_partial: int = 0
    n_fresh: int = 0
    matches: list = field(default_factory=list)  # (child_idx, parent_rank, parent_idx, status)

    @property
    def fraction(self) -> float:
        return self.inherited_params / self.total_params if self.total_params else 0.0


@torch.no_grad()
def inherit_weights(child: nn.Module, parents: list) -> InheritanceReport:
    """Copy weights into `child` from `parents` (priority order). `child` should already have been
    through `kaiming_init`. Parents are never modified."""
    child_units = extract_units(child)
    report = InheritanceReport(total_params=sum(p.numel() for p in child.parameters()))
    unfilled = list(range(len(child_units)))

    for rank, parent in enumerate(parents):
        if not unfilled:
            break
        parent_units = extract_units(parent)
        sub_child = [child_units[j] for j in unfilled]
        claimed = set()
        for sub_j, pi, status in match_layers(parent_units, sub_child):
            cj = unfilled[sub_j]
            report.inherited_params += _transfer(parent_units[pi], child_units[cj])
            report.matches.append((cj, rank, pi, status))
            if status == "exact":
                report.n_exact += 1
            else:
                report.n_partial += 1
            claimed.add(cj)
        unfilled = [j for j in unfilled if j not in claimed]

    report.n_fresh = len(unfilled)
    report.matches.sort()
    return report


def build_offspring(genome: dict, parents: list | None = None) -> tuple:
    """Build a model with the shared fresh-init, then inherit from `parents` (priority order).
    parents=None/[] gives the from-scratch baseline. Returns (model, InheritanceReport)."""
    from models.builder import build_model

    model = build_model(genome)
    kaiming_init(model)
    report = inherit_weights(model, parents or [])
    return model, report
