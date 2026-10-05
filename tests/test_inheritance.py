import copy

import torch

from evolution.inheritance import (
    build_offspring, extract_units, inherit_weights, kaiming_init, match_layers,
)
from models.builder import build_model


def conv(filters, k=3, bn=False):
    return {"type": "conv2d", "filters": filters, "kernel_size": k, "stride": 1,
            "padding": "same", "activation": "relu", "batchnorm": bn}


def dense(units, bn=False):
    return {"type": "dense", "units": units, "activation": "relu", "batchnorm": bn}


POOL = {"type": "maxpool", "kernel_size": 2, "stride": 2}
FLAT = {"type": "flatten"}


def genome(layers):
    return {"layers": layers, "optimizer": "adam", "learning_rate": 1e-3, "batch_size": 32,
            "dropout": 0.0, "input_shape": [3, 32, 32], "num_classes": 10}


BASE = genome([conv(8), conv(16), POOL, FLAT, dense(32)])


def model_of(g, seed=0):
    torch.manual_seed(seed)
    m = build_model(g)
    kaiming_init(m)
    return m


def statuses(parent_g, child_g):
    return match_layers(extract_units(model_of(parent_g)), extract_units(model_of(child_g)))


def test_identical_genome_matches_every_unit_exactly_and_reproduces_outputs():
    parent = model_of(BASE, seed=1)
    child, report = build_offspring(copy.deepcopy(BASE), [parent])
    assert [s for _, _, s in statuses(BASE, BASE)] == ["exact"] * 4  # 2 conv + dense + output
    assert report.n_exact == 4 and report.n_partial == 0 and report.n_fresh == 0
    assert report.fraction == 1.0
    parent.eval(); child.eval()
    x = torch.randn(4, 3, 32, 32)
    assert torch.allclose(parent(x), child(x))


def test_inserted_layer_is_fresh_and_later_layers_still_match_correctly():
    child_g = genome([conv(8), conv(24), conv(16), POOL, FLAT, dense(32)])  # new conv(24) inserted at index 1
    pairs = statuses(BASE, child_g)
    # parent units: 0=conv8 1=conv16 2=dense32 3=out; child units: 0=conv8 1=conv24 2=conv16 3=dense32 4=out
    assert (0, 0, "exact") in pairs
    # conv16 follows the inserted layer; it is only "partial" because its input channels went 8 -> 24
    assert (2, 1, "partial") in pairs
    assert (3, 2, "exact") in pairs and (4, 3, "exact") in pairs  # dense and output stay aligned
    assert all(cj != 1 for cj, _, _ in pairs)  # the inserted layer is not matched to anything


def test_removed_layer_leaves_others_aligned():
    child_g = genome([conv(8), POOL, FLAT, dense(32)])  # conv16 removed
    pairs = statuses(BASE, child_g)
    assert (0, 0, "exact") in pairs
    # child units: 0=conv8 1=dense32 2=out; parent units: 0=conv8 1=conv16 2=dense32 3=out
    assert (1, 2, "partial") in pairs  # dense input width halved, same spatial size
    assert (2, 3, "exact") in pairs  # output layer maps to output layer


def test_kernel_size_change_is_not_matched():
    child_g = genome([conv(8, k=5), conv(16), POOL, FLAT, dense(32)])
    pairs = statuses(BASE, child_g)
    assert all(cj != 0 for cj, _, _ in pairs)
    assert (1, 1, "exact") in pairs


def test_filter_change_copies_overlapping_slice_and_keeps_fresh_init_elsewhere():
    child_g = genome([conv(8), conv(24), POOL, FLAT, dense(32)])
    parent = model_of(BASE, seed=1)
    child = model_of(child_g, seed=2)
    fresh = copy.deepcopy(child)
    report = inherit_weights(child, [parent])
    pw = extract_units(parent)[1].layer.weight
    cw = extract_units(child)[1].layer.weight
    fw = extract_units(fresh)[1].layer.weight
    assert torch.equal(cw[:16], pw)  # the 16 shared filters are inherited
    assert torch.equal(cw[16:], fw[16:])  # the 8 new filters keep fresh init
    assert report.n_partial >= 1 and 0.0 < report.fraction < 1.0


def test_first_dense_with_scrambled_flatten_stays_fresh():
    child_g = genome([conv(8), conv(16), POOL, POOL, FLAT, dense(32)])  # spatial 16x16 -> 8x8
    pairs = statuses(BASE, child_g)
    first_dense_child = 2
    assert all(cj != first_dense_child for cj, _, _ in pairs)


def test_first_dense_with_only_channel_change_inherits_prefix():
    child_g = genome([conv(8), conv(24), POOL, FLAT, dense(32)])  # same spatial size, more channels
    pairs = statuses(BASE, child_g)
    assert (2, 2, "partial") in pairs


def test_no_parents_is_pure_scratch():
    _, report = build_offspring(copy.deepcopy(BASE), None)
    assert report.fraction == 0.0 and report.n_fresh == 4


def test_secondary_parent_only_fills_units_primary_could_not_match():
    primary = model_of(genome([conv(8), POOL, FLAT, dense(32)]), seed=1)  # no conv16
    secondary = model_of(BASE, seed=2)
    child = model_of(BASE, seed=3)
    report = inherit_weights(child, [primary, secondary])
    cu, pu, su = extract_units(child), extract_units(primary), extract_units(secondary)
    assert torch.equal(cu[0].layer.weight, pu[0].layer.weight)  # conv8: primary wins
    assert torch.equal(cu[1].layer.weight, su[1].layer.weight)  # conv16: only secondary has it
    assert {rank for _, rank, _, _ in report.matches} == {0, 1}


def test_inheritance_does_not_modify_parent():
    parent = model_of(BASE, seed=1)
    before = copy.deepcopy(parent.state_dict())
    build_offspring(genome([conv(8), conv(32), POOL, FLAT, dense(64)]), [parent])
    for k, v in parent.state_dict().items():
        assert torch.equal(v, before[k])


def test_batchnorm_stats_inherited_when_both_have_bn():
    pg = genome([conv(8, bn=True), POOL, FLAT, dense(16)])
    cg = genome([conv(12, bn=True), POOL, FLAT, dense(16)])
    parent = model_of(pg, seed=1)
    bn = [m for m in parent.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    bn.running_mean.fill_(0.5)
    bn.weight.data.fill_(2.0)
    child, _ = build_offspring(cg, [parent])
    cbn = [m for m in child.modules() if isinstance(m, torch.nn.BatchNorm2d)][0]
    assert torch.all(cbn.running_mean[:8] == 0.5) and torch.all(cbn.running_mean[8:] == 0.0)
    assert torch.all(cbn.weight[:8] == 2.0) and torch.all(cbn.weight[8:] == 1.0)


def test_kaiming_init_is_seed_deterministic_and_zeroes_bias():
    a, b = model_of(BASE, seed=7), model_of(BASE, seed=7)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert all(torch.all(m.bias == 0) for m in a.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
