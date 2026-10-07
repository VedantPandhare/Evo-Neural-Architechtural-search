import os

from evaluation.aggregate import arm_summary, format_report, load_run, reach, summarize, time_by_size
from evaluation.metrics import EvalResult
from evaluation.run_logger import RunLogger
from evolution.population import init_population
import random

POP = init_population(3, random.Random(0))


def _make_run(out_dir, name, bests, hours_per_gen=0.01, n_params=5e5, train_s=1.0, inherit=0.0):
    log = RunLogger(os.path.join(out_dir, name))
    for gen, b in enumerate(bests):
        log.log_candidate(gen, f"h{gen}", POP[0], EvalResult(0.6, int(n_params), 0.01, b), train_s,
                          inheritance=None if gen == 0 else type("R", (), dict(
                              fraction=inherit, n_exact=1, n_partial=0, n_fresh=0))())
        row = log.log_generation(gen, POP, [b, b - 0.1, b - 0.2], 1,
                                 [{"inherit_fraction": inherit if gen else 0.0}])
    # make cumulative time deterministic for assertions
    import csv
    path = os.path.join(out_dir, name, "generations.csv")
    rows = list(csv.DictReader(open(path)))
    for i, r in enumerate(rows):
        r["cumulative_seconds"] = (i + 1) * hours_per_gen * 3600
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_reach_first_generation_at_or_above_threshold(tmp_path):
    _make_run(str(tmp_path), "r", [0.5, 0.6, 0.7, 0.7], hours_per_gen=0.1)
    gens = load_run(str(tmp_path / "r"))["gens"]
    assert reach(gens, 0.6)[0] == 1
    g, h = reach(gens, 0.7)
    assert g == 2 and abs(h - 0.3) < 1e-9
    assert reach(gens, 0.9) == (None, None)


def test_arm_summary_fields(tmp_path):
    _make_run(str(tmp_path), "r", [0.5, 0.65, 0.7], inherit=0.8)
    s = arm_summary(load_run(str(tmp_path / "r")), (0.65, 0.9))
    assert s["final_best"] == 0.7 and abs(s["mean_inherit_fraction"] - 0.8) < 1e-9
    assert s["gen_to_0.65"] == 1 and s["gen_to_0.9"] is None and s["hours_to_0.9"] is None


def test_summarize_paired_differences_and_wins(tmp_path):
    out = str(tmp_path)
    _make_run(out, "search_inherit_seed0", [0.5, 0.7, 0.75])
    _make_run(out, "search_scratch_seed0", [0.5, 0.6, 0.70])
    _make_run(out, "search_inherit_seed1", [0.5, 0.6, 0.65])
    _make_run(out, "search_scratch_seed1", [0.5, 0.62, 0.70])
    r = summarize(out, [0, 1, 2])
    assert r["missing"] == [2] and set(r["per_seed"]) == {0, 1}
    d = r["paired_diff_on_minus_off"]["final_best"]
    assert [round(v, 6) for v in d["values"]] == [0.05, -0.05] and abs(d["mean"]) < 1e-9
    assert r["seeds_where_on_better"] == "1/2"
    assert abs(r["mean_std"]["on"]["final_best"][0] - 0.70) < 1e-9


def test_timing_by_size_buckets_and_report_renders(tmp_path):
    out = str(tmp_path)
    _make_run(out, "search_inherit_seed0", [0.5, 0.6], n_params=5e5, train_s=2.0)
    _make_run(out, "search_scratch_seed0", [0.5, 0.6], n_params=5e5, train_s=4.0)
    t = time_by_size(load_run(os.path.join(out, "search_scratch_seed0")))
    assert t["1e5-1e6"]["n"] == 2 and t["1e5-1e6"]["mean_train_seconds"] == 4.0
    assert t["<1e5"]["n"] == 0 and t["<1e5"]["mean_train_seconds"] is None
    text = format_report(summarize(out, [0]))
    assert "2.00s (n=2)" in text and "4.00s (n=2)" in text and "paired ON-OFF" in text


def test_same_genome_timing_ratio_and_failure_rates(tmp_path):
    from evaluation.aggregate import failure_rates, same_genome_timing
    out = str(tmp_path)
    _make_run(out, "search_inherit_seed0", [0.5, 0.6], train_s=2.0)
    _make_run(out, "search_scratch_seed0", [0.5, 0.6], train_s=3.0)
    on, off = load_run(os.path.join(out, "search_inherit_seed0")), load_run(os.path.join(out, "search_scratch_seed0"))
    t = same_genome_timing(on, off)
    assert t["generation_0"]["n"] == 1 and abs(t["generation_0"]["median_ratio_off_over_on"] - 1.5) < 1e-9
    assert failure_rates(on)["n"] == 1 and failure_rates(on)["failure_rate"] == 0.0
    assert failure_rates({"cands": [{"generation": 0, "accuracy": 0.1}]})["n"] == 0
    text = format_report(summarize(out, [0]))
    assert "Same-genome timing" in text and "Bad offspring" in text
