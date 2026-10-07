"""Cross-seed comparison of inheritance ON vs OFF, from the files RunLogger wrote.

Run directories are expected as <out_dir>/search_inherit_seed{S} and <out_dir>/search_scratch_seed{S}
(the names the notebook uses). Everything is plain Python so it works on any machine.
"""

import json
import math
import os
import statistics

ARMS = {"on": "search_inherit_seed{seed}", "off": "search_scratch_seed{seed}"}
PARAM_BUCKETS = [("<1e5", 0, 1e5), ("1e5-1e6", 1e5, 1e6), (">=1e6", 1e6, math.inf)]


def load_run(run_dir: str) -> dict:
    import csv
    with open(os.path.join(run_dir, "generations.csv"), newline="") as f:
        gens = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    with open(os.path.join(run_dir, "candidates.jsonl")) as f:
        cands = [json.loads(line) for line in f if line.strip()]
    return {"gens": gens, "cands": cands}


def reach(gens: list, threshold: float):
    """(generation, cumulative_hours) at which best fitness first >= threshold, or (None, None)."""
    for g in gens:
        if g["best"] >= threshold:
            return int(g["generation"]), g["cumulative_seconds"] / 3600
    return None, None


def arm_summary(run: dict, thresholds) -> dict:
    gens, cands = run["gens"], run["cands"]
    out = {
        "final_best": gens[-1]["best"],
        "final_mean": gens[-1]["mean"],
        "hours": gens[-1]["cumulative_seconds"] / 3600,
        "mean_inherit_fraction": statistics.fmean(c["inherit_fraction"] for c in cands if c["generation"] > 0)
        if any(c["generation"] > 0 for c in cands) else 0.0,
    }
    for t in thresholds:
        g, h = reach(gens, t)
        out[f"gen_to_{t}"], out[f"hours_to_{t}"] = g, h
    return out


def time_by_size(run: dict) -> dict:
    """Mean train_seconds per parameter-count bucket: lets you check whether two arms differ in
    speed at the *same* model size (environment effect) or only in which sizes they explored."""
    res = {}
    for name, lo, hi in PARAM_BUCKETS:
        xs = [c["train_seconds"] for c in run["cands"] if lo <= c["n_params"] < hi]
        res[name] = {"n": len(xs), "mean_train_seconds": statistics.fmean(xs) if xs else None}
    return res


def same_genome_timing(on_run: dict, off_run: dict) -> dict:
    """Train-time ratio OFF/ON on genomes (by hash) evaluated in both arms. Both arms start from the
    same population, so generation 0 is an exact controlled comparison (identical genome, both from
    scratch). A ratio well above 1 there means the OFF run was simply slower: an environment effect."""
    def by_hash(run, gen0_only):
        return {c["genome_hash"]: c["train_seconds"] for c in run["cands"] if not gen0_only or c["generation"] == 0}
    out = {}
    for label, gen0_only in (("generation_0", True), ("all_shared", False)):
        a, b = by_hash(on_run, gen0_only), by_hash(off_run, gen0_only)
        shared = sorted(set(a) & set(b))
        ratios = [b[h] / a[h] for h in shared if a[h] > 0]
        out[label] = {"n": len(shared),
                      "median_ratio_off_over_on": statistics.median(ratios) if ratios else None}
    return out


def failure_rates(run: dict, accuracy_below: float = 0.3) -> dict:
    """Share of candidates that trained badly (accuracy < accuracy_below), generation > 0 only.
    Checks whether inherited weights sometimes make offspring collapse."""
    c = [x for x in run["cands"] if x["generation"] > 0]
    if not c:
        return {"n": 0, "failure_rate": None, "mean_accuracy": None}
    return {"n": len(c), "failure_rate": sum(x["accuracy"] < accuracy_below for x in c) / len(c),
            "mean_accuracy": statistics.fmean(x["accuracy"] for x in c)}


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def summarize(out_dir: str, seeds, thresholds=(0.65, 0.70)) -> dict:
    """Per-seed table, paired ON-OFF differences, mean/std across seeds, and the size-matched
    timing check. Seeds with a missing arm are skipped (and listed under 'missing')."""
    per_seed, missing, runs = {}, [], {}
    for s in seeds:
        try:
            runs[s] = {arm: load_run(os.path.join(out_dir, pat.format(seed=s))) for arm, pat in ARMS.items()}
        except FileNotFoundError:
            missing.append(s)
            continue
        per_seed[s] = {arm: arm_summary(r, thresholds) for arm, r in runs[s].items()}

    diffs = {"final_best": [], "final_mean": [], "hours": []}
    for s, arms in per_seed.items():
        for k in diffs:
            diffs[k].append(arms["on"][k] - arms["off"][k])
    wins = sum(d > 0 for d in diffs["final_best"])

    agg = {}
    for arm in ARMS:
        agg[arm] = {k: _mean_std([per_seed[s][arm][k] for s in per_seed])
                    for k in ("final_best", "final_mean", "hours", "mean_inherit_fraction")}
    timing = {s: {arm: time_by_size(runs[s][arm]) for arm in ARMS} for s in runs}
    same_genome = {s: same_genome_timing(runs[s]["on"], runs[s]["off"]) for s in runs}
    failures = {s: {arm: failure_rates(runs[s][arm]) for arm in ARMS} for s in runs}

    return {
        "per_seed": per_seed,
        "mean_std": agg,
        "paired_diff_on_minus_off": {k: {"values": v, "mean": _mean_std(v)[0], "std": _mean_std(v)[1]}
                                      for k, v in diffs.items()},
        "seeds_where_on_better": f"{wins}/{len(per_seed)}",
        "timing_by_size": timing,
        "same_genome_timing": same_genome,
        "failure_rates": failures,
        "missing": missing,
        "thresholds": list(thresholds),
    }


def format_report(r: dict) -> str:
    lines = [f"seeds compared: {sorted(r['per_seed'])}  (missing: {r['missing'] or 'none'})"]
    lines.append(f"{'seed':>4} {'arm':>4} {'final_best':>10} {'final_mean':>10} {'hours':>6} {'inherit':>7}  "
                 + "  ".join(f"gen>={t}" for t in r["thresholds"]))
    for s, arms in sorted(r["per_seed"].items()):
        for arm in ("on", "off"):
            a = arms[arm]
            reach_s = "  ".join(f"{str(a[f'gen_to_{t}']):>8}" for t in r["thresholds"])
            lines.append(f"{s:>4} {arm:>4} {a['final_best']:>10.4f} {a['final_mean']:>10.4f} "
                         f"{a['hours']:>6.2f} {a['mean_inherit_fraction']:>7.2f}  {reach_s}")
    lines.append("")
    for arm in ("on", "off"):
        m = r["mean_std"][arm]["final_best"]
        lines.append(f"{arm.upper():>3} final best fitness: {m[0]:.4f} +/- {m[1]:.4f}")
    d = r["paired_diff_on_minus_off"]["final_best"]
    lines.append(f"paired ON-OFF final best: {[round(v, 4) for v in d['values']]}  "
                 f"mean {d['mean']:+.4f}  std {d['std']:.4f}  ON better in {r['seeds_where_on_better']} seeds")
    h = r["paired_diff_on_minus_off"]["hours"]
    lines.append(f"paired ON-OFF wall-clock hours: mean {h['mean']:+.3f}  (see timing check below)")
    lines.append("")
    lines.append("Mean train seconds per candidate at matched model size (ON vs OFF):")
    for s, arms in sorted(r["timing_by_size"].items()):
        for name, _, _ in PARAM_BUCKETS:
            on, off = arms["on"][name], arms["off"][name]
            fmt = lambda x: "  n/a " if x["mean_train_seconds"] is None else f"{x['mean_train_seconds']:5.2f}s (n={x['n']})"
            lines.append(f"  seed {s} {name:>8}:  ON {fmt(on)}   OFF {fmt(off)}")
    lines.append("")
    lines.append("Same-genome timing, median OFF/ON train-time ratio (1.0 = no speed difference):")
    for s, v in sorted(r["same_genome_timing"].items()):
        g0, al = v["generation_0"], v["all_shared"]
        f = lambda x: "n/a" if x["median_ratio_off_over_on"] is None else f"{x['median_ratio_off_over_on']:.2f}"
        lines.append(f"  seed {s}: generation 0 (identical genomes, both from scratch) {f(g0)} (n={g0['n']});"
                     f"  all shared genomes {f(al)} (n={al['n']})")
    lines.append("")
    lines.append("Bad offspring (accuracy < 0.3), generations > 0:")
    for s, v in sorted(r["failure_rates"].items()):
        on, off = v["on"], v["off"]
        lines.append(f"  seed {s}: ON {on['failure_rate']:.1%} (mean acc {on['mean_accuracy']:.3f})   "
                     f"OFF {off['failure_rate']:.1%} (mean acc {off['mean_accuracy']:.3f})")
    lines.append("")
    lines.append("If OFF is slower than ON in every bucket, the wall-clock gap is an environment/order "
                 "effect, not a property of inheritance. If buckets match, it is only which sizes were explored.")
    return "\n".join(lines)
