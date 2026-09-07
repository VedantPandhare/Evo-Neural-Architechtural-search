# EvoNAS Implementation Guide
### Evolutionary Neural Architecture Search with Weight-Inherited Offspring

This guide walks through building the EvoNAS system end to end, including the weight-inheritance extension you're using as your MethodsX contribution. It's organized so you build in the same order you'll later need to *describe* in the paper: base system first, then the inheritance protocol layered on top, then validation experiments.

---

## 1. Requirements

### 1.1 Hardware
- **Minimum viable:** 1 GPU with ≥8 GB VRAM (a Colab T4 or equivalent works for CIFAR-10 with small populations). CPU-only is possible but painfully slow for anything beyond Fashion-MNIST toy runs.
- **Comfortable:** 1 GPU with 12–16 GB VRAM (RTX 3060/3080, T4, V100, or a cloud A10). This lets you run population sizes of 20–30 with 3–5 epoch training budgets in a reasonable wall-clock time.
- **Storage:** a few GB for datasets + checkpoints. If you implement weight inheritance, budget extra disk for storing parent checkpoints across generations (or keep only the elite fraction's weights to save space).

### 1.2 Software stack
| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.10+ | |
| Deep learning | PyTorch (+ torchvision) | Dynamic graph construction makes variable-depth genomes easy to build |
| Experiment tracking | MLflow or Weights & Biases | Track fitness, generation stats; W&B is easier for quick plots |
| Visualization | Matplotlib | Fitness curves, Pareto plots |
| Profiling | `torch.profiler` or `time.perf_counter` | Latency measurement for the fitness function |
| Packaging | Docker | Reproducibility — required for MethodsX's "resource availability" section |
| Config management | YAML + a small config loader (e.g. `omegaconf`) | Every experiment should be reproducible from a config file |

### 1.3 Skills/background you'll need
- Comfortable writing custom `nn.Module` classes and assembling them programmatically from a spec (not just editing existing architectures).
- Basic genetic algorithm concepts (selection, mutation, crossover) — you don't need a GA library, a custom implementation is expected and preferred here.
- Enough profiling knowledge to measure inference latency and parameter count reliably (these numbers feed directly into your fitness function and your paper's validation section).

---

## 2. Repository structure

Use the structure from the project document, with one addition (`evolution/inheritance.py`) for the weight-inheritance protocol:

```
evo-nas/
├── genomes/
│   ├── schema.py          # genome data structure + validation
│   └── operators.py       # encode/decode genome <-> model spec
├── models/
│   └── builder.py         # genome -> PyTorch nn.Module
├── evolution/
│   ├── population.py      # population init, generation loop
│   ├── selection.py        # tournament / fitness-proportional selection
│   ├── mutation.py
│   ├── crossover.py
│   └── inheritance.py     # NEW — weight transfer protocol (your contribution)
├── training/
│   └── trainer.py          # fixed-budget training per candidate
├── evaluation/
│   └── metrics.py          # accuracy, latency, parameter count, fitness
├── experiments/             # experiment run configs + logs
├── configs/                 # YAML configs per experiment
├── results/                  # plots, checkpoints, comparison tables
├── main.py
└── README.md
```

---

## 3. Build order

Build in this order — each stage should be independently testable before you move to the next.

### Stage 1 — Genome representation (`genomes/`)
1. Define the genome schema (a JSON-serializable dict, as in the project doc): layer sequence, per-layer params, optimizer, learning rate, batch size, dropout.
2. Write `operators.py` to convert a genome into a "build spec" that `models/builder.py` can consume.
3. Write validators: reject genomes that would produce shape-incompatible layer sequences (e.g. a `dense` layer directly after a `conv` without a flatten/pool step handled correctly).
4. **Test:** generate 50 random genomes, confirm all validate or are correctly rejected — this matters later because crossover/mutation will produce invalid genomes you need to catch before wasting a training run on them.

### Stage 2 — Model builder (`models/builder.py`)
1. Write a function that takes a validated genome and returns an instantiated `nn.Module` (likely a small custom class that builds an `nn.Sequential` or `nn.ModuleList` from the layer spec).
2. Handle the flatten step between conv/pool layers and dense layers automatically — infer the flattened dimension by running a dummy forward pass.
3. **Test:** build 20 random valid genomes into models, run a dummy batch through each, confirm no shape errors.

### Stage 3 — Training and evaluation (`training/`, `evaluation/`)
1. Write a fixed-budget trainer: given a model, dataset, and epoch count, train and return validation accuracy.
2. Write latency measurement: average forward-pass time over N runs on a fixed batch size, on the target hardware (CPU or GPU — be consistent, and record which, since this is a MethodsX validation variable).
3. Write parameter counting (trivial — `sum(p.numel() for p in model.parameters())`).
4. Write the fitness function:
   ```
   fitness = accuracy - lambda_params * normalized_param_count - lambda_latency * normalized_latency
   ```
   Normalize params/latency against a reasonable reference range so the lambdas are interpretable.
5. **Test:** run this pipeline on 5 hand-picked genomes (e.g. a tiny CNN, a deeper CNN, an oversized one) and confirm fitness ranks them sensibly.

### Stage 4 — Evolution loop (`evolution/population.py`, `selection.py`, `mutation.py`, `crossover.py`)
1. Population init: generate N random valid genomes (N = 20–30 to start).
2. Selection: implement tournament selection (simpler to reason about than fitness-proportional, and easier to describe/replicate in a methods paper).
3. Mutation operators (implement each as a small, independently testable function): add layer, remove layer, change filter count, change kernel size, change dense width, change activation, change dropout, change LR/optimizer, replace layer type.
4. Crossover: layer-level crossover between two parents, with a validity check afterward (reject/repair invalid offspring rather than silently discarding — decide which, and document it, since MethodsX cares about exact protocol details).
5. Elitism: carry over the top fraction (e.g. top 10–20%) unchanged into the next generation.
6. **Test:** run 3 generations on a tiny population (5 genomes, 2 epochs each) end to end before scaling up — this is your smoke test.

### Stage 5 — Weight inheritance (`evolution/inheritance.py`) — your MethodsX contribution
This is the piece you're documenting as a reusable method, so build it carefully and log everything.

1. **Layer-matching function:** for a parent-offspring pair, walk both layer sequences and identify which offspring layers are shape-compatible with a parent layer at the corresponding position (same layer type, same or compatible input/output dimensions).
2. **Transfer rule:** for matched layers, copy the parent's weights directly into the offspring's corresponding layer *before* training begins. For mutated/added layers with no compatible parent, initialize normally (e.g. Kaiming/Xavier — pick one and be consistent, since this is a protocol detail worth stating explicitly).
3. **Partial-match handling:** define what happens when a layer's shape changed slightly (e.g. filter count changed from 32 to 48) — do you truncate/zero-pad the inherited weights, or fall back to fresh initialization for that layer? This decision is itself a documentable design choice — pick one, justify it briefly, and be consistent.
4. **Crossover case:** when an offspring comes from two parents, decide (and document) which parent's weights take priority for inherited segments — this is exactly the kind of protocol detail MethodsX wants spelled out.
5. **Logging:** for every offspring, log what fraction of its parameters were inherited vs. freshly initialized. You'll want this data for your validation section (e.g. "average 62% of offspring parameters were inherited across generations 3–15").
6. **Test:** unit test the layer-matching function directly — feed it pairs of genomes with known overlaps and confirm it identifies the correct matches. This is more important to get right than any other single component, since a silent matching bug would corrupt your validation results.

### Stage 6 — Full pipeline integration (`main.py`)
1. Wire together: population init → build models → (if generation > 0) apply inheritance → train → evaluate → fitness → select → mutate/crossover → repeat.
2. Add checkpointing so a run can resume after interruption (useful given the compute cost).
3. Add MLflow/W&B logging: per-generation best/mean/worst fitness, population diversity, inheritance fraction.

### Stage 7 — Baseline and comparison
1. Pick a hand-designed CNN baseline of comparable size (e.g. a small ResNet-ish or VGG-ish network) trained under the same budget.
2. Run the **same evolutionary search twice** — once with weight inheritance enabled, once with it disabled (every offspring trained from scratch) — holding population size, mutation rates, and generation count fixed. This paired comparison *is* your MethodsX validation.
3. Record for each run: generations-to-convergence (fitness plateau), total GPU-hours consumed, final best architecture's accuracy/latency/params.

### Stage 8 — Packaging and reproducibility
1. Dockerfile pinning Python/PyTorch/CUDA versions.
2. YAML configs for every experiment you report in the paper (population size, mutation rates, lambda values, epoch budget, generation count, random seed).
3. README with exact commands to reproduce each figure/table in the paper.

---

## 4. Suggested timeline (solo, alongside coursework/internship)

| Phase | Stages | Rough duration |
|---|---|---|
| Core build | 1–4 | 2–3 weeks |
| Inheritance protocol | 5 | 1–1.5 weeks |
| Integration + smoke tests | 6 | ~3–4 days |
| Full experiments (with vs. without inheritance, baseline) | 7 | 1–2 weeks (mostly compute wait time) |
| Docker + configs + README | 8 | 2–3 days |
| Paper writing | — | 1–1.5 weeks, can overlap with Stage 7 compute runs |

---

## 5. What to measure for the paper (don't skip this while building)

Log these from day one — retrofitting logging after the fact means re-running expensive experiments:
- Fitness (best/mean/min) per generation, for both inheritance-on and inheritance-off runs.
- Wall-clock GPU-hours per run.
- Fraction of offspring parameters inherited, per generation.
- Final best architecture's accuracy, parameter count, and latency, compared against your hand-designed baseline.
- At least one plot of accuracy vs. parameter count (and/or latency) across the final population, to show the Pareto trade-off.

These numbers map directly onto the "Method validation" subsection of the MethodsX paper draft — that section is where reviewers check that your method actually does what you claim.
