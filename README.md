# Order, Cost & Stability in Diffusion Sampling

CSE-402 (Numerical Analysis, Simulation & Modeling) course project, Section A.

Diffusion image sampling is an initial-value problem: you integrate the
probability-flow ODE from noise to data, one expensive network call per step.
This project judges the solvers by numerical-analysis standards — **convergence
order**, **stability limit near the stiff boundary**, and **cost per function
evaluation (NFE)** — rather than by image-quality scores (FID).

Base paper: Lu et al., *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic
Model Sampling in Around 10 Steps*, NeurIPS 2022 (arXiv:2206.00927).
The authoritative implementation plan is **`docs/cse402_guide.html`** (12 steps).

## Three arms

| Arm | What it integrates | Grid |
|-----|--------------------|------|
| A   | raw probability-flow ODE in `t`         | uniform in `t` |
| B   | the same ODE reparameterised in `λ` (half log-SNR) | uniform in `λ` |
| C   | DPM-Solver 1/2/3 — exact on the linear part | uniform in `λ` |

A vs B isolates the reparameterisation; B vs C isolates the exact linear treatment.
Arm C uses the authors' own code (`third_party/dpm_solver_pytorch.py`) driven by an
analytic noise oracle, so Tiers 1–2 test the published implementation with zero
model error.

## Three tiers

| Tier | Testbed | Reference answer | Measures | Where it runs |
|------|---------|------------------|----------|---------------|
| 1 | anisotropic Gaussian, sweep κ = s_max/s_min | closed-form algebra | stability limits | laptop (CPU) |
| 2 | Gaussian / point mixture | DOP853, rtol 1e-13 | order under curvature | laptop (CPU) |
| 3 | real CIFAR-10 DDPM UNet (`google/ddpm-cifar10-32`) | fine-grid, same net | survival vs a real network | Kaggle T4 GPU |

## Layout

```
src/          pure-python engine (no GPU, no plotting) — unit-tested
tests/        pytest suite; Gate G1 lives here
notebooks/    one .ipynb per milestone — imports src/, writes results/ + figures/
third_party/  vendored DPM-Solver (unmodified, pinned commit, MIT — see its README)
results/      git-tracked CSVs (schema: src/runlog.py). Big *.pt/*.npz are gitignored
figures/      git-tracked PNGs
docs/         the guide, the proposal deck, the paper
```

## Setup

Tiers 1 & 2 are CPU-only NumPy and run on any laptop.

```bash
# This Mac: Homebrew python@3.14 is broken on macOS 26 (pyexpat/pip), so use uv:
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# A normal machine:
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Run the tests from the repo root:

```bash
python -m pytest
```

## Milestone status

| # | Milestone | State |
|---|-----------|-------|
| M0 | Scaffold: layout, vendored solver, results schema, tests wired | ✅ done |
| M1 | `src/schedule.py` — noise schedule (VP linear), λ ↔ t | ✅ done |
| M2 | `src/testbeds.py` — Tier-1 Gaussian + exact-solution test | ✅ done |
| M3 | `src/solvers.py` (arm A) + `src/grids.py` (arm B) | ✅ done |
| M4 | `src/dpm.py` (arm C) + authors'-code wrapper + **Gate G1** | ✅ done |
| M5 | `src/metrics.py` order fitter + Tier-1 convergence experiment | ✅ done |
| M6 | `src/stability.py` + κ-sweep stability envelope | ✅ done |
| M7 | Tier-2 mixture testbed + reference + order under curvature | ✅ done |
| M8 | Crossover study (h\* where order-3 overtakes order-1) | ✅ done |
| M9 | Tier-3 CIFAR-10 Kaggle notebook | ⬜ next |
| M10 | Final figures, `run_all`, report tables | ⬜ |

Milestones are done **sequentially**, one owner at a time.
