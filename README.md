# Order, Cost & Stability in Diffusion Sampling

CSE-402 (Numerical Analysis, Simulation & Modeling) course project,
**Section A, Group 03**.

> [!IMPORTANT]
> **For evaluators:** each member's individual contribution is listed in
> **[Member contributions](#member-contributions)** directly below.
> The submitted report is **[`report/final/A_03.pdf`](report/final/A_03.pdf)**.

Diffusion image sampling is an initial-value problem: you integrate the
probability-flow ODE from noise to data, one expensive network call per step.
This project judges the solvers by numerical-analysis standards — **convergence
order**, **stability limit near the stiff boundary**, and **cost per function
evaluation (NFE)** — rather than by image-quality scores (FID).

Base paper: Lu et al., *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic
Model Sampling in Around 10 Steps*, NeurIPS 2022 (arXiv:2206.00927).
The authoritative implementation plan is **`docs/cse402_guide.html`** (12 steps).

## Member contributions

The project was split into milestones (M0 to M13), and each milestone was built,
tested and committed by one member. The table follows the git history, so any row
can be checked with `git log --author="<name>"`.

| Member | Student ID | Milestones | What they built |
|---|---|---|---|
| **Mehemud Azad** | 2105014 | M0, M1, M2, M13 | Project setup: repository layout, the shared results schema (`src/runlog.py`), test wiring and the vendored DPM-Solver code (`third_party/`). The VP-linear noise schedule and λ ↔ t conversion (`src/schedule.py`, notebook 01). The Tier-1 Gaussian testbed with its exact score and closed-form trajectory (`src/testbeds.py`, notebook 02). The project plan and documentation in `docs/`, including the M11 to M13 correction plan after the internal review. M13: the rewritten conclusions in `report/main.tex` and the FID sample-size floor experiment (notebook 11b). Presentation slides. |
| **Sayjad Rahman** | 2105021 | M3, M4, M11, M12 | The classical solvers (Euler, RK2, RK3, RK4, AB2) and the shared integrator with measured NFE (`src/solvers.py`), plus the λ-uniform grids (`src/grids.py`, notebook 03). DPM-Solver-1 and DDIM written by hand, the wrapper around the authors' DPM-Solver code, and Gate G1 (`src/dpm.py`, `src/arm_c.py`, notebook 04). M11: generated sample images and FID on Kaggle (`src/imaging.py`, notebook 11). M12: the control experiments (matched protocol, Tier-2 stability, matched-NFE crossover, when the exact linear split helps; notebook 12). |
| **Niloy Das Robin** | 2105019 | M5, M6 | The sliding-window order fitter (`src/metrics.py`) and the Tier-1 convergence experiment (notebook 05). The bisection test for the largest stable step size and the κ-sweep stability envelope (`src/stability.py`, notebook 06). |
| **Khalid Hasan Tuhin** | 2105002 | M7, M8 | The Tier-2 point-mixture testbed and its DOP853 reference trajectory (`src/testbeds.py`, notebook 07). The crossover study that locates where DPM-Solver-3 overtakes DPM-Solver-1 (`src/crossover.py`, notebook 08). |
| **Gourove Roy** | 2105017 | M9, M10 | **Notebooks: [`09_tier3_cifar10.ipynb`](notebooks/09_tier3_cifar10.ipynb) and [`10_report.ipynb`](notebooks/10_report.ipynb)**<br><br>• **M9, Tier 3 on the real network.** Built `src/tier3.py`, which runs every solver on the CIFAR-10 DDPM network using the checkpoint's own discrete noise schedule, and ran it on a Kaggle T4. **Notebook 09** measures error against NFE for all nine solver and arm combinations, caches the reference runs, measures the reference's own error floor, and splits Tier-3 error into discretisation, `t_end` truncation and that floor. Outputs: `results/tier3_error.csv`, `results/tier3_decomposition.csv`, `figures/09_tier3_error_vs_nfe.png`, `figures/09_tier3_decomposition.png`.<br>• **Shared engine.** Made the integrator work on GPU tensors as well as NumPy arrays (`src/solvers.py`) and let the ODE take a schedule object (`src/testbeds.py`), so the same code runs all three tiers. Tests: `tests/test_tier3.py`.<br>• **M10, reporting.** **Notebook 10** builds the master order table, the crossover summary, the predictions ledger and the proposal-coverage table directly from the results files (`results/master_order_table.csv`, `predictions_ledger.csv`, `proposal_coverage.csv`, `crossover_summary.csv`; figures `10_error_vs_nfe_all.png`, `10_efficiency_frontier.png`). M13 later added its sections 6b and 6c. Tests: `tests/test_report.py`. Added the floor-aware order fit to `src/metrics.py`.<br>• **Reproducibility.** Wrote `run_all.sh` and ran it end to end; the regenerated results matched the committed ones to 1.8×10⁻¹⁵.<br>• **Reports.** The first LaTeX report (`report/main.tex`, later rewritten in M13) and the submitted ACM report [`report/final/A_03.pdf`](report/final/A_03.pdf). |

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
run_all.sh    reproduces every figure and CSV from a clean clone
report/       working report (main.tex); report/final/A_03.pdf is the submitted
              report in ACM format (`cd report/final && latexmk -pdf A_03.tex`)
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
python -m pytest          # 129 passed
```

Reproduce every figure and results CSV from a clean clone:

```bash
./run_all.sh              # tests, then notebooks 01–08, 12, and 10
./run_all.sh --tests-only # just the suite
```

`run_all.sh` deliberately skips notebooks 09, 11 and 11b — they need a CUDA GPU
and run on a Kaggle T4 (Internet ON, Accelerator T4 ×1). Their outputs are
committed, so the report notebook reads them without a GPU present.

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
| M9 | Tier-3 CIFAR-10 Kaggle notebook (`src/tier3.py`, notebook 09) | ✅ done |
| M10 | Final figures, `run_all.sh`, report tables (notebook 10) | ✅ done |
| M11 | Samples + FID anchor (`src/imaging.py`, notebook 11) | ✅ done (Kaggle run complete) |
| M12 | Controls on the analytic tiers (notebook 12) | ✅ done |
| M13 | Rewrite the conclusions to match M11+M12: ledger, report, docs; FID floor (notebook 11b) | ✅ done |

Milestones are done **sequentially**, one owner at a time. M11–M13 are a
post-review correction pass — see `docs/MILESTONES_M11-M13.md`.

## Report deliverables

`notebooks/10_report.ipynb` runs no experiments — it reads `results/*.csv` and
emits the guide's Part-5 checklist:

| Artifact | What it is |
|---|---|
| `results/master_order_table.csv` | every (tier, arm, solver): theoretical order, measured slope, fit window, R² |
| `results/crossover_summary.csv` | `h*` and `nfe3*` per testbed and κ — the headline result |
| `results/predictions_ledger.csv` | the guide's six predictions and the 2026-09-24 review's eight, each marked confirmed / refuted / narrowed / pending **from the data** (`source` column says which) |
| `results/matched_protocol_slopes.csv` | measured order under the asymptotic and the practitioner protocol, all three tiers side by side |
| `results/reparam_gain.csv` | what the λ-reparameterisation buys: error of arm A / arm B, per tier and protocol |
| `results/tier3_fid_vs_l2.csv` | FID-5k beside the paper's FID and the L2 distance, per sampler at ~10 and ~20 NFE, with ranks |
| `results/proposal_coverage.csv` | the honest ledger vs the proposal deck: delivered / substituted / dropped / added |
| `figures/10_error_vs_nfe_all.png` | error vs NFE, every tier, every arm |
| `figures/10_efficiency_frontier.png` | error per NFE — higher order is not automatically cheaper |

FID was cut at Gate G2 and **restored by M11 in a limited role**: FID-5k as an
anchor to the paper's Table 6 and a counterpoint to L2 — never as the grading
metric. L2-to-reference remains the primary Tier-3 read-out.

The report is `report/main.tex` → `report/main.pdf`. Build it with `make -C report`
(latexmk) or `make -C report tectonic` (no TeX install needed: `brew install tectonic`).
Its FID caveats use the FID floor from `notebooks/11b_fid_floor.ipynb` (Kaggle): real CIFAR-10
test images follow FID ≈ 3.0×10⁴/N, 5.9 at M11's 5,000 samples.
