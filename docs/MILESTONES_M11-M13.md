# Milestones M11–M13 — post-review fixes

**Status:** proposed 2026-09-25, awaiting approval. Nothing here is implemented yet.
**Read first:** `docs/CLAUDE.md` (working agreement, M0–M10 specs). Same rules apply:
plan approved before code, one milestone at a time, notebooks are the deliverable,
the agent never commits.

M0–M10 are done and the engine is sound: the full test suite passes, Gate G1 holds
(DDIM ≡ DPM-Solver-1 to 5.5e-16), and DPM-Solver-1/2/3 hit orders 0.99 / 2.03 / 3.09
against exact answers. A review on 2026-09-24 found that several of the **conclusions**
drawn from that engine rest on comparisons made under different settings. M11–M13
fix that. Order matters: **M11 first**, because its result decides whether the Tier-3
numbers can be trusted at all.

---

## Why these milestones exist

### Finding 1 — On our metric, DPM-Solver does not beat RK at the paper's budgets

Same checkpoint as paper Table 6's row "CIFAR-10 (DDPM discrete-time model),
ε = 1e-3". Our error is converted from `results/tier3_error.csv` into average
per-pixel distance from the converged 201-NFE image, in 8-bit gray levels
(1 = invisible, 20+ = visibly a different picture):

| NFE | DDIM (= DPM-1) | DPM-2 | DPM-3 | RK2 in t (arm A) |
|---|---|---|---|---|
| 10 | 21.8 | **47.8** | 80.4 (at 9) | 12.4 |
| 12 | 18.9 | 27.5 | 29.8 | 10.0 |
| 20 | 13.0 | 7.6 | ~7 (at 18) | 4.3 |
| 30 | 9.6 | 2.9 | 1.4 | 2.3 |

Paper Table 6 (FID) at 10 NFE: **DPM-2 7.90** < DPM-1 16.69 < DPM-3 24.37. Our
ranking of DPM-2 is the opposite. The overall speed-up does match the paper (~10×):
DDIM needs ~360 NFE to get within 1 gray level of the converged image, DPM-Solver-3
needs ~36. The disagreement is only at 10–12 NFE.

I found no code bug: arm C calls the authors' code with Table 6's exact settings, and
the reference is DPM-Solver-3 itself (a bias would *favour* arm C). Two gaps remain:
- **DPM-Solver-fast was never run.** It is the sampler the paper actually uses at
  ≤ 20 NFE, and the best entry in Table 6 on this checkpoint (FID 6.42 at 10 NFE).
- **No generated image exists anywhere in the project.** Notebook 09 saves only
  error plots.

So either (a) L2-to-reference and image quality disagree — DPM-Solver lands on a
*different but sharp* image — which is a real finding, or (b) our Tier-3 setup
differs from the paper's. Images plus a small FID decide which. → **M11**

### Finding 2 — Tiers 1–2 and Tier 3 were run under different settings

Notebooks 05 and 07 stop at `t_end = 0.2` with NFE ≥ 16–64. Tier 3 runs to
`t_end = 1e-3` with NFE 8–120. The report does not mention this. Re-running Tiers
1–2 with Tier 3's exact settings (exact score, zero model error):

| Slope | Theory | Tier 1 (matched) | Tier 2 (matched) | Tier 3 (network) |
|---|---|---|---|---|
| A / midpoint | 2 | 2.40 | 1.49 | 1.53 |
| A / rk4 | 4 | 3.30 | 2.03 | 1.21 |
| B / rk4 | 4 | 4.72 | 4.51 | 2.23 |
| C / dpm3 | 3 | 3.49 | 4.21 | 2.47 |

Most of the "order collapse" happens with no network at all: the worst gap from theory
is 1.97 on Tier 2 vs 2.79 on Tier 3, not 0.094 vs 2.79 ("a factor of thirty-three").
Also, Assumption B.1 (paper App. B) only requires the λ-derivatives of ε to *exist and
be continuous*. The DDPM UNet uses SiLU activations, which are smooth, so "the network
violates B.1" is hard to defend. The defensible claim is *pre-asymptotic behaviour at
practitioner budgets, made worse by the network*. → **M12.1**

### Finding 3 — "Stability claim refuted" was only tested on the linear problem

On Tier 1 every run was stable at the coarsest grid tried (n = 2), so the bisection
never engaged (all `h_max` values in `results/stability_envelope.csv` equal the n = 2
step). On the nonlinear Tier 2, the project's own `max_stable_h` finds a real limit:
**A/rk4 `h_max` = 0.1998**. At 2 / 3 / 5 steps, RK4-in-t outputs 11.7× / 6.5× / 3.0×
the norm of the input noise (the correct answer is 0.63×); RK4-in-λ stays bounded
(0.87× → 0.63×). That *supports* paper §4.2, and the λ form fixes it. → **M12.2**

### Finding 4 — The λ-reparameterisation does not "only pay off on the real network"

At `t_end = 1e-3`, 120 NFE, B/rk4 beats A/rk4 by **48×** on Tier 1 and **~8600×** on
Tier 2 (exact scores), against 10.6× on Tier 3. Arms A and B only looked the same
because Tiers 1–2 stopped at t = 0.2. → **M12.1**

### Finding 5 — The crossover is compared at equal step size; Table 6 compares at equal NFE

`results/crossover_summary.csv` uses matched h (DPM-3 spends 3× the NFE). At matched
NFE, DPM-3 overtakes DPM-1 at ≈ 5 (Tier 1), 9–17 with re-crossings (Tier 2) and
**14.1 (Tier 3)**. The Tier-3 value is already in `tier3_error.csv` but was never
computed; it is a direct reproduction of Table 6's anomaly. → **M12.3**

### Finding 6 (new) — DPM-Solver's split can hurt

On Tier 1 (κ = 10), DPM-Solver loses to plain RK at every NFE (20 NFE: DPM-2 0.133,
RK2-in-t 0.034, RK2-in-λ 0.0089). On the coordinate with data variance s = 1 the
t-form right-hand side is identically zero (8.9e-16), so Euler is exact (error 0),
while DPM-1's error is 0.34: treating only the linear part exactly destroys a
cancellation. On Tier 2 (point masses, i.e. concentrated data) DPM-2 is the best
solver at 10–12 NFE. So the split helps on concentrated data and hurts on spread-out
data. → **M12.4**

### Documentation errors found

- The report quotes Table 6's continuous VP-deep row (18.37 vs 11.83); our
  checkpoint's row is DDPM discrete, ε = 1e-3 (24.37 vs 16.69).
- `docs/PROJECT_EXPLANATION.md` §5–6 (the viva cheat sheet) repeats the claims above
  and adds its own: "DPM-Solver-3 collapsed from 2.94 → 1.32" matches no results file
  (Tier 1 dpm3 is 3.09; Tier 3 is 2.47, or 2.995 floor-excluded); "ReLU/Swish …
  non-smooth" is wrong for this SiLU network; "Order 3 only advantageous > 20 NFE"
  contradicts the data (~14).
- The report says 108 tests; the README says 111.

The review's check scripts are in the reviewer's scratchpad, not the repo. M12 turns
them into a notebook.

---

## Step 0 — Repair `run_all.sh` (5 minutes, before anything else)

Commit `9801b81` ("project explanation added") also committed an accidental VS Code
edit: the title cell of `notebooks/10_report.ipynb` became a **code** cell. Its prose
is a Python `SyntaxError`, so `run_all.sh` now fails at notebook 10. Restore the
notebook from the previous commit:

```bash
git checkout 2b6a3ff -- notebooks/10_report.ipynb
python3 -c "import json;print(json.load(open('notebooks/10_report.ipynb'))['cells'][0]['cell_type'])"   # expect: markdown
git commit -m "Restore notebook 10 title cell (accidentally turned into a code cell in 9801b81)" notebooks/10_report.ipynb
```

---

## M11 — Look at the samples; anchor Tier 3 to the paper  (Kaggle T4)

**Goal.** Produce the images the project never had, reproduce the paper's Figure 4
(which uses *our* checkpoint), and use a small FID to check that our Tier-3 setup
matches the paper's. Resolves Finding 1.

### Deliverables

| File | What |
|---|---|
| `notebooks/11_samples_fid.ipynb` | Kaggle notebook, same bootstrap as 09 (Internet ON, T4 ×1) |
| `src/tier3.py` | `sample_dpm_solver_t3(..., method="singlestep_fixed", skip_type="logSNR")` — two new keyword arguments, defaults unchanged so notebook 09 still reproduces |
| `src/imaging.py` | `to_uint8(x)` — map a [-1, 1] batch to uint8 HWC arrays. No plotting in `src/` |
| `tests/test_tier3.py` | new cases for the new sampler options, on CPU with a tiny fake UNet (the existing tests already build the schedule from synthetic betas) |
| `figures/11_samples_grid.png` | the Figure-4 analogue |
| `figures/11_per_image_l2.png` | per-image error distribution |
| `figures/11_fid_vs_l2.png` | FID against L2 for each configuration — shows where the two metrics disagree |
| `results/tier3_fid.csv` | `config, nfe, n_samples, fid, l2_batch, l2_per_image_median` |

### Samplers (all from the same `x_T`, seed 0, `t_end = 1e-3`, same checkpoint)

| Label | How |
|---|---|
| DDIM (quadratic) | DPM-1, `method="singlestep_fixed"`, `skip_type="time_quadratic"` — the paper's Figure 4 baseline (DDIM ≡ DPM-1 on any grid) |
| DPM-1 / DPM-2 / DPM-3 | `singlestep_fixed`, `logSNR` — what notebook 09 ran |
| **DPM-Solver-fast** | `method="singlestep"`, `order=3`, `logSNR` — mixes orders to spend exactly `steps` NFE (vendored docstring: "i.e. DPM-Solver-fast in the paper") |
| RK2 in t (arm A) | `integrate(pf_rhs_t, …, "midpoint")` with `DiscreteSchedule`, as in notebook 09 |
| RK4 in λ (arm B) | `integrate(pf_rhs_lambda, …, "rk4")`, as in notebook 09 |
| Reference | DPM-3, 201 steps (recompute if the cached `.pt` is gone — `results/*.pt` is gitignored) |

### Parts

- **A — Figure 4 on our checkpoint.** 8 starting noises. Rows = the samplers above;
  columns = NFE 10, 12, 15, 20, 50, plus the reference. Label every tile with its
  per-image error in gray levels.
- **B — Per-image L2.** Box plot per sampler and NFE over 64 images, instead of one
  batch-summed number. Shows whether a few images dominate.
- **C — FID anchor.** 5k samples per configuration, at 10 and 20 NFE, for DDIM
  (quadratic), DPM-1, DPM-2, DPM-3, DPM-fast and RK2-in-t (12 runs), plus DPM-3 at
  ~100 NFE as the converged baseline. Use `clean-fid` with precomputed CIFAR-10 train
  statistics (`mode="legacy_tensorflow"` to be comparable with published numbers);
  fall back to computing train statistics from torchvision's CIFAR-10 if the download
  fails. Absolute FIDs will be higher than the paper's (5k vs 50k samples); compare
  **rankings**.

### Validation / decision gate

1. The grid renders and DPM-Solver-fast at 10 NFE looks like the paper's Figure 4.
2. **FID ranking at 10 NFE matches Table 6:** DPM-fast ≲ DPM-2 < DDIM (quadratic) ≈
   DPM-1 < DPM-3.
   - **Matches →** Tier 3 is validated. The L2-vs-FID disagreement is a headline
     result (it is the project's thesis made visible). Proceed to M12.
   - **Doesn't match →** stop. Debug the Tier-3 setup before M12. First suspects:
     the discrete-schedule conversion (the paper used "Type-2 discrete" from its
     original code; the vendored newer version interpolates log α piecewise-linearly)
     and the continuous-to-discrete time-input mapping.
3. `python -m pytest` passes on the laptop.

**Cost:** roughly 30–40 minutes on a T4 at 5k samples; about double at 10k.

---

## M12 — Controls on the analytic tiers  (laptop, CPU only)

**Goal.** Re-run the comparisons behind Findings 2–6 under matched settings, so each
report claim rests on a like-for-like comparison.

### Deliverables

`notebooks/12_controls.ipynb`, new `src/` functions with tests, and:

| # | Experiment | Output |
|---|---|---|
| 12.1 | **Matched protocol.** Tier 1 (κ = 10, d = 3) and Tier 2 (mog8) at `t_end = 1e-3` with Tier 3's NFE budgets `[10, 12, 15, 20, 30, 50, 80, 120]`, arms A/B (euler, midpoint, rk4) and C (dpm1/2/3). Fit over all points and with the auto window. Keep the existing `t_end = 0.2` runs, labelled as the check of the theorem's asymptotic order. | `results/controls_matched_protocol.csv`, slope table beside Tier 3, `figures/12_matched_protocol.png` |
| 12.2 | **Stability on Tier 2.** `max_stable_h` for all 5 steppers × arms A/B; output-growth curves vs step count; sensitivity to the divergence rule (current 10× `‖x_T‖` vs stricter, e.g. 2× the largest data norm). Mechanism: numerical Jacobian of the right-hand side along trajectories near t → 0 — bounded on Tier 1, growing between mixture modes on Tier 2. | `results/stability_tier2.csv`, `figures/12_stability_tier2.png` |
| 12.3 | **Crossover at matched NFE**, all three tiers (Tier 3 from `tier3_error.csv`). Add `crossover_nfe(...)` to `src/crossover.py` (or generalise `crossover_h` to any x-axis). | `results/crossover_matched_nfe.csv`, `figures/12_crossover_matched_nfe.png` |
| 12.4 | **When does the split help?** Gaussian data with variance `s` swept over `logspace(-4, 0)`; errors at 10, 20, 50 NFE for DPM-1 vs Euler-in-t vs Euler-in-λ, and DPM-2 vs RK2-in-t / RK2-in-λ. Include the exact s = 1 cancellation as a unit test and a one-paragraph local-error argument for where the threshold should fall. | `results/split_benefit.csv`, `figures/12_split_benefit.png` |

### Validation

- The matched-protocol slopes reproduce the review's table (Finding 2) within fitting
  noise.
- Tier-2 A/rk4 `h_max` = 0.1998 under the current rule.
- Matched-NFE crossover on Tier 3 ≈ 14 NFE.
- 12.4: at s = 1, Euler-in-t error is ≤ 1e-12 while DPM-1's is not.
- `python -m pytest` passes; the notebook ends with its own inline test cell, like
  01–10.

---

## M13 — Rewrite the conclusions

**Goal.** Make every written claim match the data from M11 and M12.

- **Ledger** (`notebooks/10_report.ipynb` → `results/predictions_ledger.csv`,
  `proposal_coverage.csv`), regenerated from data as before:
  - stability → "absent on the linear problem (with mechanism); present on the
    nonlinear problem for arm A, cured by the λ form";
  - order collapse → "pre-asymptotic at practitioner budgets even with exact scores;
    the network makes it moderately worse";
  - λ-reparameterisation → corrected magnitudes on every tier;
  - crossover → matched-NFE values;
  - new rows for M11 (FID ranking vs Table 6; L2 vs FID) and M12.4;
  - if M11 ran, FID moves from "dropped" to "delivered (small-scale anchor)".
- **`report/main.tex`:** abstract; §Stability; §Crossover (correct Table 6 row:
  DDPM discrete, ε = 1e-3); §Tier 3 order (drop "violates B.1", add the
  matched-protocol control); §reparameterisation; new §Samples and FID; new §When the
  split helps; test count; conclusion.
- **`docs/PROJECT_EXPLANATION.md` §5–6:** fix the viva cheat sheet (see
  "Documentation errors" above).
- **`README.md`, `docs/CLAUDE.md`** (§6 status table, §7 specs), **`run_all.sh`**
  (add notebook 12; skip 11 like 09, since it needs a GPU and its outputs are
  committed).
- **Validation:** `run_all.sh` passes from a clean clone; `tests/test_report.py`
  updated for the new ledger rows; `latexmk` builds the report.

---

## The story after M11–M13 (if M11's gate passes)

1. **The order theorem holds where its assumptions hold**, and Gate G1 confirms the
   published code (unchanged from M0–M10).
2. **When DPM-Solver's split helps:** hurts on spread-out data, helps on concentrated
   data, with a mechanism (M12.4).
3. **Instability comes from curvature, not stiffness:** provably none on the linear
   problem, a real blow-up of RK4-in-t on the mixture, cured by the λ form (M12.2).
4. **Table 6's anomaly reproduced on the paper's own checkpoint** at matched NFE
   (~14), with a rising trend across tiers (~5 → 9–17 → 14) (M12.3).
5. **L2 and FID disagree on the same checkpoint** at 10–12 NFE — shown with images
   (M11). This is the project's thesis ("judge solvers as ODE solvers, not by FID")
   made concrete.
6. **At practitioner budgets, measured order is pre-asymptotic on every tier**; the
   network makes it moderately worse (M12.1).

---

## Open decisions (for the owner)

- **FID sample size:** 5k (~30–40 min on T4) or 10k (~2×, less biased).
- **Image grid rows:** include arms A and B (recommended — they are needed to show the
  L2-vs-FID contrast), or DPM-Solver variants only.
- **Ownership:** one person per milestone, sequentially, as before.
