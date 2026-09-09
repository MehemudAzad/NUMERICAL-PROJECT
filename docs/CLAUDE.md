# CLAUDE.md — project context & working agreement

Onboarding for anyone (or any agent) picking up this repo mid-stream. Read this
first, then skim `docs/cse402_guide.html`.

---

## 1. What this project is

CSE-402 (Numerical Analysis) course project, Section A, ~3 weeks, 5 members.
Repo: `github.com/MehemudAzad/NUMERICAL-PROJECT`.

**Thesis.** Diffusion image sampling = numerically integrating the probability-flow
ODE from noise to data, one expensive network call ("NFE") per step. We judge the
solvers by **numerical-analysis standards** — convergence order, stability limit
near the stiff boundary `t → 0`, and cost per NFE — not by image-quality scores (FID).

**Base paper.** Lu et al., *DPM-Solver* (NeurIPS 2022, arXiv:2206.00927) —
`docs/dpm-solver-paper.pdf`. We vendor their solver file unmodified and drive it
with an analytic oracle so Tiers 1–2 test the *published code* with zero model error.

**THE authoritative plan is `docs/cse402_guide.html`** — a 12-step implementation
guide (steps 0–11) with all the formulas. `docs/CSE402_Proposal_Deck_v3.pdf` is the
older proposal; the guide lists 11 corrections to it. `docs/numerical-project.md` is
a chat log. `docs/general_instructions.md` is the owner's working rules (§2 below).

### Three arms (separating the two ideas in DPM-Solver)

| Arm | Integrates | Grid | Purpose |
|-----|-----------|------|---------|
| A | raw PF-ODE in `t` | uniform in `t` | baseline |
| B | same ODE reparameterised in `λ` (half log-SNR) | uniform in `λ` | isolates the reparameterisation (A vs B) |
| C | DPM-Solver 1/2/3 — exact on the linear part | uniform in `λ` | isolates the exact linear treatment (B vs C) |

### Three tiers (trading reference exactness for realism)

| Tier | Testbed | Reference | Measures | Runs on |
|------|---------|-----------|----------|---------|
| 1 | anisotropic Gaussian, sweep κ = s_max/s_min | closed-form algebra | stability limits | laptop (CPU) |
| 2 | Gaussian / point mixture | DOP853, rtol 1e-13 | order under curvature | laptop (CPU) |
| 3 | real CIFAR-10 DDPM UNet `google/ddpm-cifar10-32` | fine-grid, same net | survival vs a real network | **Kaggle T4 GPU** |

### The headline result (M8)

The **crossover**: the step size `h*` below which DPM-Solver-3 beats DPM-Solver-1
and above which it *loses*. This explains an unexplained blow-up in the paper's own
Table 6 (order-3 far worse than order-1 at ~10 NFE). Lead with it.

---

## 2. How we work — non-negotiable

From `docs/general_instructions.md` and the owner's direct instructions:

1. **Plan before code.** Present an implementation plan / milestone spec and get
   **explicit approval** before writing code. This is research work.
2. **One milestone at a time.** Finish + validate one before starting the next.
   Every delivered milestone must be runnable by the owner **without the agent's help**.
3. **Notebooks (`.ipynb`) are the deliverable** for anything that produces results.
   The engine lives in `src/` (plain Python, unit-tested); notebooks import it, run
   the sweeps, and write `results/` + `figures/`. Small local sanity checks in the
   venv are fine.
4. **Real compute is Kaggle** (T4 GPU; RTX 6000 Pro if needed). Laptops have no CUDA.
   Only Tier 3 needs a GPU — Tiers 1–2 are pure NumPy.
5. **The owner runs the code and checks results**, unless they explicitly ask
   otherwise. The agent's job is correct code + a notebook that demonstrates it.
6. **The agent never commits.** It provides copy-paste `git add … && git commit`
   commands; the owner runs them. Work directly on `main` — no feature branches.
7. **End every response** with three lines: **Purpose/agenda**, **Newly implemented**,
   **Next steps for me**.
8. **5-person team, sequential.** One person works at a time and may hand off (this
   file is the handoff). Don't try to build the whole project in one go.

---

## 3. Environment

Tiers 1 & 2 are CPU-only NumPy — any laptop works.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # from repo root — expect all green
```

> **If `pip install` fails** with a truststore / `platform.mac_ver()` error: the
> original owner's Homebrew `python@3.14` is broken on macOS 26. Workaround that
> machine used: `uv venv --python 3.12 .venv` then `uv pip install -r requirements.txt`.
> Homebrew `python@3.11` also works. On a normal machine ignore this.

`requirements.txt` is pinned (numpy 2.5.2, scipy 1.18.1, matplotlib, pandas,
pytest, torch 2.14). Tier-3 extras (`diffusers`, `pytorch-fid`) are commented out —
they're Kaggle-only.

Notebooks: `jupyter lab notebooks/NN_*.ipynb`, Run All. Each ends with a cell that
runs its own pytest file and prints `M<n> validation: PASS`.

---

## 4. Repo layout

```
src/               pure-python engine — NO gpu, NO plotting. Unit-tested.
  schedule.py      VP-linear noise schedule; λ ↔ t          [M1 ✅]
  testbeds.py      pf_rhs_t / pf_rhs_lambda + GaussianTier1  [M2 ✅]  (MixtureTier2 → M7)
  runlog.py        THE results-CSV schema (see §5)           [M0 ✅]
  solvers.py       arm A steppers (RK1-4 + AB2) + integrate() [M3 ✅]
  grids.py         grid_t vs grid_lambda                     [M3 ✅]
  dpm.py           arm C: DPM-Solver-1 + DDIM, hand-written  [M4 ✅]
  arm_c.py         wrapper around third_party/ for DPM-2/3   [M4 ✅]
  metrics.py       sliding-window log-log order fitter       [M5]
  stability.py     bisection for max stable h                [M6]
tests/             pytest; runs in seconds on the laptop. Gate G1 lives here.
notebooks/         one NN_*.ipynb per milestone
third_party/       vendored dpm_solver_pytorch.py (unmodified, pinned commit
                   8acf2bb, MIT). See third_party/README.md. NEVER edit it —
                   if a patch is needed, copy to src/dpm_solver_patched.py.
results/           git-tracked CSVs. Big *.pt/*.npz are gitignored.
figures/           git-tracked PNGs.
docs/              the guide, the deck, the paper, the chat log, the rules.
```

---

## 5. Conventions — agreed at step 0, do not renegotiate

- **float64 everywhere** on Tiers 1 & 2. A float32 ~1e-7 error floor destroys order fits.
- **Every run seeded.** No exceptions.
- **One results schema** — `src/runlog.py`, `append_row(csv_path, **fields)`:
  `tier, testbed, arm, solver, order, kappa, h, nfe, err_l2, diverged, seed`.
  Every experiment appends rows in this exact shape so tables concatenate.
- **λ is strictly DECREASING in t.** `t=1` → most negative λ; `t→0` → λ→+∞.
  Getting this backwards is the single most common bug in this project.
- **NFE ≠ step count.** RK4 = 4 network calls / step; DPM-Solver-k = k / step;
  AB2 = 1 / step (+1 startup). Every cross-arm plot uses measured **NFE**. Use `h`
  only *within* an arm, for order fits.
- **Order is defined in `h`, not NFE.** Fit and report both slopes.
- Arm C `singlestep_fixed` real NFE is `(steps // order) * order` — record the real number.

---

## 6. Status

| # | Milestone | State | Delivered |
|---|-----------|-------|-----------|
| M0 | Scaffold, results schema, test wiring, fix vendored solver (was a 404 stub) | ✅ | `src/runlog.py`, `pyproject.toml`, `requirements.txt`, `.gitignore` |
| M1 | VP-linear noise schedule, λ ↔ t | ✅ | `src/schedule.py`, `tests/test_schedule.py` (7), `notebooks/01_schedule.ipynb` |
| M2 | Tier-1 Gaussian testbed: exact score + exact trajectory + ODE-consistency gate | ✅ | `src/testbeds.py`, `tests/test_tier1.py` (9), `notebooks/02_tier1_testbed.ipynb` |
| M3 | arm A solvers (Euler, RK2, RK3, RK4, AB2) + arm B grids | ✅ | `src/solvers.py`, `src/grids.py`, `tests/test_solvers.py` (14), `notebooks/03_solvers.ipynb` |
| M4 | arm C (DPM-Solver-1 + authors' code) + **Gate G1** | ✅ | `src/dpm.py`, `src/arm_c.py`, `tests/test_gate_g1.py` (2), `tests/test_arm_c.py` (5), `notebooks/04_arm_c.ipynb` |
| M5 | order fitter + Tier-1 convergence experiment (first results/figures) | ✅ | `src/metrics.py`, `tests/test_metrics.py` (5), `notebooks/05_convergence.ipynb`, `results/tier1_order.csv`, `figures/05_error_vs_h.png`, `figures/05_error_vs_nfe.png` |
| M6 | stability envelope, κ sweep | ✅ | `src/stability.py`, `tests/test_stability.py` (6), `notebooks/06_stability.ipynb`, `results/stability_envelope.csv`, `figures/06_stability_envelope.png` |
| M7 | Tier-2 mixture testbed + DOP853 reference + order under curvature | ✅ | `MixtureTier2`/`swiss_roll`/`dop853_reference` in `src/testbeds.py`, `tests/test_tier2.py` (13), `notebooks/07_tier2.ipynb`, `results/tier2_order.csv`, `figures/07_error_vs_h.png`, `figures/07_error_vs_nfe.png` |
| **M8** | **crossover study (h\* where order-3 overtakes order-1) — headline** | ✅ | `src/crossover.py`, `tests/test_crossover.py` (7), `notebooks/08_crossover.ipynb`, `results/crossover_sweep.csv`, `figures/08_crossover.png` |
| M9 | Tier-3 CIFAR-10 Kaggle notebook | ⬜ **next** | |
| M10 | final figures, `run_all`, report tables (confirmed / refuted / dropped claims) | ⬜ | |

`python -m pytest` → **71 passed** (was 19 as of commit `27dd657`; +14 M3, +7 M4,
+5 M5, +6 M6, +13 M7, +7 M8).

Key facts already verified: closed-form Tier-1 trajectory satisfies the ODE to
~1e-11 (finite-diff) and matches an independent DOP853 integration to 1e-12; our
`schedule.py` agrees with the vendored `NoiseScheduleVP('linear')` to 1e-11 (so
arms A/B and arm C provably share one schedule). All arm-A/B steppers (euler,
midpoint, heun3, rk4, ab2) hit their textbook order on Tier 1 to within ±0.15;
measured NFE matches `NFE_PER_STEP[s]*N` (`N+1` for ab2) exactly, not just
approximately. **Gate G1 passes**: `dpm_solver_1` ≡ `ddim_step` to
`5.5e-16` (bar was `1e-14`); `dpm_solver_1` measures at order `0.99`. The
`arm_c.py` wrapper (authors' code, `algorithm_type="dpmsolver"`) reproduces
`dpm_solver_1` to `9e-16` and measures orders 1/2/3 at `0.99/2.05/3.15`.

**M5 order table** (`notebooks/05_convergence.ipynb`, κ=10, `results/tier1_order.csv`):
every solver in every arm hits its theoretical order to within ±0.1, R² > 0.9997
across the board — arm A/B euler/midpoint/heun3/rk4/ab2 at
1.01/2.03/3.01/3.94/2.03 (A) and 1.02/2.06/3.01/3.92/1.94 (B), arm C
ddim/dpm1/dpm2/dpm3 at 0.99/0.99/2.03/3.09. `fit_order`'s sliding window
correctly excludes the round-off tail without being told where it is (verified
in `tests/test_metrics.py` against both a clean power law and a synthetic curve
with a noisy round-off floor).

**Sweep ranges are per-solver, not shared** — a uniform `n ∈ [8..256]` sweep
was tried first and measurably biased two entries: `rk4` (pre-asymptotic
contamination at `n=8` alone dragged arm A's slope down to 3.84) and, more
sharply, **`ab2` on arm B**, which has a real, reproducible non-monotonic bump
in its error curve around `n≈16–32` (error rises before resuming its order-2
decay — shown directly in the notebook's §2b with a fine `n` sweep, and
independently checked outside the notebook, not an artifact of the fit).
`fit_order`'s window search couldn't fully see past this with only 6 candidate
points and its fixed `+0.01/point` width bonus preferring the wider, biased
window over a narrower, cleaner one (measured 1.75 vs. a true ~1.94) — fixed by
starting each solver's sweep past its own pre-asymptotic region (`ab2` at
`n=64`, others at `n=16`), not by post-hoc filtering. Worth remembering for M7:
`fit_order` is not immune to a badly-chosen sweep range, and a non-monotonic
bump like ab2's arm-B one won't announce itself unless checked at finer
resolution.

**M6 finding — arm A does not destabilize on Tier 1, contrary to the spec's
expectation.** `max_stable_h` (guide's exact bisection) was swept over
κ ∈ {1, 10, 10², 10³, 10⁴} × all five arm-A/B steppers, and separately
stress-tested to κ up to 1e8 and `t_end` down to 1e-6
(`notebooks/06_stability.ipynb`): **`h_max` is exactly flat** — arm A always
reports the coarsest grid tried (`n_lo=2`, `h≈0.5`) as already stable, for every
κ and every stepper. This is not a bug (bisection logic is verified separately
against a synthetic linear ODE with a closed-form instability threshold, in
`tests/test_stability.py`); it's structural to the VP-linear schedule: the
uniform-t grid's node nearest `t_end` sits at `t_end + h`, and the exact
stability product `h · J(t_end + h)` (`J` = local Jacobian of the linear PF-ODE)
never exceeds ≈0.9 for any `h` or κ — well under explicit Euler's threshold of
2 — because `σ(t)² ~ β₀t` near the boundary makes the local stiffness and the
grid's shrinking distance to `t_end` roughly self-cancelling. **This narrows,
not confirms, paper §4.2's instability claim**: it doesn't show up on Tier 1's
linear, decoupled problem under this metric. Real instability (if any) is now
an open question for Tier 2 (M7, curved score) or Tier 3 (M9, real network),
where this cancellation has no reason to hold — worth deciding explicitly as a
team before M7/M9, since it changes what M6's figure can claim in the report
(the flat stability map is still worth showing, but the "arm A degrades
sharply" framing in the M6 spec above should not be used until/unless a later
tier reproduces it).

**M7 finding — every solver keeps its textbook order under real curvature.**
`MixtureTier2` (`mog8`: 8 modes on a Swiss-roll curve, `swiss_roll(n=8,
noise=0.0)`) makes the PF-ODE right-hand side genuinely nonlinear in `x` (the
posterior mean is a softmax over 8 modes, not a fixed linear shrinkage like
Tier 1's), with no algebraic reference — `dop853_reference` (SciPy DOP853,
`rtol=1e-13`) stands in, checked converged against a tighter `rtol=1e-11`
(agreement `6.9e-13`, far below every measured error). Re-running M5's exact
sweep design on this testbed (`notebooks/07_tier2.ipynb`): worst
`|measured − theoretical|` across all 14 (arm, solver) pairs is **0.084**,
mean R² **0.99996** — as clean as Tier 1's linear numbers, and the max slope
shift from Tier 1's kappa=10 table is only **0.059**. One deliberate
deviation from M6: `T_END=0.2`, not `1e-3` — pushing to the stiff boundary
*combined with* real curvature left every solver's sweep still pre-asymptotic
at the M5 `n_steps` ranges (measured slopes off by up to 1.2), so this
experiment isolates curvature alone, the same way M5 isolated stiffness alone
at a moderate `t_end`. Net read: the order theorem's smoothness hypothesis
(paper Assumption B.1) holds on this curved-but-exact-score testbed, same as
the linear one — no order loss from curvature by itself.

**M8 finding — the crossover is real, flat across kappa, and moved toward
the paper's number by curvature, but doesn't reach it.** `src/crossover.py`
(`crossover_h`) finds every sign change of `log(err_dpm3) − log(err_dpm1)`
between the two curves swept at *matched h* (a shared macro-step-count list,
so both orders share the same `h`; NFE differs 3x, per `singlestep_fixed`'s
cost). At `T_END=1e-3` (the paper's own `ε=1e-3`), Tier 1 crosses at
`nfe3_star` in **4.4–5.0** across all five swept kappa (1 → 1e4) — essentially
flat, echoing M6's kappa-independence finding for stability. Tier 2 (`mog8`)
crosses later, at `nfe3_star ≈ 8.9` — closer to the paper's observed ~10–12
NFE (Table 6), but still short of it. Read together with M7: **stiffness
alone doesn't move the crossover toward the paper's number; curvature does,
partway.** Both analytic tiers still have an *exact* score (zero model
error), so the honest conclusion is that the remaining gap is a reasonable
place to lay the blame on the real network's approximation error and much
higher intrinsic dimensionality (d=3072 vs d=2–3 here) — this narrows, not
confirms, Table 6's finding, the same shape of result M6 reported for
stability. `notebooks/08_crossover.ipynb` §5 reproduces the guide's own
illustrative sketch with real data (order-3 visibly worse than order-1 left
of the dashed `h*` line, on both Tier 1 kappa=10 and `mog8`). One thing worth
remembering if M9/M10 revisit this: DPM-Solver-3's error is genuinely
non-monotonic at the coarsest grids tested (a real dip-then-rise, not noise —
visible directly in `figures/08_crossover.png`), which is exactly why
`crossover_h` reports *every* sign change rather than assuming a single
crossing; in every group actually swept here there was only one anyway.

**Gotcha found in `third_party/dpm_solver_pytorch.py`:** for `schedule='linear'`,
`NoiseScheduleVP(..., dtype=...)` is silently ignored — `get_time_steps()` builds
its own `torch.tensor`/`torch.linspace` using torch's *ambient* default dtype
(float32), which floors the wrapper's precision at ~1e-7 regardless of the
`dtype` argument. `src/arm_c.py` fixes this the only way that doesn't touch the
vendored file: `torch.set_default_dtype(torch.float64)` as a module-level
side effect. Harmless project-wide (nothing else here uses torch), but if a
future milestone imports `third_party` directly without going through
`src/arm_c.py`, it will silently get float32 timesteps again.

---

## 7. Milestone specs (M3 onward)

Each milestone = `src/` module(s) + `tests/test_*.py` + `notebooks/NN_*.ipynb` that
ends with an inline pytest cell. Follow the guide step in parentheses.

### M3 — arm A solvers (incl. AB2) + arm B grids  (guide steps 3–4)

`src/solvers.py`:
- **One-step explicit RK** — `euler, midpoint, heun3, rk4`, **exactly the paper's
  Appendix E.4 schemes**: explicit **midpoint** for RK2; **Heun's 3rd-order with
  r₁=1/3, r₂=2/3** for RK3 (mirrors DPM-Solver-3 — generic Heun/RK4 would break
  comparability with Table 1). Signature `stepper(rhs, x, a, b) -> x_next`.
- **`ab2` — Adams–Bashforth 2, linear multistep** (restores the deck's multistep
  row; RK45/adaptive from the deck is **cut** — step size isn't a free variable, so
  "order vs h" and "max stable h" aren't defined on it; mention it only as a
  reference curve if wanted). Order 2, **1 NFE/step after startup**. Recurrence
  `x_{n+1} = x_n + h(3/2 f_n − 1/2 f_{n-1})` with `f_k = rhs(x_k, node_k)` cached
  and reused. **Startup:** one explicit-midpoint (RK2) step for `x_1` (O(h³) local,
  so it can't contaminate the order-2 slope). Total NFE for N steps = `N + 1`.
  Because it's multistep, `ab2` does not fit the one-step signature — `integrate`
  handles it as a special branch, not via `STEPPERS`.
- `NFE_PER_STEP = {euler:1, midpoint:2, heun3:3, rk4:4, ab2:1}` (reference/sanity
  only — see below), `STEPPERS = {euler, midpoint, heun3, rk4}`.
- `integrate(rhs, x0, grid, stepper) -> (x_final, nfe)`: wrap `rhs` in a call
  counter and return the **measured** NFE (so AB2's `+1` and any partial count on
  divergence are automatic); `nfe = nan` and stop if any state goes non-finite.

`src/grids.py`: `grid_t(T, t_end, n)` (uniform in t, descending),
`grid_lambda(T, t_end, n)` (uniform in λ; `np.linspace(lmbda(T), lmbda(t_end), n+1)`).

Tests: each solver hits its textbook order on Tier-1 at moderate κ — euler 1,
midpoint 2, heun3 3, rk4 4, **ab2 2** (slope within ±0.15); measured NFE equals
`N+1` for ab2 and `NFE_PER_STEP[s]·N` for the RK steppers; diverged runs flagged.

### M4 — arm C + Gate G1  (guide steps 5–6)  ← CRITICAL CHECKPOINT

`src/dpm.py` (hand-written, ~4 lines each):
```
dpm_solver_1(eps_fn, x, t_prev, t_cur):  h = λ(t_cur) - λ(t_prev)
    return (alpha(t_cur)/alpha(t_prev)) * x  -  sigma(t_cur) * expm1(h) * eps_fn(x, t_prev)
ddim_step(eps_fn, x, t_prev, t_cur):     # paper eq 4.1, original form
    return (a_c/a_p)*x - a_c*(s_p/a_p - s_c/a_c) * eps_fn(x, t_prev)
```
Use `np.expm1(h)`, **not** `np.exp(h)-1` (Appendix D.6 — catastrophic for small h,
which is exactly the regime we measure).

`src/arm_c.py`: wrap the vendored solver. It only calls `model_fn(x, t) -> noise`,
so an analytic oracle works. Critical settings:
- `method="singlestep_fixed"` (NOT `singlestep`/`multistep` — those *mix* orders 1/2/3
  and a convergence slope on them is meaningless, Appendix D.3)
- `skip_type="logSNR"` (uniform in λ = constant h)
- `denoise_to_zero=False` (adds an NFE, moves the endpoint)
- `NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)`, float64

**Gate G1** (`tests/test_gate_g1.py`, end of week 1 — do not build past it):
1. `dpm_solver_1` ≡ `ddim_step`, marched over a shared λ-grid, **`max|xa-xb| < 1e-14`**
   (exact algebraic identity, paper §4.1).
2. `dpm_solver_1` measures as **first order** on Tier-1 (log-log slope in [0.85, 1.15]).

If (1) fails, λ handling is broken — stop and debug. If (2) fails, the error-
measurement machinery is broken.

### M5 — order fitter + Tier-1 convergence  (guide step 7)

`src/metrics.py`: `fit_order(hs, errs, min_pts=4)` — sliding-window log-log fit that
returns the straightest window (auto-excludes the pre-asymptotic and round-off
regions). Report slope + fit window + R² for every solver. `l2(x, ref)`.

Notebook: sweep `n_steps` for every solver in every arm on Tier-1, write rows via
`runlog.append_row`, produce the order table (solver / theoretical order / measured
slope / window / R²) and the error-vs-h and error-vs-NFE log-log plots.

### M6 — stability envelope  (guide step 8)  ← most original result

`src/stability.py`: `diverged(x, x_T, factor=10)`; `max_stable_h(tb, arm, stepper,
...)` — bisect on step count for the coarsest grid that still converges. Sweep
κ ∈ {1,10,10²,10³,10⁴} × all steppers × arms A,B. Plot `h_max(κ)`. Expected: arm A
degrades sharply (the `g²/(2σ)` factor blows up as σ→0), arms B/C stay ~flat.

### M7 — Tier-2 mixture  (guide step 9)

Append `MixtureTier2` to `src/testbeds.py`: weighted point cloud, softmax posterior
weights via `scipy.special.logsumexp` (overflow otherwise at small σ), exact score
`ε*(x,t) = (x - α x̂₀)/σ`. Reference = `solve_ivp(..., method="DOP853", rtol=1e-13,
atol=1e-14)` (SciPy rejects rtol < ~2.2e-14; verify convergence by recomputing at
1e-11). Re-run the M5 convergence machinery under real curvature.

### M8 — crossover  (guide step 11)  ← lead with this

For each testbed and κ: the `h*` where the DPM-Solver-3 error curve crosses the
DPM-Solver-1 curve. Then check whether `h*` measured on the analytic tiers predicts
the NFE ≈ 10–12 crossover the paper reports on images (Table 6).

### M9 — Tier 3, Kaggle  (guide step 10)

Notebook built for a Kaggle T4. `UNet2DModel.from_pretrained("google/ddpm-cifar10-32")`
+ `model_wrapper(lambda x,t: unet(x,t).sample, ns, model_type="noise")` (the
`.sample` matters — it returns a dataclass). `NoiseScheduleVP('discrete',
betas=sched.betas)`. **Cache the 200-NFE reference trajectory to disk** — recomputing
it is the biggest time sink. Same checkpoint + same `x_T` for reference and test
(measuring discretization error, not model error). float32 floors the measurement
at ~1e-3 relative — report where the curve flattens, don't fit through the flat part.
Targets: paper Table 4 (discrete DDPM checkpoint), **not** the deck's 4.70.

**FID — open scope decision.** L2-to-reference is the primary Tier-3 read-out and is
non-negotiable. FID-5k vs NFE (the deck's "recover baseline sample quality" claim) is
*in scope only if week 3 has room*; Gate G2 fallback is "ship Tier 3 as L2 only".
Decide explicitly with the team before M9 rather than letting it drift.

### M10 — finish  (guide Part 5)

`run_all.sh` reproduces every figure from a clean clone. Three-way Tier-3 error
decomposition (discretization / `t_end` truncation / network approximation).

**Proposal-coverage table** (a required deliverable — the honest ledger vs
`CSE402_Proposal_Deck_v3.pdf`). Known going in:

- *Delivered as promised:* the thesis, measured orders vs ground truth, the stiff-
  boundary stability map, the efficiency frontier, Sections 01–02, Tier-1 κ-sweep,
  Tier-3 under neural error.
- *Substituted (with justification):* efficiency frontier in **NFE** not wall-clock
  (paper Table 7); Tier-1 **κ-sweep at fixed d** not the d = 2…100 sweep (linear case
  decouples — optionally show one small d-sweep to demonstrate this empirically);
  Tier-2 **point/Gaussian mixture** not Swiss roll (Swiss roll has no closed-form
  score — a deck error); DOP853 rtol 1e-13 not 1e-14 (SciPy floor).
- *Dropped:* **RK45 / adaptive** solver (step size not a free variable);
  reimplementing DPM-Solver (use the authors' code + analytic oracle instead).
- *Conditional:* Tier-3 **FID** (see M9).
- *Added beyond the deck:* arm B, Gate G1, the crossover study (M8), the Tier-3
  three-way error decomposition.

---

## 8. Gotchas already hit

- Homebrew `python@3.14` is broken on macOS 26 → use `uv` + Python 3.12 (see §3).
- The vendored solver was committed as a 15-byte `404: Not Found` stub; re-fetched
  from commit `8acf2bb`. If it looks wrong, re-run the curl in `third_party/README.md`.
- The vendored file emits a benign `SyntaxWarning: invalid escape sequence '\h'` in a
  docstring — leave it, the file is unmodified on purpose.
- `NoiseScheduleVP(..., dtype=...)` doesn't reach the `'linear'` schedule path —
  `get_time_steps()` builds its own float32 tensors regardless. `src/arm_c.py`
  sets `torch.set_default_dtype(torch.float64)` globally to fix it (see §6, M4).
  Anything that imports `third_party` directly, bypassing `src/arm_c.py`, loses
  this fix and silently gets ~1e-7-floored timesteps.
- Notebooks are committed **with** their output figures (~400 KB each). To commit
  clean: `jupyter nbconvert --clear-output --inplace notebooks/NN_*.ipynb`.
- Notebook path bootstrap (works from anywhere in the repo):
  ```python
  import sys, pathlib
  p = pathlib.Path.cwd()
  while not (p / "pyproject.toml").exists() and p != p.parent: p = p.parent
  sys.path.insert(0, str(p))
  ```

---

## 9. Formula quick-reference

Full derivations in `docs/cse402_guide.html` Part 2. Constants: β₀=0.1, β₁=20, T=1.

```
log α(t) = -(β₁-β₀)/4 t²  - β₀/2 t          σ(t) = sqrt(1 - α²)
β(t) = β₀ + t(β₁-β₀)     f(t) = -β/2       g²(t) = β(t)
λ(t) = log α - log σ  (decreasing!)         t(λ): src/schedule.t_of_lmbda
α̂(λ)=1/√(1+e^{-2λ})   σ̂(λ)=1/√(1+e^{2λ})

PF-ODE in t   (arm A):  dx/dt = f(t) x + g²(t)/(2σ(t)) · ε(x,t)
PF-ODE in λ   (arm B):  dx/dλ = σ̂(λ)² x  - σ̂(λ) · ε(x, t(λ))          [paper E.1]
DPM-Solver-1  (arm C):  x_i = (α_i/α_{i-1}) x_{i-1} - σ_i (e^h - 1) ε(x_{i-1}, t_{i-1}),  h = λ_i - λ_{i-1}   [paper 3.7]

Tier 1:  v_j(t) = α² s_j + σ²
         ε*(x,t)_j = σ(t) x_j / v_j(t)
         x_j(t) = x_j(T) · sqrt(v_j(t) / v_j(T))          (exact, no integration)

Tier 2:  r_k ∝ w_k exp(-‖x - α μ_k‖² / 2σ²)   (softmax)
         x̂₀ = Σ_k r_k μ_k ;   ε*(x,t) = (x - α x̂₀)/σ
```
