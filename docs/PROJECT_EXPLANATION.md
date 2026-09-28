# CSE-402 Numerical Project: Intuitive & Simple Guide

> **Project Title:** Order, Cost & Stability in Diffusion Sampling: A Numerical-Analysis Audit of DPM-Solver  
> **Course:** CSE-402 (Numerical Analysis, Simulation & Modeling) — Section A  
> **Base Paper:** Lu et al., *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*, NeurIPS 2022.

---

## 1. The Big Picture: How AI Generates Images

Modern image generators (Stable Diffusion, Midjourney, DALL-E) are **Diffusion Models**:
* **Forward Process (Adding Noise):** Take a clean photo and gradually add Gaussian noise over time $t$ from $t=0$ to $t=1$, until it becomes pure static.
* **Reverse Process (Generating Images):** Start from pure random noise at $t=1$ and progressively remove the noise step-by-step until a clean image appears at $t=0$.

### The Numerical Analysis Connection
In mathematics, this reverse denoising process is an **Initial Value Problem (IVP)** for an **Ordinary Differential Equation (ODE)** called the **Probability-Flow ODE**:

$$\frac{\mathrm{d}x}{\mathrm{d}t} = f(t)x + \frac{g^2(t)}{2\sigma(t)}\,\epsilon(x, t)$$

To take one step of this ODE, a numerical solver must evaluate the function $\epsilon(x, t)$, which represents the noise predicted by a **massive neural network (UNet)**.
* Calling this neural network is **very slow and computationally expensive**.
* We measure cost by **NFE (Number of Function Evaluations)** = how many times we run the neural network.
* Older samplers (like DDIM) required **100 to 250 steps (NFE)** to create a single image.

---

## 2. What the Original Paper (DPM-Solver) Proposed

The authors of DPM-Solver (NeurIPS 2022) observed two critical mathematical properties of the diffusion ODE:

### Idea 1: Don't Approximate What You Can Solve Exactly (Exponential Integrator)
The ODE splits into two parts:
$$\frac{\mathrm{d}x}{\mathrm{d}t} = \underbrace{\text{Linear Part}}_{\text{easy closed-form math}} + \underbrace{\text{Neural Network Part}}_{\text{hard nonlinear function}}$$

* Traditional solvers (Euler, Runge-Kutta RK4) treat the whole equation as an unknown black box and approximate both parts.
* DPM-Solver solves the linear part **exactly** using an integrating factor ($e^h$), and spends its approximation budget **only** on the neural network part using Taylor-like polynomial expansions.
* They formulated **DPM-Solver-1** (Order 1), **DPM-Solver-2** (Order 2), and **DPM-Solver-3** (Order 3).

### Idea 2: Change Coordinates from Time $t$ to Half Log-SNR $\lambda$
As $t \to 0$ (when the image becomes clean), the equation becomes **stiff** and changes rapidly.
* By reparameterizing the ODE with $\lambda(t) = \log(\alpha(t)/\sigma(t))$, uniform steps in $\lambda$ automatically bunch up time steps near $t \to 0$ where the trajectory is most sensitive.

### The Paper's Claim
They claimed their solver could generate high-quality images in just **10 to 20 steps** instead of 100+, and validated this with **FID (Fréchet Inception Distance)** — a metric measuring whether generated pictures look visually appealing to an AI classifier.

---

## 3. Why Our CSE-402 Project Exists (The "Numerical Audit")

In Numerical Analysis, **FID is not an ODE metric**. A picture can look plausible even if the ODE solver drifted wildly off the true mathematical trajectory!

A rigorous numerical analysis asks:
1. **Convergence Order ($p$):** If step size $h$ is halved, does error decrease by $2^p$? ($2\times$ for Order 1, $4\times$ for Order 2, $8\times$ for Order 3, $16\times$ for Order 4).
2. **Stability Limits:** Does the solver blow up near the stiff boundary $t \to 0$? (The paper suggests, citing the literature, that classical explicit solvers "may suffer from unstable numerical issues for large step size", but never measures a stability limit.)
3. **True Cost Efficiency:** High-order methods take multiple network evaluations per step. Dividing error by total NFE, does Order 3 actually beat Order 1 at low budgets?

**Our project is the missing numerical audit of DPM-Solver.**

---

## 4. Experimental Architecture

To isolate each factor scientifically, we designed two orthogonal axes: **3 Arms** and **3 Tiers**.

```
              ┌────────────────────────────────────────────────────────┐
              │                   SOLVER ARMS                          │
              │  Arm A: raw ODE in t     (classical baseline)          │
              │  Arm B: ODE in λ         (isolates coordinate change)  │
              │  Arm C: DPM-Solver in λ  (isolates exact linear part)  │
              └───────────────────────────┬────────────────────────────┘
                                          │ evaluated across
                                          ▼
              ┌────────────────────────────────────────────────────────┐
              │                   TESTBED TIERS                        │
              │  Tier 1: Anisotropic Gaussian    (exact closed form)   │
              │  Tier 2: Gaussian Mixture        (DOP853, tol 1e-13)   │
              │  Tier 3: Real CIFAR-10 UNet      (201-step fine grid)  │
              └────────────────────────────────────────────────────────┘
```

### The 3 Solver Arms (Isolating the Innovations)
* **Arm A:** Classical explicit schemes (Euler, midpoint RK2, Heun's RK3, RK4, and the multistep AB2) on the raw ODE in time $t$.
* **Arm B:** Classical schemes on the ODE transformed into $\lambda$ coordinates.
* **Arm C:** DPM-Solver 1, 2, and 3 (exact linear solution + $\lambda$ coordinates).

> **Scientific Value:** 
> * Comparing **Arm A vs Arm B** reveals exactly what changing coordinates to $\lambda$ accomplishes.
> * Comparing **Arm B vs Arm C** reveals exactly what solving the linear term analytically accomplishes.

### The 3 Testbed Tiers (Finding the Ground Truth)
To calculate numerical error, one needs the exact mathematical solution. We created three levels of testbeds:
1. **Tier 1 (Anisotropic Gaussian):**
   * Data distribution is Gaussian. The ODE decouples into independent 1D linear ODEs with a **closed-form algebraic solution**.
   * Reference error is **0.000000** (exact pen-and-paper formula).
   * Runs on CPU in seconds. Lets us test stiffness condition number $\kappa = s_{\max}/s_{\min}$ over 5 values, $\kappa \in \{1, 10, 10^2, 10^3, 10^4\}$ (stress-tested to $10^8$).
2. **Tier 2 (Gaussian Mixture / Swiss Roll):**
   * Data distribution is an 8-mode mixture on a nonlinear Swiss roll.
   * Score function is closed-form, but nonlinear.
   * Reference answer is computed using SciPy’s 8th-order Runge-Kutta solver (**DOP853**) at machine-level tolerance (`rtol=1e-13`).
3. **Tier 3 (Real CIFAR-10 Neural Network):**
   * A real pretrained diffusion UNet (`google/ddpm-cifar10-32`, dimension $d=3072$).
   * Reference answer is a 201-NFE fine-grid trajectory from the exact same initial noise point $x_T$.
   * Runs on a Kaggle T4 GPU.

---

## 5. Major Discoveries & Novel Contributions

### 1. Order: exact where the theorem applies, pre-asymptotic at practitioner budgets
* **On smooth math (Tiers 1 & 2):** Every solver achieved close to its exact theoretical order:
  * Order 1 $\approx 0.99$ – $1.00$
  * Order 2 $\approx 2.02$ – $2.03$
  * Order 3 $\approx 3.08$ – $3.09$
  * RK4 $\approx 3.94$ – $4.00$
* **On the real neural network (Tier 3), at 10–120 NFE down to $t_{\text{end}}=10^{-3}$:** measured slopes fall well below theory:
  * **RK4 collapsed from $3.94 \to 1.21$**
  * **DPM-Solver-3 collapsed from $3.09 \to 2.47$**
  * **Euler barely dropped ($1.01 \to 0.84$)**
  * (source: `results/master_order_table.csv`, updated M9/M11 — an earlier
    draft of this page misquoted the DPM-3 numbers as "$2.94 \to 1.32$",
    which matches no results file)

#### Why did this happen?
The paper's convergence proof relies on **Assumption B.1** (that the model's
$\lambda$-derivatives exist and are continuous up to order $k{+}1$). Analytic
Gaussians satisfy this trivially. The checkpoint's UNet uses **SiLU/Swish**
activations (not ReLU — ReLU is not differentiable at 0, but this network
does not use it), which *are* smooth, so "the network violates B.1" is hard to
defend as the mechanism. **M12.1 found a better explanation**: re-running
Tiers 1–2 at Tier 3's *exact* protocol (`t_end=1e-3`, NFE 3–120, not the
original sweep's `t_end=0.2`) reproduces most of the order collapse with an
**exact score and no network at all** — e.g. Tier 2's worst gap from theory is
1.9 at matched settings vs 0.08 at the original asymptotic settings. The
defensible claim is *pre-asymptotic behaviour at practitioner NFE budgets*,
which the network then makes somewhat worse on top of (see
`notebooks/12_controls.ipynb`, §12.1).

---

### 2. The NFE Crossover Point ($h^*$)
* High-order methods take more network calls per step (DPM-3 takes 3 calls per step; RK4 takes 4).
* At low budgets, **Order 1 can beat Order 3** — this is a direct reproduction
  of the paper's own Table 6 anomaly (order-3 far worse than order-1 at ~10 NFE).
* Compared **at matched step size** (M8), the crossover sits around $h^*$
  corresponding to $\text{NFE} \approx 4$–$9$. Compared **at matched NFE** —
  the paper's own comparison, since DPM-3 costs 3× DPM-1 per step — the
  crossover moves to **$\text{NFE} \approx 5$ (Tier 1)**, **9–17 with
  re-crossings (Tier 2)**, and **$\approx 14$ (Tier 3, the real network)**,
  a rising trend that reproduces the paper's ~10–12 NFE crossover directly
  (`results/crossover_matched_nfe.csv`, `notebooks/12_controls.ipynb` §12.3).
  An earlier draft of this page claimed "Order 3 only becomes advantageous
  above 20 NFE," which contradicts this data.

---

### 3. Dissecting the Value of $\lambda$ Coordinates
Error of arm A (uniform in $t$) divided by arm B (uniform in $\lambda$) at the largest shared NFE — above 1 means $\lambda$ is better (`results/reparam_gain.csv`):

| | Euler | RK2 | RK4 |
|---|---|---|---|
| Tier 1, $t_{\text{end}}=0.2$ | 0.93 | 0.32 | 0.47 |
| Tier 1, $t_{\text{end}}=10^{-3}$ | 0.69 | 0.85 | **48** |
| Tier 2, $t_{\text{end}}=10^{-3}$ | 9.9 | 33 | **8551** |
| Tier 3 (network) | 0.63 | 0.56 | **10.6** |

* For **RK4**, $\lambda$ helps on **every** tier once the run reaches the data end ($t_{\text{end}}=10^{-3}$) — not only on the network. An earlier draft of this page said "on smooth analytical problems it changes very little"; that was measured at $t_{\text{end}}=0.2$, which never reaches the stiff region.
* For the low-order steppers it is mixed: a big win on the curved Tier 2, slightly worse on Tier 1 and on the network.
* Why: a uniform-$\lambda$ grid packs nodes near $t \to 0$; a uniform-$t$ grid leaves one long last step across it, and RK4 is the stepper whose stages sample that region hardest.

---

### 4. Stability: caused by curvature, not stiffness
* **Tier 1 (linear):** no instability at any $\kappa$ — every stepper is stable even with 2 steps. The product $h \cdot J(t_{\text{end}}+h) \lesssim 0.9$ near $t\to0$ (an exact cancellation), so Euler's threshold of 2 is never reached.
* **Tier 2 (curved mixture):** the Jacobian at $t=10^{-3}$ is ~1000× larger (545 vs 0.54). **RK4 in $t$ has a real stability limit**, $h_{\max}=0.1998$ (stable from 5 steps, diverges at 2–4); RK4 is the only stepper whose last stage evaluates at $t_{\text{end}}$ itself. **Arm B (λ) is stable everywhere.**
* So the paper's §4.2 claim is *narrowed*: explicit steppers in $t$ do destabilise — because of curvature, not the stiffness knob $\kappa$ — and $\lambda$ fixes it. (`results/stability_tier2.csv`, `notebooks/12_controls.ipynb` §12.2)

---

### 5. When does DPM-Solver's "exact linear part" actually help?
* For data **concentrated at a point** ($s \to 0$): $\epsilon = x/\sigma$ is *constant* along the trajectory, so DPM-Solver-1 is **exact**.
* For data **as spread out as the noise** ($s = 1$): the classical right-hand side is **identically zero**, so Euler/RK are exact while DPM-Solver-1 is not.
* Measured (`results/split_benefit.csv`): at 20 NFE with $s<10^{-2}$, DPM-1 beats Euler-in-$t$ at 83% of points and DPM-2 beats RK2-in-$t$ at 100%; at $s=1$ the classical steppers are exact to $10^{-16}$.
* On Tier 1 (variances up to 1) DPM-Solver-1/2 lose to same-order RK at every budget; on Tier 2 (point masses) DPM-2 is the best solver at 10–12 NFE. Natural images are concentrated data — consistent with DPM-Solver's success there.

---

### 6. What the samples look like: accuracy vs image quality (M11)
* The project's first generated images: `figures/11_samples_grid.png` — at 10 NFE, DDIM/DPM-1 give the **right picture, blurred**; RK2-in-$t$ the right picture, noisy; **DPM-2 a sharp but different picture**; DPM-3 (9 NFE) static.
* **FID-5k at 10 NFE** ranks samplers the same way the paper's Table 6 does: DPM-2 24.9 < DPM-fast 31.3 < DDIM 39.5 < DPM-1 44.9 < DPM-3 143 (only the top pair is swapped).
* **But distance to the converged image ranks them in the exact reverse order** (rank correlation −1.00 among the five image-producing samplers; +0.60 by 20 NFE). DPM-2 is 2.3× further from the converged image than DPM-1, yet 1.8× better FID.
* Meaning: "DPM-Solver makes good images in 10 NFE" (true — FID) and "DPM-Solver solves the ODE accurately in 10 NFE" (false — no sampler is within a few gray levels) are different claims. This is the project's thesis, with data. (`results/tier3_fid_vs_l2.csv`)
* Caveat: our absolute FIDs are ~3× the paper's, so only rankings are compared. `notebooks/11b_fid_floor.ipynb` measured the sample-size part: real CIFAR-10 images follow FID ≈ 3×10⁴/N (5.9 at our 5k, 3.15 at 10k — the documented value), which explains about **half** of the gap; the other half is a real difference in our pipeline (FID code / discrete-time conversion). It shifts every sampler equally, so rankings are unaffected.

---

## 6. The Complete Milestones Journey: From M0 to M13

The project was executed in two major phases, with all 14 milestones (M0 through M13) fully completed, scientifically verified, and tested.

```
       Phase 1: M0–M10 (Core Engine & Initial Benchmarks)
   ┌───────────────────────────────────────────────────────────┐
   │ M0-M4: Scaffold, Schedule, Testbeds, Solvers, Gate G1     │
   │ M5-M8: Convergence Order, Stability, Curvature, Crossover │
   │ M9-M10: Tier-3 GPU Run & Initial Report                   │
   └─────────────────────────────┬─────────────────────────────┘
                                 │ Peer Review
                                 ▼
       Phase 2: M11–M13 (Post-Review Controls & Visual Proof)
   ┌───────────────────────────────────────────────────────────┐
   │ M11: Visual Image Grid, DPM-fast, FID vs L2 Inversion     │
   │ M12: Matched Protocol Controls, Real Stability, Split Map │
   │ M13: FID Floor Analysis, Final 826-Line Report & Ledger   │
   └───────────────────────────────────────────────────────────┘
```

### Phase 1: Core Numerical Engine & Initial Benchmarks (M0–M10)
* **M0 (Scaffold):** Repository structure, test harness, results schema, and vendored original DPM-Solver code.
* **M1 (Noise Schedule):** VP linear noise schedule implementation and analytical mapping $\lambda \leftrightarrow t$.
* **M2 (Tier 1 Testbed):** Exact closed-form anisotropic Gaussian testbed with analytical ground truth.
* **M3 (Solvers):** Classical steppers (Euler, Midpoint RK2, Heun RK3, RK4, AB2) in time $t$ (Arm A) and $\lambda$ (Arm B).
* **M4 (DPM-Solver & Gate G1):** Wrapped DPM-Solver (Arm C). Verified **Gate G1**: DDIM and DPM-Solver-1 are identical to $5.5 \times 10^{-16}$.
* **M5 (Convergence Order):** Automatic log-log order fitting pipeline; confirmed asymptotic orders on Tier 1.
* **M6 (Stiffness & Stability):** Condition number sweep $\kappa \in [1, 10^4]$ on linear Gaussian.
* **M7 (Tier 2 Testbed):** 8-mode nonlinear Swiss-Roll Gaussian mixture referenced against SciPy DOP853 at $10^{-13}$ tolerance.
* **M8 (Step-Size Crossover):** Analytical crossover study ($h^*$) between DPM-Solver-1 and DPM-Solver-3.
* **M9 (Tier 3 CIFAR-10 Run):** First neural network evaluation on Kaggle T4 GPU using `google/ddpm-cifar10-32`.
* **M10 (Report & Synthesis):** Synthesis notebook, LaTeX report drafting, and automated pipeline verification (`run_all.sh`).

---

### Phase 2: Post-Review Rigorous Controls & Ground-Truth Verification (M11–M13)
A peer review of M0–M10 revealed that some conclusions compared asymptotic settings with practitioner settings. M11–M13 established strict, like-for-like scientific controls:

#### 🖼️ Milestone 11: Real Samples & The FID Anchor (Kaggle T4 GPU)
* **Notebook:** `notebooks/11_samples_fid.ipynb` (executed on Kaggle T4 GPU).
* **Deliverables & Discoveries:**
  * Generated the project's first visual samples (`figures/11_samples_grid.png`).
  * Implemented and benchmarked `DPM-Solver-fast` (the adaptive hybrid solver actually used in practice).
  * Evaluated 5,000 samples across 12 configurations using `clean-fid`.
  * **The $L_2$ vs. FID Inversion:** At $\sim 10$ NFE, visual quality (FID) ranks: $\text{DPM-2} (24.9) < \text{DPM-fast} (31.3) < \text{DDIM} (39.5) < \text{DPM-1} (44.9) < \text{DPM-3} (143)$. However, mathematical distance to the converged image ($L_2$) ranks them in the **exact reverse order** (rank correlation $-1.00$)! DPM-2 produces a sharp image that looks great to a human, but it diverges from the true ODE path!

#### 🔬 Milestone 12: Controlled Scientific Experiments (Laptop CPU)
* **Notebook:** `notebooks/12_controls.ipynb` (CPU-only, fully automated).
* **Sub-Milestones & Discoveries:**
  * **12.1 (Matched Protocol):** Re-ran Tiers 1 & 2 using Tier 3's exact settings ($t_{\text{end}}=10^{-3}$, low NFEs). Proved that 68% of the "order collapse" happens on pure math without any neural network due to pre-asymptotic step sizes.
  * **12.2 (Nonlinear Stability):** Measured the true stability limit on Tier 2: RK4 in time $t$ blows up ($h_{\max} = 0.1998$, diverging below 5 steps), while RK4 in $\lambda$ stays completely stable.
  * **12.3 (Matched-NFE Crossover):** Compared solvers at equal computational budget (NFE). Proved DPM-3 beats DPM-1 at $\text{NFE} \approx 5$ (Tier 1), $9$–$17$ (Tier 2), and $\mathbf{14.1}$ (Tier 3), confirming the base paper's anomaly.
  * **12.4 (When the Split Helps):** Swept data variance $s \in [10^{-4}, 1]$. Proved DPM-Solver's exponential integrator beats classical RK on concentrated data ($s \to 0$, like real image manifolds), but loses on spread-out data ($s \to 1$).

#### 📊 Milestone 13: FID Floor Analysis & Final Report Synthesis
* **Notebooks:** `notebooks/11b_fid_floor.ipynb` and `notebooks/10_report.ipynb`.
* **Deliverables & Discoveries:**
  * **FID Sample-Size Floor:** Modeled the finite-sample bias of FID ($\text{FID} \approx 30,453 / N$), proving that sample size explains 47% of the absolute gap between our 5k run and the paper's 50k run, while preserving rankings perfectly.
  * **Predictions Ledger (`results/predictions_ledger.csv`):** All 14 hypotheses updated and backed strictly by data (confirmed, narrowed, or clarified).
  * **Full LaTeX Report (`report/main.tex` & `report/main.pdf`):** Completely updated with all M11–M13 findings, matched tables, and dual-axis conclusions.

---

### Complete Milestone Status Table

| Milestone | Description | Environment | Status |
| :--- | :--- | :--- | :---: |
| **M0** | Scaffold, test harness, results schema, vendored code | Local CPU | ✅ Done |
| **M1** | Noise schedule & analytical $\lambda \leftrightarrow t$ mapping | Local CPU | ✅ Done |
| **M2** | Tier 1 anisotropic Gaussian exact testbed | Local CPU | ✅ Done |
| **M3** | Solvers in time $t$ (Arm A) and $\lambda$ (Arm B) | Local CPU | ✅ Done |
| **M4** | DPM-Solver wrapper (Arm C) & Gate G1 ($5.5 \times 10^{-16}$) | Local CPU | ✅ Done |
| **M5** | Automatic order fitting pipeline & Tier 1 convergence | Local CPU | ✅ Done |
| **M6** | Stiffness sweep ($\kappa \in [1, 10^4]$) & stability envelope | Local CPU | ✅ Done |
| **M7** | Tier 2 Gaussian mixture & DOP853 reference ($10^{-13}$) | Local CPU | ✅ Done |
| **M8** | Step-size crossover analysis ($h^*$) | Local CPU | ✅ Done |
| **M9** | Tier 3 CIFAR-10 real UNet evaluation | Kaggle T4 GPU | ✅ Done |
| **M10** | Synthesis notebook & report tables | Local CPU | ✅ Done |
| **M11** | Samples grid, DPM-fast & 5k-FID anchor (`notebooks/11_samples_fid.ipynb`) | Kaggle T4 GPU | ✅ Done |
| **M12** | Matched protocol, Tier-2 stability & split analysis (`notebooks/12_controls.ipynb`) | Local CPU | ✅ Done |
| **M13** | FID floor analysis (`notebooks/11b_fid_floor.ipynb`), LaTeX report & ledger finalized | Local CPU | ✅ Done |

---

## 7. Cheat-Sheet: 30-Second Viva / Exam Answer

If your instructor or examiner asks: **"What did you do in this project and what did you find?"**

> *"Diffusion models generate images by integrating an initial-value ODE, where each step requires an expensive neural network forward pass. The base NeurIPS 2022 paper (DPM-Solver) claimed fast high-order sampling, but only tested image quality using FID.*
>
> *In this project, we performed a formal numerical-analysis audit across 14 systematic milestones (M0–M13). We tested convergence order, boundary stiffness stability, and error per NFE across three controlled tiers: an exact anisotropic Gaussian, a nonlinear Gaussian mixture, and a real CIFAR-10 UNet.*
>
> *Our key findings were:*
> 1. *DPM-Solver achieves its theoretical order under the assumptions the proof actually needs — a smooth score, matched to the step-size range being fit. Most of what looks like "order collapse" on the real network is already present with an exact score, once Tiers 1–2 are measured at the network's own practitioner NFE budgets rather than an easier asymptotic range; the network then makes it moderately worse on top.*
> 2. *At practical low-step budgets, Order 1 can beat Order 3 — a direct reproduction of the paper's own Table 6 anomaly. Compared the way the paper compares (equal NFE, not equal step count), the crossover rises from ≈5 NFE (the exact linear tier) to ≈14 NFE on the real network, bracketing the paper's own ~10–12.*
> 3. *Instability comes from curvature, not stiffness: provably absent on the linear, decoupled tier (an exact cancellation), but real on the curved tier, where RK4 in $t$ has a measured stability limit and the $\lambda$-reparameterisation removes it. For RK4, $\lambda$ coordinates cut error on every tier (48×, 8551×, 10.6×), not only on the network.*
> 4. *Most importantly, trajectory accuracy and image quality disagree: at 10 NFE, ranking samplers by distance to the converged image is the exact reverse of ranking them by FID. DPM-Solver-2 lands 2.3× further from the true ODE solution than DPM-Solver-1 yet scores 1.8× better FID, because it produces a sharp but different image. Our FID ranking matches the paper's, so the paper is right about image quality — but that is a different claim from solving the ODE accurately, which is exactly why solvers should also be judged as numerical methods."*
