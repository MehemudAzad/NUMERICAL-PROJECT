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
2. **Stability Limits:** Does the solver blow up near the stiff boundary $t \to 0$? (The paper asserted classical solvers destabilize, but never measured a stability limit).
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
* **Arm A:** Classical explicit schemes (Euler, Heun, RK3, RK4) on the raw ODE in time $t$.
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
   * Runs on CPU in seconds. Lets us test stiffness condition number $\kappa = s_{\max}/s_{\min}$ over 5 orders of magnitude ($10^1$ to $10^5$).
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

### 1. Order Collapse on Real Neural Networks
* **On smooth math (Tiers 1 & 2):** Every solver achieved close to its exact theoretical order:
  * Order 1 $\approx 0.99$ – $1.00$
  * Order 2 $\approx 2.02$ – $2.03$
  * Order 3 $\approx 3.08$ – $3.09$
  * RK4 $\approx 3.94$ – $4.00$
* **On the real neural network (Tier 3):** Higher-order methods **collapsed**!
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
* On smooth analytical problems, switching from $t$ to $\lambda$ changes very little.
* On the real neural network, switching from $t$ to $\lambda$ yields an **order-of-magnitude error reduction** because it automatically packs steps near the stiff data boundary ($t \to 0$), where the neural network's job is hardest.

---

## 6. Cheat-Sheet: 30-Second Viva / Exam Answer

If your instructor or examiner asks: **"What did you do in this project and what did you find?"**

> *"Diffusion models generate images by integrating an initial-value ODE, where each step requires an expensive neural network forward pass. The base NeurIPS 2022 paper (DPM-Solver) claimed fast high-order sampling, but only tested image quality using FID.*
>
> *In this project, we performed a formal numerical-analysis audit. We tested convergence order, boundary stiffness stability, and error per NFE across three controlled tiers: an exact anisotropic Gaussian, a nonlinear Gaussian mixture, and a real CIFAR-10 UNet.*
>
> *Our key findings were:*
> 1. *DPM-Solver achieves its theoretical order under the assumptions the proof actually needs — a smooth score, matched to the step-size range being fit. Most of what looks like "order collapse" on the real network is already present with an exact score, once Tiers 1–2 are measured at the network's own practitioner NFE budgets rather than an easier asymptotic range; the network then makes it moderately worse on top.*
> 2. *At practical low-step budgets, Order 1 can beat Order 3 — a direct reproduction of the paper's own Table 6 anomaly. Compared the way the paper compares (equal NFE, not equal step count), the crossover rises from ≈5 NFE (the exact linear tier) to ≈14 NFE on the real network, bracketing the paper's own ~10–12.*
> 3. *Instability comes from curvature, not stiffness: provably absent on the linear, decoupled tier (an exact cancellation), but real and measurable once we tested the project's own stability bisector on the nonlinear tier. The λ-reparameterisation's benefit is not network-specific either — it is largest on the exact analytic tiers and still substantial on the real network."*
