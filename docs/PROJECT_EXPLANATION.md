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
* **On smooth math (Tiers 1 & 2):** Every solver achieved its exact theoretical order:
  * Order 1 $\approx 0.99$ – $1.00$
  * Order 2 $\approx 2.01$ – $2.02$
  * Order 3 $\approx 2.94$ – $3.00$
  * RK4 $\approx 3.94$ – $3.99$
* **On the real neural network (Tier 3):** Higher-order methods **collapsed**!
  * **RK4 collapsed from $3.94 \to 1.21$**
  * **DPM-Solver-3 collapsed from $2.94 \to 1.32$**
  * **Euler barely dropped ($1.01 \to 0.84$)**

#### Why did this happen?
The paper's convergence proof relies on **Assumption B.1** (that the score function and its higher derivatives are smooth and bounded). Analytic Gaussians satisfy this assumption perfectly. But a real deep neural network (with ReLU/Swish activations, layer norms, and training residuals) is **non-smooth**. High-order Taylor expansions break down when high-order derivatives are rough and noisy!

---

### 2. The NFE Crossover Point ($h^*$)
* High-order methods take more network calls per step (DPM-3 takes 3 calls per step; RK4 takes 4).
* At low budgets (**$\text{NFE} < 15$**), **Order 1 and Order 2 actually beat Order 3**!
* Order 3 only becomes advantageous if you have a budget of $> 20$ evaluations.

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
> 1. *While DPM-Solver achieves its theoretical order on smooth analytical problems, its high order collapses on real neural networks because the network's score function violates the paper's smoothness assumption.*
> 2. *At practical low-step budgets ($\text{NFE} < 15$), low-order solvers match or beat Order 3 because Order 3 spends too many evaluations per step.*
> 3. *We isolated the two ideas of the paper, showing that coordinate reparameterization ($\lambda$) provides the major stability and error improvement on real networks."*
