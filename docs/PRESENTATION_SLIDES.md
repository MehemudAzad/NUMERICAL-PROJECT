# Presentation Slide Deck: Order, Cost & Stability in Diffusion Sampling
### A Numerical-Analysis Audit of DPM-Solver (NeurIPS 2022)
**Course:** CSE-402 (Numerical Analysis, Simulation & Modeling) — Section A  
**Reference Document:** `docs/defense_guide.pdf`

---

## Slide 1: The Core Problem — Diffusion as an ODE

### Title: Diffusion Image Generation is an Initial Value Problem (IVP)
* **What is Diffusion AI? (Stable Diffusion, Midjourney):**  
  Starts from pure random static noise and progressively removes noise until a photorealistic image appears.
* **The Mathematics Under the Hood:**  
  Denoising is not magic — it is integrating a continuous **Probability-Flow ODE**:
  $$\frac{\mathrm{d}x}{\mathrm{d}t} = \underbrace{f(t)x}_{\text{simple linear part}} + \underbrace{\frac{g^2(t)}{2\sigma(t)}\,\epsilon_\theta(x, t)}_{\text{expensive neural network part}}$$
* **The Currency of Cost:**  
  Every single evaluation of $\epsilon_\theta(x, t)$ requires one full forward pass of a deep neural network (UNet).  
  We measure computational cost in **NFE (Number of Function Evaluations)**.
* **The Base Paper (Lu et al., NeurIPS 2022):**  
  * *Claim:* DPM-Solver generates high-quality images in **10 to 20 steps** (instead of 100–250 steps).
  * *How they judged it:* Solely by **FID** (Fréchet Inception Distance) — a computer vision score for visual attractiveness.
* **Our Research Question:**  
  *FID only checks if an image looks pretty. Does the solver actually solve the ODE accurately? What are its true convergence order, stiffness stability, and cost-per-NFE?*

---

## Slide 2: An Intuitive Primer — How Diffusion & ODEs Work

### Analogy: Walking Home in Thick Fog
* Imagine you are walking home through a dense fog using a compass.
* Every time you stop to check the compass, it costs precious time (**1 NFE**).
* **Few large steps:** Fast and cheap, but you easily veer off the road into a ditch.
* **Many small steps:** Highly accurate, but painstakingly slow and expensive.
* **The Goal of DPM-Solver:** A smarter mathematical stepping rule so that **big steps stay on the true path**.

### The Two Core Inventions of DPM-Solver:
1. **Solve the Easy Part Exactly (Exponential Integrator):**  
   The ODE splits into a simple linear piece and a hard neural network piece. Old solvers approximate both. DPM-Solver solves the linear piece **exactly with calculus ($e^h$)**, only approximating the network.
2. **Change the Clock to Lambda ($\lambda = \text{half log-SNR}$):**  
   Normal time $t$ gets dangerously stiff near $t \to 0$ ($\sigma(t) \to 0$ in denominator). Stepping uniformly in $\lambda$ automatically takes **giant leaps in blurry noise** and **tiny baby steps near the finished image**.

---

## Slide 3: Our Experimental Framework (3 Arms × 3 Tiers)

To isolate each idea scientifically, we designed a two-dimensional testing matrix:

### The 3 Solver Arms (Isolating the Innovations)
* **Arm A (Baseline):** Classical solvers (Euler, Midpoint RK2, Heun RK3, RK4, AB2) stepping in **time $t$**.
* **Arm B (Isolating $\lambda$):** The exact same classical solvers stepping in **Lambda ($\lambda$)**.
* **Arm C (DPM-Solver):** Exact linear solution + stepping in **Lambda ($\lambda$)** (DPM-1, DPM-2, DPM-3).
> *Comparing A vs. B isolates the $\lambda$-clock. Comparing B vs. C isolates the exact linear math.*

### The 3 Testbed Tiers (Establishing Ground Truth)
| Tier | Testbed | Reference Truth | What it Tests | Environment |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | Anisotropic Gaussian ($\kappa \le 10^4$) | Exact pen-and-paper algebra ($0$ error) | Clean textbook order & stiffness | Laptop CPU |
| **Tier 2** | 8-mode Swiss Roll Spiral | SciPy DOP853 solver (tol $10^{-13}$) | Behavior under sharp curvature | Laptop CPU |
| **Tier 3** | Real CIFAR-10 UNet ($d=3072$) | 201-step fine-grid simulation | Real-world deep learning network | Kaggle T4 GPU |

### The Safety Check: Gate G1 Passed
* DPM-Solver-1 and original DDIM are algebraically identical.
* Our independent implementations matched to **$5.5 \times 10^{-16}$** (machine precision limit).  
* Confirms zero implementation bugs before running benchmarks.

---

## Slide 4: Finding 1 — Convergence Order: Theory vs. Practice

### Does Each Solver Achieve its Theoretical Order?
* **On Smooth Math (Tiers 1 & 2):**  
  **Yes, textbook perfection.** All solvers hit theoretical order within $0.094$:  
  Euler $\approx 1.01$, Midpoint $\approx 2.02$, DPM-3 $\approx 3.09$, RK4 $\approx 3.99$.
* **On Real CIFAR-10 Network (Tier 3, 10–120 NFE):**  
  **Higher orders collapse:**  
  * **RK4 collapses from $3.94 \to 1.21$** (acted like a 1st-order solver!).
  * **DPM-Solver-3 drops from $3.09 \to 2.47$**.
  * Euler barely changed ($1.01 \to 0.84$).

### Why Did Order Collapse? (The Control Experiment Twist)
* *Old Hypothesis:* "The neural network is non-smooth and violates Assumption B.1."  
  *(False: the network uses SiLU activations, which are infinitely smooth).*
* *Our Discovery (Milestone 12.1 Control):*  
  When we re-ran Tier 2 (pure math, no neural net) under the exact same practitioner budget (10–120 NFE down to $t_{\text{end}}=10^{-3}$), **the slope collapsed to $2.03$ without any network**!
* **Conclusion:** 68% of the order collapse is **pre-asymptotic behavior** — at 10 to 30 steps, steps are too coarse for asymptotic Taylor expansions to take effect. The network only adds a modest penalty on top.

---

## Slide 5: Findings 2 & 3 — Crossover Budget & Stability Limits

### Finding 2: The Table 6 Crossover Anomaly Explained
* In the base paper's Table 6 at 10 NFE, **Order 3 produced worse FID than Order 1** (24.37 vs 16.69), but the authors never explained why.
* **Our Explanation:** DPM-Solver-3 costs **3 network evaluations per step**. At 10 NFE, it only takes 3 coarse steps. Large steps make higher-order polynomial corrections worse, not better!
* **Measured Crossover Budget (Matched NFE):**  
  DPM-Solver-3 only overtakes DPM-Solver-1 once the budget reaches:
  * Tier 1 (Gaussian): $\approx 5$ NFE
  * Tier 2 (Mixture): $\approx 9$ NFE (re-crosses at 14 and 17)
  * **Tier 3 (Real CIFAR-10): $\approx 14.1$ NFE**
* *Rule of Thumb:* Never use DPM-Solver-3 below 15 evaluations.

### Finding 3: Explosions Come from Curvature, Not Stiffness
* The paper claimed explicit solvers destabilize near $t \to 0$ due to stiffness.
* **Tier 1 (Straight, Stiff $\kappa = 10^8$):** No solver ever exploded, even at 2 steps (stiffness and step size cancel out).
* **Tier 2 (Nonlinear Curvature):** **RK4 in time $t$ really explodes!** Diverges at fewer than 5 steps ($h_{\max} = 0.1998$).
* **The Cure:** RK4 in **Lambda ($\lambda$) never explodes** on any testbed.  
  Switching to the $\lambda$-clock reduced RK4 error by **$48\times$ (Tier 1)**, **$8,551\times$ (Tier 2)**, and **$10.6\times$ (Tier 3)**!

---

## Slide 6: Finding 4 — When Does the Exact Math Split Help?

### Sweeping Data Variance $s \in [10^{-4}, 1]$
Does solving the linear part with exact calculus ($e^h$) always help?
* **On Concentrated Data ($s \to 0$, real image manifolds):**  
  $\epsilon(x, t) \approx x/\sigma$ is nearly constant. DPM-Solver-1 is nearly exact.  
  *At 20 NFE, DPM-1 beats Euler at 83% of points, and DPM-2 beats RK2 at 100%.*
* **On Spread-Out Data ($s = 1$, variance equals noise):**  
  The classical ODE right-hand side is identically zero! Classical Euler is exact to machine precision ($10^{-16}$), while DPM-Solver-1 has an error of $5.9 \times 10^{-3}$.
* **Takeaway:** DPM-Solver's exponential split is brilliant for natural images because image data lives on low-dimensional, concentrated manifolds.

---

## Slide 7: The Headline Result — Good-Looking is NOT Accurate!

### The M11 Kaggle Experiment (5,000 CIFAR-10 Images at 10 NFE)
We scored generated images two ways:
1. **FID (The Paper's Way):** Do the 5,000 pictures look realistic to an AI classifier?
2. **L2 Distance (Numerical Analysis Way):** How far is each image from the true 201-NFE ODE solution?

### The Result: An Exact Inversion (Rank Correlation: -1.00)
| Solver (at 10 NFE) | Visual Quality (FID) *(Lower = Better)* | Distance to True Answer (L2) *(Lower = Closer)* | Visual vs. Mathematical Outcome |
| :--- | :---: | :---: | :--- |
| **DPM-Solver-2** | **24.9 (Best Looking!)** | **180.6 (Furthest Away!)** | Sharp, crisp photo — but of a **different** object! |
| **DPM-Solver-fast**| 31.3 | 148.2 | Good visual quality, high trajectory drift. |
| **DDIM (quadratic)**| 39.5 | 98.4 | Balanced drift and visual realism. |
| **DPM-Solver-1** | 44.9 (Blurry) | **78.4 (Closest to Truth!)** | Faithful to the true ODE path, but softer pixels. |
| **DPM-Solver-3** | 143.3 (Static) | 279.6 (Failed) | Completely diverged at only 3 steps. |

### The Core Thesis:
> **"DPM-Solver makes good images in 10 steps" is TRUE (FID agrees).**  
> **"DPM-Solver solves the differential equation accurately in 10 steps" is FALSE.**  
> DPM-Solver-2 achieves low FID because it takes an inaccurate leap that happens to land on a sharp image. Evaluating diffusion samplers purely on image quality hides their numerical behavior — which is why they must also be audited as numerical ODE methods.
