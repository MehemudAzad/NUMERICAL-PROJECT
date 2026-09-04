# Numerical-project

### User

how should I start the coding? should I fork theirs or start a new repo from the start

---

### Assistant

**New repo. Vendor the one file you need.** Don't fork.

The reasoning is practical. Their repo is mostly things you'll never touch — `examples/stable-diffusion`, `examples/score_sde_jax`, `examples/ddpm_and_guided-diffusion`, JAX implementations. The part you actually use is a single self-contained file, `dpm_solver_pytorch.py`, which the README explicitly tells you to copy into your own project. That's the intended usage.

Three more reasons that matter for a graded course project:

**Your commit history is evidence.** A fork opens with 77 commits that aren't yours. In your own repo, every commit is your team's work, which is exactly what an examiner wants to see when five people claim to have contributed.

**You're not modifying their solver.** A fork makes sense when you're changing upstream code and might send a PR back. You're wrapping it and comparing against it. Nothing flows upstream.

**Your project isn't shaped like theirs.** You need `testbeds.py`, `solvers.py` for arms A and B, a stability bisector, an order fitter. None of that has a home in their tree. You'd be building your project in a stranger's directory structure.

Here's the setup:

```bash
mkdir cse402-solvers && cd cse402-solvers
git init

python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib pandas pytest torch
pip freeze > requirements.txt

mkdir -p src tests results figures third_party

# the one file you need, pinned to a specific commit
curl -o third_party/dpm_solver_pytorch.py \
  https://raw.githubusercontent.com/LuChengTHU/dpm-solver/52bde38e24398476f8eb1e0760c81346c6a00812/dpm_solver_pytorch.py

touch src/__init__.py tests/__init__.py
git add -A && git commit -m "scaffold: env, dirs, vendored DPM-Solver"
```

Note the commit hash in that URL rather than `main`. If upstream changes, your results stay reproducible — and you can state the exact version in your report.

Handle the licence properly, because it's free marks and takes two minutes. MIT permits vendoring but requires the copyright notice travel with the code:

```bash
curl -o third_party/LICENSE-dpm-solver \
  https://raw.githubusercontent.com/LuChengTHU/dpm-solver/main/LICENSE
```

Then a short `third_party/README.md`:

```markdown
`dpm_solver_pytorch.py` is taken unmodified from
github.com/LuChengTHU/dpm-solver at commit 52bde38,
MIT licence (see LICENSE-dpm-solver).

Used as the arm-C reference implementation. We call it with an
analytic noise oracle instead of a neural network so that Tiers 1
and 2 test the published code with zero model error.
```

That paragraph also happens to be your methods section. If you later need to patch their file — say, to log intermediate states for the crossover study — copy it to `src/dpm_solver_patched.py`, leave the original untouched, and diff the two in your report. A reviewer can then see precisely what you changed and trust that arm C is otherwise the real thing.

For branches, keep it simple with five people: `main` stays green, each workstream works on `ws1-testbeds`, `ws2-solvers`, `ws3-metrics`, `ws4-tier3`, and merges when its tests pass. The reviewer in each pair approves. Nobody pushes broken tests to `main` — Gate G1 lives there.

---

