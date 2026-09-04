# third_party/

## `dpm_solver_pytorch.py`

Taken **unmodified** from <https://github.com/LuChengTHU/dpm-solver> at commit
`8acf2bb9419165fec8b3ffd01621aa735be5dee3` (2023-12-06), MIT licence — see
`LICENSE-dpm-solver`. The upstream README explicitly instructs users to copy this
single self-contained file into their own project, which is what we have done.

### How we use it

This is the **arm-C reference implementation**. We call it with an *analytic noise
oracle* (an exact closed-form `ε(x, t)` written by us) instead of a neural
network, so that Tiers 1 and 2 test the published code with **zero model error** —
only its discretization behaviour.

We never edit this file. If we ever need to patch it (e.g. to log intermediate
states for the crossover study), we will copy it to `src/dpm_solver_patched.py`,
leave this original untouched, and diff the two in the report.

### Pinning

The commit hash is fixed so results stay reproducible even if upstream changes.
To re-fetch:

```bash
PIN=8acf2bb9419165fec8b3ffd01621aa735be5dee3
curl -sS -o third_party/dpm_solver_pytorch.py \
  "https://raw.githubusercontent.com/LuChengTHU/dpm-solver/$PIN/dpm_solver_pytorch.py"
curl -sS -o third_party/LICENSE-dpm-solver \
  "https://raw.githubusercontent.com/LuChengTHU/dpm-solver/$PIN/LICENSE"
```
