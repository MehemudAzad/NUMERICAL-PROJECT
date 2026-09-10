"""M10 -- guards against the report going stale.

`notebooks/10_report.ipynb` builds every report table out of `results/*.csv`,
so the failure mode it needs protection from is not a wrong number but a
*missing or malformed input*: a sweep re-run that drops a column, a CSV that
never got committed, a solver quietly disappearing from a tier. Those would
silently shrink the master order table rather than raise anything.

Two kinds of check live here:

* **inputs** -- every committed results CSV exists and conforms exactly to
  ``src.runlog.COLUMNS``, and each tier covers the arms and solvers the report
  claims it does;
* **outputs** -- the artifacts notebook 10 produces, checked only if present, so
  this file is green both before the notebook has been run (as in ``run_all.sh``,
  where pytest runs first) and after.

Tier 3 is optional throughout: it comes from a Kaggle GPU run, and the analytic
tiers must stay verifiable on a laptop without it.
"""

import pathlib

import pandas as pd
import pytest

from src.runlog import COLUMNS

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Committed by milestones M5-M8; these must always be present.
REQUIRED_CSVS = [
    "tier1_order.csv",
    "tier2_order.csv",
    "stability_envelope.csv",
    "crossover_sweep.csv",
]

# Produced by M9 on Kaggle. Optional -- a laptop clone is still valid without it.
TIER3_CSVS = ["tier3_error.csv", "tier3_decomposition.csv"]

# What each convergence sweep is claimed to cover, in the report's order table.
EXPECTED_COVERAGE = {
    "tier1_order.csv": {
        "A": {"euler", "midpoint", "heun3", "rk4", "ab2"},
        "B": {"euler", "midpoint", "heun3", "rk4", "ab2"},
        "C": {"ddim", "dpm1", "dpm2", "dpm3"},
    },
    "tier2_order.csv": {
        "A": {"euler", "midpoint", "heun3", "rk4", "ab2"},
        "B": {"euler", "midpoint", "heun3", "rk4", "ab2"},
        "C": {"ddim", "dpm1", "dpm2", "dpm3"},
    },
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


# --- inputs -----------------------------------------------------------------


@pytest.mark.parametrize("name", REQUIRED_CSVS)
def test_required_results_csv_exists(name):
    assert (RESULTS / name).exists(), (
        f"results/{name} is missing -- re-run the notebook that produces it "
        "before building the report"
    )


@pytest.mark.parametrize("name", REQUIRED_CSVS + TIER3_CSVS)
def test_results_csv_conforms_to_the_schema(name):
    """One schema for the whole project, so every table concatenates (step 0)."""
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not present (Tier 3 is produced on Kaggle)")
    df = pd.read_csv(path)
    if name == "tier3_decomposition.csv":
        # Not a sweep: a three-row-per-component summary, its own shape.
        assert list(df.columns) == ["component", "setting", "value"]
        assert len(df) > 0
        return
    assert tuple(df.columns) == COLUMNS, f"{name} has columns {tuple(df.columns)}"
    assert len(df) > 0
    if name == "stability_envelope.csv":
        # M6 records a stability *limit*, not an error sweep: `h` carries the
        # measurement and err_l2/nfe are nan by design.
        assert df["h"].notna().all(), "the stability envelope has no h values"
    else:
        assert df["err_l2"].notna().any(), f"{name} has no usable error values"


@pytest.mark.parametrize("name", sorted(EXPECTED_COVERAGE))
def test_convergence_sweeps_cover_every_claimed_solver(name):
    """A solver silently vanishing from a sweep would shrink the report's table."""
    df = _read(name)
    for arm, solvers in EXPECTED_COVERAGE[name].items():
        got = set(df[df["arm"] == arm]["solver"].unique())
        assert solvers <= got, f"{name} arm {arm} is missing {sorted(solvers - got)}"


def test_every_sweep_row_has_at_least_four_points_to_fit():
    """`fit_order`'s default min_pts is 4; fewer means a nan slope in the table."""
    for name in ["tier1_order.csv", "tier2_order.csv"]:
        df = _read(name)
        counts = df.groupby(["arm", "solver", "kappa"], dropna=False).size()
        assert counts.min() >= 4, f"{name}: a sweep has only {counts.min()} points"


def test_crossover_sweep_has_both_orders_in_every_group():
    """`crossover_h` needs a dpm1 curve and a dpm3 curve at matched h."""
    df = _read("crossover_sweep.csv")
    for key, g in df.groupby(["tier", "testbed", "kappa"], dropna=False):
        got = set(g["solver"].unique())
        assert {"dpm1", "dpm3"} <= got, f"crossover group {key} has only {got}"
        h1 = set(g[g["solver"] == "dpm1"]["h"].round(12))
        h3 = set(g[g["solver"] == "dpm3"]["h"].round(12))
        assert h1 == h3, f"crossover group {key} is not swept at matched h"


def test_tier3_uses_the_tier3_label_and_one_testbed():
    path = RESULTS / "tier3_error.csv"
    if not path.exists():
        pytest.skip("tier3_error.csv not present (produced on Kaggle by notebook 09)")
    df = pd.read_csv(path)
    assert set(df["tier"].unique()) == {3}
    assert set(df["arm"].unique()) == {"A", "B", "C"}, "Tier 3 must cover all three arms"


# --- outputs ----------------------------------------------------------------


def test_master_order_table_is_complete_if_built():
    path = RESULTS / "master_order_table.csv"
    if not path.exists():
        pytest.skip("master_order_table.csv not built yet -- run notebooks/10_report.ipynb")
    m = pd.read_csv(path)
    for name, tier in [("tier1_order.csv", 1), ("tier2_order.csv", 2)]:
        src = _read(name)
        expected = set(map(tuple, src[["arm", "solver"]].drop_duplicates().to_numpy()))
        got = set(map(tuple, m[m["tier"] == tier][["arm", "solver"]].to_numpy()))
        assert expected <= got, f"tier {tier} missing {sorted(expected - got)}"
    assert m["measured_slope"].notna().all(), "a fit produced nan -- check its sweep range"


def test_ledger_and_coverage_are_built_together():
    led = RESULTS / "predictions_ledger.csv"
    cov = RESULTS / "proposal_coverage.csv"
    if not led.exists() and not cov.exists():
        pytest.skip("report tables not built yet -- run notebooks/10_report.ipynb")
    assert led.exists() and cov.exists(), "the ledger and the coverage table ship together"

    d_led = pd.read_csv(led)
    assert list(d_led.columns) == ["prediction", "verdict", "evidence"]
    assert set(d_led["verdict"]) <= {"confirmed", "refuted", "narrowed", "pending"}
    assert d_led["evidence"].str.len().min() > 20, "every verdict needs its number"

    d_cov = pd.read_csv(cov)
    assert list(d_cov.columns) == ["bucket", "item", "note"]
    assert set(d_cov["bucket"]) <= {
        "delivered", "substituted", "dropped", "added", "pending",
    }
    for bucket in ["delivered", "substituted", "dropped", "added"]:
        assert (d_cov["bucket"] == bucket).any(), f"coverage table has no '{bucket}' rows"


def test_report_figures_exist_if_built():
    figs = ["10_error_vs_nfe_all.png", "10_efficiency_frontier.png"]
    if not any((FIGURES / f).exists() for f in figs):
        pytest.skip("report figures not built yet -- run notebooks/10_report.ipynb")
    for f in figs:
        assert (FIGURES / f).exists(), f"figures/{f} missing"


def test_milestone_figures_are_committed():
    """Every earlier milestone's figure is a report deliverable in its own right."""
    for f in [
        "05_error_vs_h.png",
        "05_error_vs_nfe.png",
        "06_stability_envelope.png",
        "07_error_vs_h.png",
        "07_error_vs_nfe.png",
        "08_crossover.png",
    ]:
        assert (FIGURES / f).exists(), f"figures/{f} missing"
