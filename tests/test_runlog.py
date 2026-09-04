import math

import pytest

from src.runlog import COLUMNS, append_row, load


def _row(**over):
    base = dict(
        tier=1, testbed="gaussian", arm="C", solver="dpm3", order=3,
        kappa=100.0, h=0.05, nfe=60, err_l2=1.2e-6, diverged=False, seed=0,
    )
    base.update(over)
    return base


def test_roundtrip(tmp_path):
    csv_path = str(tmp_path / "r.csv")
    append_row(csv_path, **_row())
    append_row(csv_path, **_row(solver="dpm1", order=1, nfe=20, err_l2=3e-3))
    df = load(csv_path)
    assert list(df.columns) == list(COLUMNS)
    assert len(df) == 2
    assert df.loc[0, "solver"] == "dpm3"
    assert df.loc[1, "order"] == 1


def test_diverged_and_nan(tmp_path):
    csv_path = str(tmp_path / "r.csv")
    append_row(csv_path, **_row(diverged=True, nfe=float("nan"), err_l2=float("inf")))
    df = load(csv_path)
    assert bool(df.loc[0, "diverged"]) is True
    assert math.isnan(df.loc[0, "nfe"])
    assert math.isinf(df.loc[0, "err_l2"])


def test_rejects_bad_keys(tmp_path):
    csv_path = str(tmp_path / "r.csv")
    with pytest.raises(ValueError):
        append_row(csv_path, tier=1, testbed="x")  # missing most columns
    with pytest.raises(ValueError):
        append_row(csv_path, **_row(), kappaa=1.0)  # typo -> extra key
