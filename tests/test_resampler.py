import numpy as np
import pandas as pd

from src.simulation.resampler import bootstrap_paths, compute_returns


def _make_prices() -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with a known Close sequence."""
    close = [100.0, 110.0, 99.0, 103.95]
    idx = pd.date_range("2020-01-01", periods=len(close), freq="B")
    df = pd.DataFrame({"Close": close}, index=idx)
    return df


def test_compute_returns():
    df = _make_prices()
    returns = compute_returns(df)

    assert len(returns) == len(df) - 1
    assert returns.isna().sum() == 0

    expected = [0.100, -0.100, 0.050]
    for actual, exp in zip(returns, expected):
        assert round(actual, 3) == exp


def test_bootstrap_paths():
    # use a short returns Series as the sampling pool
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005])

    matrix = bootstrap_paths(returns, n_sims=100, n_days=50)

    assert matrix.shape == (100, 50)
    assert not np.isnan(matrix).any()
    # every path must open at exactly $100
    assert (matrix[:, 0] == 100.0).all()
