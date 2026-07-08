import numpy as np
import pandas as pd

from src.simulation.resampler import bootstrap_paths, bootstrap_paths_multi, compute_returns


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

    matrix = bootstrap_paths(returns, n_sims=100, n_days=50, method="iid")

    assert matrix.shape == (100, 50)
    assert not np.isnan(matrix).any()
    # every path must open at exactly $100
    assert (matrix[:, 0] == 100.0).all()


def test_block_bootstrap_paths_shape():
    returns = _make_autocorrelated_returns()

    matrix = bootstrap_paths(returns, n_sims=100, n_days=50, method="block", block_length=15)

    assert matrix.shape == (100, 50)
    assert not np.isnan(matrix).any()
    assert (matrix[:, 0] == 100.0).all()


def _make_autocorrelated_returns(n: int = 2000, phi: float = 0.9) -> pd.Series:
    """Return an AR(1) daily-return Series with strong, known lag-1 autocorrelation."""
    rng = np.random.default_rng(0)
    shocks = rng.normal(0, 0.01, n)
    returns = np.zeros(n)
    for t in range(1, n):
        returns[t] = phi * returns[t - 1] + shocks[t]
    return pd.Series(returns)


def _mean_lag1_autocorr(price_matrix: np.ndarray) -> float:
    """Mean lag-1 autocorrelation of each simulated path's daily returns."""
    path_returns = np.diff(price_matrix, axis=1) / price_matrix[:, :-1]
    autocorrs = [
        np.corrcoef(r[:-1], r[1:])[0, 1] for r in path_returns
    ]
    return float(np.mean(autocorrs))


def test_block_bootstrap_preserves_autocorrelation():
    # AR(1) pool: block sampling should keep the serial dependence that iid destroys
    returns = _make_autocorrelated_returns()

    iid_matrix = bootstrap_paths(returns, n_sims=200, n_days=252, method="iid")
    block_matrix = bootstrap_paths(
        returns, n_sims=200, n_days=252, method="block", block_length=15
    )

    iid_ac = _mean_lag1_autocorr(iid_matrix)
    block_ac = _mean_lag1_autocorr(block_matrix)

    # iid scrambling leaves no measurable structure
    assert abs(iid_ac) < 0.1
    # block paths retain most of the pool's phi=0.9 dependence (join points dilute it slightly)
    assert block_ac > 0.5
    assert block_ac > iid_ac + 0.4


def test_multi_bootstrap_shape():
    rng = np.random.default_rng(1)
    panel = pd.DataFrame(rng.normal(0, 0.01, size=(500, 7)))

    tensor = bootstrap_paths_multi(panel, n_sims=50, n_days=60, block_length=15)

    assert tensor.shape == (50, 60, 7)
    assert not np.isnan(tensor).any()
    # every ticker of every path must open at exactly $100
    assert (tensor[:, 0, :] == 100.0).all()


def test_multi_bootstrap_shared_dates_invariant():
    # sentinel returns: date d, ticker k → 1e-6·d + 1e-3·k, so every cell decodes back to its date
    n_dates, n_tickers = 100, 7
    d = np.arange(n_dates)[:, None]
    k = np.arange(n_tickers)[None, :]
    panel = pd.DataFrame(1e-6 * d + 1e-3 * k)

    tensor = bootstrap_paths_multi(panel, n_sims=20, n_days=60, block_length=15)

    # recover sampled returns from prices, strip the ticker offset, decode the historical date
    recovered = tensor[:, 1:, :] / tensor[:, :-1, :] - 1
    decoded = (recovered - 1e-3 * np.arange(n_tickers)) / 1e-6
    dates = np.rint(decoded)

    assert np.allclose(decoded, dates, atol=1e-3)
    # design invariant: all 7 tickers on simulation i, day t come from the SAME original date
    assert (dates == dates[:, :, :1]).all()
