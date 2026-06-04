import numpy as np
import pandas as pd

from src.strategy.ma_crossover import (
    compute_final_return,
    compute_signals,
    compute_strategy_returns,
)


def _make_prices(n: int = 100) -> pd.Series:
    """Return a synthetic price Series with a gentle upward trend."""
    return pd.Series(np.linspace(100, 150, n))


def test_compute_signals():
    prices = _make_prices(100)
    signals = compute_signals(prices)

    assert len(signals) == len(prices)
    assert (signals.iloc[:49] == 0).all()
    assert signals.isin([0, 1]).all()


def test_no_lookahead():
    # flat price until day 59, then spike — crossover occurs on day 60
    prices = pd.Series([100.0] * 60 + [200.0] * 40)
    signals = compute_signals(prices)
    strategy_returns = compute_strategy_returns(prices, signals)

    # signal generated on day 60 must not affect returns until day 61
    # strategy_returns is shifted by 1, so index 60 reflects the signal from day 59 (still 0)
    assert strategy_returns.iloc[60] == 0.0


def test_compute_final_return():
    prices = _make_prices(100)
    # all-zero signals — never in market
    signals = pd.Series(np.zeros(100))
    strategy_returns = compute_strategy_returns(prices, signals)

    assert compute_final_return(strategy_returns) == 0.0
