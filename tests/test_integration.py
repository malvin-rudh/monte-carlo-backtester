import numpy as np
import pandas as pd

from src.data.fetcher import get_prices, handle_missing
from src.simulation.resampler import bootstrap_paths, compute_returns
from src.strategy.ma_crossover import (
    compute_final_return,
    compute_signals,
    compute_strategy_returns,
)
from src.analysis.metrics import (
    compute_expected_return,
    compute_volatility,
    compute_sharpe,
    compute_var,
)

TICKER = "AAPL"
START = "2020-01-01"
END = "2023-01-01"
N_SIMS = 100
N_DAYS = 252
FAST = 20
SLOW = 50


def test_full_pipeline():
    # fetch and clean data
    df = get_prices(TICKER, START, END)
    df = handle_missing(df)

    # build price matrix from bootstrapped returns
    returns = compute_returns(df)
    price_matrix = bootstrap_paths(returns, N_SIMS, N_DAYS)

    # run strategy across all simulations
    final_returns = []
    for i in range(N_SIMS):
        prices = pd.Series(price_matrix[i])
        signals = compute_signals(prices, fast=FAST, slow=SLOW)
        strat_returns = compute_strategy_returns(prices, signals)
        final_returns.append(compute_final_return(strat_returns))

    arr = np.array(final_returns)

    # array shape and type checks
    assert len(arr) == N_SIMS
    assert all(isinstance(v, float) for v in final_returns)
    assert not np.isnan(arr).any()

    # metrics are all scalar floats
    assert isinstance(compute_expected_return(arr), float)
    assert isinstance(compute_volatility(arr), float)
    assert isinstance(compute_sharpe(arr), float)
    assert isinstance(compute_var(arr), float)
