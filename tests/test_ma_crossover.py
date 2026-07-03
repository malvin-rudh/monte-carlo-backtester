import numpy as np
import pandas as pd
import pytest

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
    # cost_bps=0 isolates the shift logic from the execution-day cost charge
    strategy_returns = compute_strategy_returns(prices, signals, cost_bps=0.0)

    # signal generated on day 60 must not affect returns until day 61
    # strategy_returns is shifted by 1, so index 60 reflects the signal from day 59 (still 0)
    assert strategy_returns.iloc[60] == 0.0


def test_compute_final_return():
    prices = _make_prices(100)
    # all-zero signals — never in market
    signals = pd.Series(np.zeros(100))
    strategy_returns = compute_strategy_returns(prices, signals)

    assert compute_final_return(strategy_returns) == 0.0


def test_transaction_costs_charged_per_flip():
    # constant prices — market P&L is exactly 0, isolating the cost drag
    prices = pd.Series(np.full(200, 100.0))
    # 4 flips: enter, exit, enter, exit
    signals = pd.Series(np.zeros(200))
    signals.iloc[60:100] = 1
    signals.iloc[140:180] = 1
    n_flips = 4
    cost_bps = 5.0
    cost = cost_bps / 10_000

    zero_cost = compute_strategy_returns(prices, signals, cost_bps=0.0)
    with_cost = compute_strategy_returns(prices, signals, cost_bps=cost_bps)

    # daily drag is exactly cost on each of the 4 flip days, 0 elsewhere
    drag = zero_cost - with_cost
    assert (drag[drag > 0] == cost).all()
    assert (drag > 0).sum() == n_flips

    # final return is n_flips × cost lower (compounding cross-term < 1e-6 at 5 bps)
    final_zero = compute_final_return(zero_cost)
    final_cost = compute_final_return(with_cost)
    assert final_cost == pytest.approx(final_zero - n_flips * cost, abs=2e-6)


def test_costs_do_not_disturb_lookahead_shift():
    # same setup as test_no_lookahead, but with costs on
    prices = pd.Series([100.0] * 60 + [200.0] * 40)
    signals = compute_signals(prices)
    strategy_returns = compute_strategy_returns(prices, signals, cost_bps=5.0)

    # crossover fires on day 60, trade executes day 61 — cost hits 61, never 60
    assert signals.loc[60] == 1
    assert strategy_returns.loc[60] == 0.0
    assert strategy_returns.loc[61] == -5.0 / 10_000
