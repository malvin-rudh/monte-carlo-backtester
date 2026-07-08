import numpy as np
import pandas as pd
import pytest

from src.strategy.cross_sectional_momentum import (
    compute_ranks,
    compute_strategy_returns,
    compute_weights,
)

TICKERS = list("ABCDEFG")


def _make_ordered_panel(n: int = 60) -> pd.DataFrame:
    """Return a 7-ticker panel where ticker k earns a constant k×0.1% per day — a fixed pecking order."""
    data = {ticker: np.full(n, k * 0.001) for k, ticker in enumerate(TICKERS)}
    return pd.DataFrame(data)


def test_known_relative_performance():
    panel = _make_ordered_panel()
    ranks = compute_ranks(panel, lookback=10)
    weights = compute_weights(ranks, n_long=2, n_short=2)

    # warm-up rows stay flat
    assert (weights.iloc[:10] == 0).all().all()

    # after warm-up: long the two best (F, G), short the two worst (A, B), flat in between
    row = weights.iloc[20]
    assert row["F"] == 0.25 and row["G"] == 0.25
    assert row["A"] == -0.25 and row["B"] == -0.25
    assert (row[["C", "D", "E"]] == 0).all()

    # dollar-neutral with gross exposure 1
    assert row.sum() == pytest.approx(0.0)
    assert row.abs().sum() == pytest.approx(1.0)


def test_no_lookahead():
    # all returns zero except ticker A jumps on day 40 — first possible exposure is day 41
    panel = pd.DataFrame(np.zeros((60, 7)), columns=TICKERS)
    panel.loc[40, "A"] = 0.10

    ranks = compute_ranks(panel, lookback=10)
    weights = compute_weights(ranks, n_long=1, n_short=0)
    strategy_returns = compute_strategy_returns(panel, weights, cost_bps=5.0)

    # ranking reacts on day 40, but nothing (return or cost) may hit day 40
    assert weights.loc[40, "A"] == 0.5
    assert strategy_returns.loc[40] == 0.0
    # day 41 carries the entry cost for the 0.5 position opened at day 40's close
    assert strategy_returns.loc[41] == pytest.approx(-0.5 * 5.0 / 10_000)


def test_cost_only_on_rebalance_days():
    # fixed pecking order — weights change once (initial entry), then never again
    panel = _make_ordered_panel()
    ranks = compute_ranks(panel, lookback=10)
    weights = compute_weights(ranks, n_long=2, n_short=2)

    zero_cost = compute_strategy_returns(panel, weights, cost_bps=0.0)
    with_cost = compute_strategy_returns(panel, weights, cost_bps=5.0)

    drag = zero_cost - with_cost
    charged_days = drag[drag > 0]

    # exactly one charge: the execution day of the initial entry (weights set day 10, executed day 11)
    assert len(charged_days) == 1
    assert charged_days.index[0] == 11
    # turnover at entry is |Δw| = 1.0 (0.5 long + 0.5 short), so the charge is the full 5 bps
    assert charged_days.iloc[0] == pytest.approx(5.0 / 10_000)


def test_rebalance_freq_holds_weights():
    # noisy panel so the daily ranking genuinely changes
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(rng.normal(0, 0.02, size=(60, 7)), columns=TICKERS)

    ranks = compute_ranks(panel, lookback=10)
    weights = compute_weights(ranks, n_long=2, n_short=2, rebalance_freq=5)

    # between rebalance days, weights must equal the last rebalance day's weights
    for t in range(len(weights)):
        last_rebalance = t - t % 5
        pd.testing.assert_series_equal(weights.iloc[t], weights.iloc[last_rebalance], check_names=False)
