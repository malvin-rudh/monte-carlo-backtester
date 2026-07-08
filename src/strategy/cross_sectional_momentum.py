import numpy as np
import pandas as pd


def compute_ranks(returns_panel: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Rank tickers cross-sectionally by trailing lookback-day compounded return (1 = worst performer)."""
    cumulative = (1 + returns_panel).cumprod()

    # trailing return over exactly lookback days, causal — row t uses returns up to and including t
    trailing = cumulative / cumulative.shift(lookback) - 1

    # warm-up rows rank as NaN, which compute_weights treats as flat
    return trailing.rank(axis=1)


def compute_weights(
    ranks: pd.DataFrame, n_long: int = 2, n_short: int = 2, rebalance_freq: int = 1
) -> pd.DataFrame:
    """Build dollar-neutral weights: half of capital long the top n_long ranks, half short the bottom n_short."""
    n_names = ranks.shape[1]

    # each leg gets 0.5 of capital, equal-weighted within the leg — gross exposure 1, net 0
    # NaN ranks (warm-up) compare False on both sides → flat
    weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    if n_long:
        weights += (ranks > n_names - n_long) * (0.5 / n_long)
    if n_short:
        weights -= (ranks <= n_short) * (0.5 / n_short)

    # hold weights between rebalance days
    if rebalance_freq > 1:
        is_rebalance = pd.Series(
            np.arange(len(weights)) % rebalance_freq == 0, index=weights.index
        )
        weights = weights.where(is_rebalance).ffill().fillna(0)

    return weights


def compute_strategy_returns(
    returns_panel: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 5.0
) -> pd.Series:
    """Apply shifted weights to returns, minus turnover-scaled costs on each execution day."""
    # shift by 1 — today's position comes from yesterday's ranking
    shifted = weights.shift(1).fillna(0)

    gross = (shifted * returns_panel).sum(axis=1)

    # cost scales with traded notional: sum of |Δweight| across names on the execution day
    turnover = shifted.diff().abs().sum(axis=1).fillna(0)
    return gross - turnover * (cost_bps / 10_000)
