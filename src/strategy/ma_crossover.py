import pandas as pd

WARM_UP = 49


def compute_signals(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Return a long-only signal Series (1=in market, 0=cash) based on fast/slow MA crossover."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    # 1 when fast MA is above slow MA, 0 otherwise
    signals = (fast_ma > slow_ma).astype(int)

    # zero out warm-up period — no valid slow MA until day 49
    signals.iloc[:WARM_UP] = 0

    return signals


def compute_strategy_returns(
    prices: pd.Series, signals: pd.Series, cost_bps: float = 5.0
) -> pd.Series:
    """Multiply shifted signals by daily returns, minus a round-trip cost on each signal flip."""
    daily_returns = prices.pct_change()

    # shift by 1 — today's signal is based on yesterday's crossover
    shifted_signals = signals.shift(1).fillna(0)

    strategy_returns = shifted_signals * daily_returns

    # charge round-trip cost on the execution day of each flip, after the shift
    trades = (shifted_signals.diff().fillna(0) != 0).astype(float)
    strategy_returns = strategy_returns - trades * (cost_bps / 10_000)

    # drop the first NaN from pct_change
    return strategy_returns.dropna()


def compute_final_return(strategy_returns: pd.Series) -> float:
    """Compound daily strategy returns and return the final cumulative return as a scalar."""
    return float((1 + strategy_returns).cumprod().iloc[-1] - 1)
