import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily returns from the Close column of an OHLCV DataFrame."""
    returns = df["Close"].pct_change()

    # drop the first NaN produced by pct_change
    returns = returns.dropna()

    return returns


BASE_PRICE = 100.0


def bootstrap_paths(
    returns: pd.Series, n_sims: int, n_days: int, seed: int = 42
) -> np.ndarray:
    """Sample returns with replacement to produce a (n_sims, n_days) price matrix starting at $100."""
    rng = np.random.default_rng(seed)

    # sample n_days-1 returns so column 0 can be the $100 anchor
    sampled = rng.choice(returns.values, size=(n_sims, n_days - 1), replace=True)

    # compound sampled returns into prices after day 0
    compounded = BASE_PRICE * np.cumprod(1 + sampled, axis=1)

    # prepend $100 starting column
    start_col = np.full((n_sims, 1), BASE_PRICE)
    price_matrix = np.hstack([start_col, compounded])

    return price_matrix
