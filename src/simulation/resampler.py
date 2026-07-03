import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily returns from the Close column of an OHLCV DataFrame."""
    returns = df["Close"].pct_change()

    # drop the first NaN produced by pct_change
    returns = returns.dropna()

    return returns


BASE_PRICE = 100.0


def _sample_iid(rng: np.random.Generator, values: np.ndarray, n_sims: int, n_needed: int) -> np.ndarray:
    """Sample single days independently with replacement — destroys autocorrelation."""
    return rng.choice(values, size=(n_sims, n_needed), replace=True)


def _sample_block(
    rng: np.random.Generator, values: np.ndarray, n_sims: int, n_needed: int, block_length: int
) -> np.ndarray:
    """Sample contiguous blocks of block_length days and concatenate — preserves short-range serial dependence."""
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    if block_length > len(values):
        raise ValueError("block_length exceeds the historical return pool")

    n_blocks = -(-n_needed // block_length)  # ceil division

    # each block starts anywhere a full block fits
    starts = rng.integers(0, len(values) - block_length + 1, size=(n_sims, n_blocks))

    # expand starts into contiguous index runs, trim final block to n_needed
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets).reshape(n_sims, -1)[:, :n_needed]

    return values[indices]


def bootstrap_paths(
    returns: pd.Series,
    n_sims: int,
    n_days: int,
    seed: int = 42,
    method: str = "block",
    block_length: int = 15,
) -> np.ndarray:
    """Resample returns ("block" or "iid") into a (n_sims, n_days) price matrix starting at $100."""
    rng = np.random.default_rng(seed)

    # sample n_days-1 returns so column 0 can be the $100 anchor
    n_needed = n_days - 1
    if method == "iid":
        sampled = _sample_iid(rng, returns.values, n_sims, n_needed)
    elif method == "block":
        sampled = _sample_block(rng, returns.values, n_sims, n_needed, block_length)
    else:
        raise ValueError(f"unknown bootstrap method: {method!r}")

    # compound sampled returns into prices after day 0
    compounded = BASE_PRICE * np.cumprod(1 + sampled, axis=1)

    # prepend $100 starting column
    start_col = np.full((n_sims, 1), BASE_PRICE)
    price_matrix = np.hstack([start_col, compounded])

    return price_matrix
