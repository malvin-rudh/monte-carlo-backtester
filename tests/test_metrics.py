import numpy as np

from src.analysis.metrics import (
    compute_expected_return,
    compute_sharpe,
    compute_var,
    compute_volatility,
)

# synthetic returns: mean=0.15, std=0.07071..., sharpe=2.121..., 5th pct=0.054
RETURNS = np.array([0.05, 0.10, 0.15, 0.20, 0.25])


def test_metrics():
    expected_mean = float(np.mean(RETURNS))        # 0.15
    expected_std  = float(np.std(RETURNS))         # ~0.07071
    expected_sharpe = expected_mean / expected_std  # ~2.121
    expected_var = float(np.percentile(RETURNS, 5)) # ~0.054

    assert round(compute_expected_return(RETURNS), 10) == round(expected_mean, 10)
    assert round(compute_volatility(RETURNS), 10)      == round(expected_std, 10)
    assert round(compute_sharpe(RETURNS), 10)          == round(expected_sharpe, 10)
    assert round(compute_var(RETURNS), 10)             == round(expected_var, 10)
