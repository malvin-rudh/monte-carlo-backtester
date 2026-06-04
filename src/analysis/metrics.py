import numpy as np


def compute_expected_return(final_returns: np.ndarray) -> float:
    """Return the mean of the simulated final returns."""
    return float(np.mean(final_returns))


def compute_volatility(final_returns: np.ndarray) -> float:
    """Return the standard deviation of the simulated final returns."""
    return float(np.std(final_returns))


def compute_sharpe(final_returns: np.ndarray) -> float:
    """Return mean divided by std of final returns — return per unit of cross-scenario variability."""
    return float(np.mean(final_returns) / np.std(final_returns))


def compute_var(final_returns: np.ndarray, percentile: int = 5) -> float:
    """Return the given percentile of final returns as a Value at Risk estimate."""
    return float(np.percentile(final_returns, percentile))


def print_summary(final_returns: np.ndarray) -> None:
    """Print a formatted summary of all four metrics across the simulated return distribution."""
    exp_ret = compute_expected_return(final_returns)
    vol = compute_volatility(final_returns)
    sharpe = compute_sharpe(final_returns)
    var = compute_var(final_returns)

    print("=" * 35)
    print("  Monte Carlo Simulation Summary")
    print("=" * 35)
    print(f"  Expected Return : {exp_ret * 100:>8.2f}%")
    print(f"  Volatility      : {vol * 100:>8.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:>9.3f}")
    print(f"  VaR (95%)       : {var * 100:>8.2f}%")
    print("=" * 35)
