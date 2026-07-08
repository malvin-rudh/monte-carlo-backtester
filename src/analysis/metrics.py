from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"


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


def print_comparison(results: dict[str, np.ndarray]) -> None:
    """Print expected return, volatility, Sharpe, and VaR side by side, one column per strategy."""
    col_w = max(max(len(name) for name in results), 10) + 3

    header = "  " + "Metric".ljust(18) + "".join(name.rjust(col_w) for name in results)
    width = max(len(header) + 2, 35)

    print("=" * width)
    print("  Monte Carlo Strategy Comparison")
    print("=" * width)
    print(header)
    for label, fn, as_pct in [
        ("Expected Return", compute_expected_return, True),
        ("Volatility", compute_volatility, True),
        ("Sharpe Ratio", compute_sharpe, False),
        ("VaR (95%)", compute_var, True),
    ]:
        cells = ""
        for arr in results.values():
            value = fn(arr)
            cell = f"{value * 100:.2f}%" if as_pct else f"{value:.3f}"
            cells += cell.rjust(col_w)
        print("  " + label.ljust(18) + cells)
    print("=" * width)


def plot_return_distribution(final_returns: np.ndarray) -> None:
    """Plot a histogram of final returns with a VaR 95% line and save to outputs/return_distribution.png."""
    var = compute_var(final_returns)

    fig, ax = plt.subplots()
    ax.hist(final_returns * 100, bins=50, edgecolor="white")

    # VaR threshold marker
    ax.axvline(var * 100, color="red", linestyle="--", label=f"VaR 95%: {var * 100:.2f}%")

    ax.set_title("Monte Carlo Return Distribution")
    ax.set_xlabel("Final Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUTS_DIR / "return_distribution.png")
    plt.close(fig)


def plot_sample_paths(price_matrix: np.ndarray, n_sample: int = 100) -> None:
    """Plot a fan chart of sampled price paths with a bold median line and save to outputs/sample_paths.png."""
    rng = np.random.default_rng(42)

    # randomly select n_sample row indices without replacement
    n_sims = price_matrix.shape[0]
    indices = rng.choice(n_sims, size=min(n_sample, n_sims), replace=False)

    fig, ax = plt.subplots()

    for idx in indices:
        ax.plot(price_matrix[idx], color="steelblue", alpha=0.1, linewidth=0.8)

    # median path across all simulations, not just the sample
    median_path = np.median(price_matrix, axis=0)
    ax.plot(median_path, color="black", linewidth=2, label="Median path")

    ax.set_title("Simulated Price Paths")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price ($)")
    ax.legend()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUTS_DIR / "sample_paths.png")
    plt.close(fig)
