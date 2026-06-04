import argparse

import numpy as np
import pandas as pd

from src.analysis.metrics import print_summary
from src.data.fetcher import get_prices, load_from_csv, save_to_csv
from src.simulation.resampler import bootstrap_paths, compute_returns
from src.strategy.ma_crossover import (
    compute_final_return,
    compute_signals,
    compute_strategy_returns,
)


def run(
    ticker: str,
    start: str,
    end: str,
    n_sims: int,
    n_days: int,
    fast: int,
    slow: int,
) -> dict:
    """Run the full Monte Carlo backtest pipeline and return a dict of summary metrics."""
    # fetch, persist, and reload price data
    df = get_prices(ticker, start, end)
    save_to_csv(df, ticker)
    df = load_from_csv(ticker)

    # build the return pool and generate simulated price paths
    returns = compute_returns(df)
    price_matrix = bootstrap_paths(returns, n_sims, n_days)

    # run strategy on each simulated path
    final_returns = []
    for i in range(n_sims):
        path = price_matrix[i]
        prices_series = pd.Series(path)

        signals = compute_signals(prices_series, fast=fast, slow=slow)
        strategy_returns = compute_strategy_returns(prices_series, signals)
        final_returns.append(compute_final_return(strategy_returns))

    final_returns_array = np.array(final_returns)
    print_summary(final_returns_array)

    return {
        "expected_return": float(np.mean(final_returns_array)),
        "volatility": float(np.std(final_returns_array)),
        "sharpe": float(np.mean(final_returns_array) / np.std(final_returns_array)),
        "var_95": float(np.percentile(final_returns_array, 5)),
    }


def main() -> None:
    """Parse CLI arguments and run the Monte Carlo backtest pipeline."""
    parser = argparse.ArgumentParser(description="Monte Carlo strategy backtester")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start",  required=True)
    parser.add_argument("--end",    required=True)
    parser.add_argument("--n_sims", type=int, default=1000)
    parser.add_argument("--n_days", type=int, default=252)
    parser.add_argument("--fast",   type=int, default=20)
    parser.add_argument("--slow",   type=int, default=50)
    args = parser.parse_args()

    run(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        n_sims=args.n_sims,
        n_days=args.n_days,
        fast=args.fast,
        slow=args.slow,
    )


if __name__ == "__main__":
    main()
