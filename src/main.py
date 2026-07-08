import argparse

import numpy as np
import pandas as pd

from src.analysis.metrics import (
    plot_return_distribution,
    plot_sample_paths,
    print_comparison,
    print_summary,
)
from src.data.fetcher import MAG7, get_prices, get_returns_panel, load_from_csv, save_to_csv
from src.simulation.resampler import bootstrap_paths, bootstrap_paths_multi, compute_returns
from src.strategy.cross_sectional_momentum import (
    compute_ranks,
    compute_weights,
)
from src.strategy.cross_sectional_momentum import (
    compute_strategy_returns as compute_xs_returns,
)
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
    bootstrap_method: str = "block",
    block_length: int = 15,
    cost_bps: float = 5.0,
) -> dict:
    """Run the full Monte Carlo backtest pipeline and return a dict of summary metrics."""
    # fetch, persist, and reload price data
    df = get_prices(ticker, start, end)
    save_to_csv(df, ticker)
    df = load_from_csv(ticker)

    # build the return pool and generate simulated price paths
    returns = compute_returns(df)
    price_matrix = bootstrap_paths(
        returns, n_sims, n_days, method=bootstrap_method, block_length=block_length
    )

    # run strategy on each simulated path
    final_returns = []
    for i in range(n_sims):
        path = price_matrix[i]
        prices_series = pd.Series(path)

        signals = compute_signals(prices_series, fast=fast, slow=slow)
        strategy_returns = compute_strategy_returns(prices_series, signals, cost_bps=cost_bps)
        final_returns.append(compute_final_return(strategy_returns))

    final_returns_array = np.array(final_returns)
    print_summary(final_returns_array)
    plot_return_distribution(final_returns_array)
    plot_sample_paths(price_matrix)

    return {
        "expected_return": float(np.mean(final_returns_array)),
        "volatility": float(np.std(final_returns_array)),
        "sharpe": float(np.mean(final_returns_array) / np.std(final_returns_array)),
        "var_95": float(np.percentile(final_returns_array, 5)),
    }


def run_multi(
    tickers: list[str],
    start: str,
    end: str,
    n_sims: int,
    n_days: int,
    fast: int,
    slow: int,
    bootstrap_method: str = "block",
    block_length: int = 15,
    cost_bps: float = 5.0,
    lookback: int = 20,
    n_long: int = 2,
    n_short: int = 2,
    rebalance_freq: int = 5,
    include_ma: bool = True,
) -> dict:
    """Run the multi-ticker pipeline and compare strategies on one shared simulation tensor."""
    panel = get_returns_panel(tickers, start, end)
    tensor = bootstrap_paths_multi(
        panel, n_sims, n_days, method=bootstrap_method, block_length=block_length
    )

    # per-day simulated returns, shared by every strategy so differences are strategy-driven
    sim_returns = tensor[:, 1:, :] / tensor[:, :-1, :] - 1

    results: dict[str, np.ndarray] = {}

    if include_ma:
        # MA crossover runs on the first ticker's slice of the shared tensor
        ma_finals = []
        for i in range(n_sims):
            prices = pd.Series(tensor[i, :, 0])
            signals = compute_signals(prices, fast=fast, slow=slow)
            strategy_returns = compute_strategy_returns(prices, signals, cost_bps=cost_bps)
            ma_finals.append(compute_final_return(strategy_returns))
        results[f"ma_{tickers[0]}"] = np.array(ma_finals)

    xs_finals = []
    for i in range(n_sims):
        sim_panel = pd.DataFrame(sim_returns[i], columns=tickers)
        ranks = compute_ranks(sim_panel, lookback=lookback)
        weights = compute_weights(ranks, n_long=n_long, n_short=n_short, rebalance_freq=rebalance_freq)
        strategy_returns = compute_xs_returns(sim_panel, weights, cost_bps=cost_bps)
        xs_finals.append(compute_final_return(strategy_returns))
    results["xs_momentum"] = np.array(xs_finals)

    # equal-weighted basket, rebalanced daily — the drift benchmark on the same paths
    basket_daily = sim_returns.mean(axis=2)
    results["buy_and_hold"] = np.prod(1 + basket_daily, axis=1) - 1

    print_comparison(results)
    plot_return_distribution(results["xs_momentum"])
    plot_sample_paths(tensor[:, :, 0])

    return {
        name: {
            "expected_return": float(np.mean(arr)),
            "volatility": float(np.std(arr)),
            "sharpe": float(np.mean(arr) / np.std(arr)),
            "var_95": float(np.percentile(arr, 5)),
        }
        for name, arr in results.items()
    }


def main() -> None:
    """Parse CLI arguments and run the Monte Carlo backtest pipeline."""
    parser = argparse.ArgumentParser(description="Monte Carlo strategy backtester")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start",  required=True)
    parser.add_argument("--end",    required=True)
    parser.add_argument("--n_sims", type=int, default=1000)
    parser.add_argument("--n_days", type=int, default=252)
    parser.add_argument("--fast",   type=int, default=20)
    parser.add_argument("--slow",   type=int, default=50)
    parser.add_argument("--bootstrap_method", choices=["iid", "block"], default="block")
    parser.add_argument("--block_length", type=int, default=15)
    parser.add_argument("--cost_bps", type=float, default=5.0)
    parser.add_argument("--tickers", default=",".join(MAG7))
    parser.add_argument("--strategy", choices=["ma", "xsmom", "both"], default="ma")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--n_long", type=int, default=2)
    parser.add_argument("--n_short", type=int, default=2)
    parser.add_argument("--rebalance_freq", type=int, default=5)
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    if args.strategy == "ma":
        # single-ticker path unchanged: --ticker wins, else the first of --tickers
        run(
            ticker=args.ticker or tickers[0],
            start=args.start,
            end=args.end,
            n_sims=args.n_sims,
            n_days=args.n_days,
            fast=args.fast,
            slow=args.slow,
            bootstrap_method=args.bootstrap_method,
            block_length=args.block_length,
            cost_bps=args.cost_bps,
        )
    else:
        run_multi(
            tickers=tickers,
            start=args.start,
            end=args.end,
            n_sims=args.n_sims,
            n_days=args.n_days,
            fast=args.fast,
            slow=args.slow,
            bootstrap_method=args.bootstrap_method,
            block_length=args.block_length,
            cost_bps=args.cost_bps,
            lookback=args.lookback,
            n_long=args.n_long,
            n_short=args.n_short,
            rebalance_freq=args.rebalance_freq,
            include_ma=args.strategy == "both",
        )


if __name__ == "__main__":
    main()
