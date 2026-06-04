# Monte Carlo Backtester

Tests the robustness of a moving average crossover strategy by running it across thousands of bootstrapped price paths. Instead of a single backtest result, the output is a full distribution of outcomes, separating genuine edge from luck.

## Project structure

```
src/
├── data/fetcher.py          — download, clean, and cache OHLCV data from Yahoo Finance
├── simulation/resampler.py  — bootstrap N price paths from historical returns
├── strategy/ma_crossover.py — MA crossover signals, strategy returns, and final P&L
├── analysis/metrics.py      — summary statistics and output charts
└── main.py                  — CLI orchestrator
```

## Installation

```bash
git clone <repo-url>
cd monte-carlo-backtester
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py \
  --ticker AAPL \
  --start  2015-01-01 \
  --end    2024-12-31 \
  --n_sims 1000 \
  --n_days 252 \
  --fast   20 \
  --slow   50
```

| Argument  | Default | Description                          |
|-----------|---------|--------------------------------------|
| `--ticker`| —       | Yahoo Finance ticker symbol          |
| `--start` | —       | Historical data start date (YYYY-MM-DD) |
| `--end`   | —       | Historical data end date (YYYY-MM-DD)   |
| `--n_sims`| 1000    | Number of Monte Carlo simulations    |
| `--n_days`| 252     | Trading days per simulation path     |
| `--fast`  | 20      | Fast MA window (days)                |
| `--slow`  | 50      | Slow MA window (days)                |

## Example output

```
===================================
  Monte Carlo Simulation Summary
===================================
  Expected Return :    12.34%
  Volatility      :     8.21%
  Sharpe Ratio    :     1.503
  VaR (95%)       :    -4.67%
===================================
```

Two charts are saved to `outputs/`:
- `return_distribution.png` — histogram of final returns across all simulations with the VaR 95% threshold marked
- `sample_paths.png` — fan chart of 100 randomly sampled price paths with the median path overlaid

## Running tests

```bash
pytest tests/
```

Unit tests use synthetic data only. `tests/test_integration.py` and `tests/test_fetcher.py::test_get_prices` make live network calls to Yahoo Finance.
