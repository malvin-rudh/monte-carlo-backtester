# Monte Carlo Backtester

Tests the robustness of a moving average crossover strategy by running it across thousands of bootstrapped price paths. Instead of a single backtest result, the output is a full distribution of outcomes, separating genuine edge from luck.

## Project structure

```
src/
├── data/fetcher.py          — download, clean, cache, and align OHLCV data from Yahoo Finance
├── simulation/resampler.py  — bootstrap price paths (single-ticker matrix or shared-date multi-ticker tensor)
├── strategy/ma_crossover.py — MA crossover signals, strategy returns, and final P&L
├── strategy/cross_sectional_momentum.py — Mag 7 rank momentum: long winners, short losers
├── analysis/metrics.py      — summary statistics, strategy comparison, and output charts
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

| Argument             | Default | Description                          |
|----------------------|---------|--------------------------------------|
| `--ticker`           | —       | Yahoo Finance ticker symbol          |
| `--start`            | —       | Historical data start date (YYYY-MM-DD) |
| `--end`              | —       | Historical data end date (YYYY-MM-DD)   |
| `--n_sims`           | 1000    | Number of Monte Carlo simulations    |
| `--n_days`           | 252     | Trading days per simulation path     |
| `--fast`             | 20      | Fast MA window (days)                |
| `--slow`             | 50      | Slow MA window (days)                |
| `--bootstrap_method` | `block` | Resampling method: `block` (contiguous blocks, preserves autocorrelation) or `iid` (single days, null test) |
| `--block_length`     | 15      | Days per block for the block bootstrap (sensible range 10–20) |
| `--cost_bps`         | 5       | Round-trip transaction cost in basis points, charged per unit of turnover on the execution day |
| `--strategy`         | `ma`    | `ma` (single-ticker MA crossover), `xsmom` (Mag 7 rank momentum + basket benchmark), or `both` (three-way comparison) |
| `--tickers`          | Mag 7   | Comma-separated list for the multi-asset path (default `AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA`) |
| `--lookback`         | 20      | Trailing-return window (days) for the cross-sectional ranking |
| `--n_long`           | 2       | Number of top-ranked names to hold long |
| `--n_short`          | 2       | Number of bottom-ranked names to hold short |
| `--rebalance_freq`   | 5       | Trading days between ranking refreshes |

With `--strategy ma`, `--ticker` selects the single ticker (falling back to
the first entry of `--tickers` if omitted). With `xsmom`/`both`, all 7
tickers are simulated jointly via a shared-date block bootstrap — one block
sequence per simulation applied to every ticker, so cross-asset correlation
comes from real joint history — and an equal-weighted buy-and-hold basket
benchmark is included automatically. `both` additionally runs the MA
crossover on the first ticker's slice of the same simulation tensor.

## Example output

AAPL 2015–2024, n_sims=1000, n_days=252, fast=20/slow=50.

**Before** — original method (`--bootstrap_method iid --cost_bps 0`):

```
===================================
  Monte Carlo Simulation Summary
===================================
  Expected Return :    14.10%
  Volatility      :    27.01%
  Sharpe Ratio    :     0.522
  VaR (95%)       :   -19.33%
===================================
```

**After** — fixed method (`--bootstrap_method block --block_length 15 --cost_bps 5`):

```
===================================
  Monte Carlo Simulation Summary
===================================
  Expected Return :    14.80%
  Volatility      :    25.41%
  Sharpe Ratio    :     0.582
  VaR (95%)       :   -18.33%
===================================
```

**Three-way comparison** — `--strategy both --start 2015-01-01 --end 2024-12-31`
(Mag 7, n_sims=1000, n_days=252, defaults otherwise):

```
===================================================================
  Monte Carlo Strategy Comparison
===================================================================
  Metric                    ma_AAPL    xs_momentum   buy_and_hold
  Expected Return            14.79%          1.41%         42.40%
  Volatility                 25.41%         16.15%         37.40%
  Sharpe Ratio                0.582          0.087          1.134
  VaR (95%)                 -18.33%        -23.26%        -11.29%
===================================================================
```

The basket's +42% is inherited Mag-7 drift (a hindsight-selected winners
basket), not skill; the dollar-neutral momentum book, which hedges that
drift out, earns roughly nothing after costs. See the selection-bias caveat
in `docs/theory.md` before quoting these numbers.

Both fixes came out of the methodology audit in `docs/audit.md`: the i.i.d.
bootstrap zeroed out the autocorrelation a trend-following strategy needs
(issue #1), and the zero-cost assumption flattered whipsaw-heavy paths
(issue #2). The block bootstrap with per-flip costs is now the default; the
old i.i.d. mode is retained as a built-in null test.

Two charts are saved to `outputs/`:
- `return_distribution.png` — histogram of final returns across all simulations with the VaR 95% threshold marked
- `sample_paths.png` — fan chart of 100 randomly sampled price paths with the median path overlaid

## Running tests

```bash
pytest tests/
```

Unit tests use synthetic data only. `tests/test_integration.py` and `tests/test_fetcher.py::test_get_prices` make live network calls to Yahoo Finance.
