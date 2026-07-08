import numpy as np
import pandas as pd
import pytest

from src.data.fetcher import (
    align_closes,
    get_prices,
    get_returns_panel,
    handle_missing,
    load_from_csv,
    save_to_csv,
)


def _make_ohlcv(n: int = 10) -> pd.DataFrame:
    """Return a minimal synthetic OHLCV DataFrame with a DatetimeIndex."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {
        "Open": np.ones(n) * 100,
        "High": np.ones(n) * 105,
        "Low": np.ones(n) * 95,
        "Close": np.ones(n) * 100,
        "Volume": np.ones(n) * 1_000_000,
    }
    return pd.DataFrame(data, index=idx)


def test_get_prices():
    df = get_prices("AAPL", "2020-01-01", "2020-03-31")
    assert df.shape[0] > 0
    assert df["Close"].isna().sum() == 0
    assert isinstance(df.index, pd.DatetimeIndex)


def test_handle_missing():
    df = _make_ohlcv(10)
    # inject NaN in rows 3, 4, 5 (a run of 3 — within the fill limit)
    df.iloc[3:6] = np.nan
    result = handle_missing(df)
    assert result.isna().sum().sum() == 0


def test_align_closes_narrows_to_common_history():
    # ticker B starts 4 sessions later (like META vs the rest) — join must narrow to B's range
    idx_a = pd.date_range("2020-01-01", periods=10, freq="B")
    idx_b = idx_a[4:]
    closes = {
        "A": pd.Series(np.linspace(100, 109, 10), index=idx_a),
        "B": pd.Series(np.linspace(50, 55, 6), index=idx_b),
    }

    panel = align_closes(closes)

    assert panel.index.equals(idx_b)
    assert list(panel.columns) == ["A", "B"]
    # no missing ticker on any remaining date
    assert panel.notna().all().all()


def test_get_returns_panel_live():
    # META lists 2012-05-18 — a 2010 start must narrow to META's history (network test)
    panel = get_returns_panel(["AAPL", "META"], "2010-01-01", "2015-01-01")

    assert list(panel.columns) == ["AAPL", "META"]
    assert panel.notna().all().all()
    assert panel.index.min() >= pd.Timestamp("2012-05-18")


def test_save_load_roundtrip(tmp_path, monkeypatch):
    import src.data.fetcher as fetcher_module

    # redirect DATA_DIR to a temp directory
    monkeypatch.setattr(fetcher_module, "DATA_DIR", tmp_path)

    df = _make_ohlcv(5)
    save_to_csv(df, "TEST")
    loaded = load_from_csv("TEST")

    pd.testing.assert_frame_equal(df, loaded, check_freq=False)
