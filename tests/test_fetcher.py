import numpy as np
import pandas as pd
import pytest

from src.data.fetcher import handle_missing, load_from_csv, save_to_csv, get_prices


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


def test_save_load_roundtrip(tmp_path, monkeypatch):
    import src.data.fetcher as fetcher_module

    # redirect DATA_DIR to a temp directory
    monkeypatch.setattr(fetcher_module, "DATA_DIR", tmp_path)

    df = _make_ohlcv(5)
    save_to_csv(df, "TEST")
    loaded = load_from_csv("TEST")

    pd.testing.assert_frame_equal(df, loaded, check_freq=False)
