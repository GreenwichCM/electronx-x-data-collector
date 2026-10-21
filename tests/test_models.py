from datetime import datetime

import pandas as pd
import pytest

from electron_x_data_collector import Credentials, Dataset, Market, QueryParams, Symbol, client, normalize_frame
from electron_x_data_collector.client import _day_windows
from electron_x_data_collector.models import NY_TZ


def test_symbol_parse():
    s = Symbol.parse("BFUT500-ERCOT-HB_NORTH-260701-18")
    assert s.market is Market.ERCOT and s.product_code == "BFUT500" and s.location == "HB_NORTH"
    assert s.expiry == datetime(2026, 7, 1, 18)


def test_symbol_parse_hyphenated_market_and_hour_24():
    s = Symbol.parse("BFUT-ISO-NE-.H.INTERNAL_HUB-260901-24")
    assert s.market is Market.ISO_NE and s.product_variant is None
    assert s.expiry == datetime(2026, 9, 2, 0)


def test_symbol_parse_rejects_garbage():
    with pytest.raises(ValueError):
        Symbol.parse("AAPL")


def test_query_params_where_clause():
    p = QueryParams(
        dataset="L1 TAQ", back_to="2026-10-15", as_of="2026-10-16", markets=["PJM", "ISO-NE"], products="BFUT", event_type="trade"
    )
    assert p.back_to.tzinfo is NY_TZ
    assert p.where_clause() == (
        "(contains(symbol, '-PJM-') OR contains(symbol, '-ISO-NE-')) AND (starts_with(symbol, 'BFUT')) AND event_type = 'trade'"
    )


@pytest.mark.parametrize(("back_to", "as_of"), [("2026-10-16", "2026-10-15"), ("2025-01-01", "2025-01-02")])
def test_query_params_rejects_bad_ranges(back_to, as_of):
    with pytest.raises(ValueError):
        QueryParams(dataset="L1 TAQ", back_to=back_to, as_of=as_of)


def test_day_windows_split_on_ny_midnight():
    start = datetime(2026, 10, 14, 12, tzinfo=NY_TZ)
    end = datetime(2026, 10, 16, 6, tzinfo=NY_TZ)
    w = _day_windows(start, end)
    assert [(lo.day, lo.hour, hi.day, hi.hour) for lo, hi in w] == [(14, 12, 15, 0), (15, 0, 16, 0), (16, 0, 16, 6)]


def test_normalize_l1_frame():
    raw = pd.DataFrame(
        {
            "timestamp": ["2026-10-15 10:00:00.024"],
            "transact_time": ["2026-10-15 14:00:00.024333415Z"],
            "best_bid": ["40.25"],
            "best_ask": [""],
            "best_bid_qty": ["10"],
            "event_type": ["quote"],
            "symbol": ["BFUT500-ERCOT-HB_NORTH-261015-09"],
        }
    )
    df = normalize_frame(raw, Dataset.L1_TAQ)
    assert str(df["timestamp"].dt.tz) == "America/New_York"
    assert df["transact_time"].iloc[0].nanosecond == 415
    assert df["best_bid"].iloc[0] == 40.25 and pd.isna(df["best_ask"].iloc[0])
    assert df["best_bid_qty"].dtype == "Int64" and df["event_type"].dtype == "category"


@pytest.fixture
def no_dotenv(monkeypatch):
    """Isolate from the developer's real .env and environment."""
    monkeypatch.setattr(client, "load_dotenv", lambda: None)
    for var in ("ELECTRONX_USER", "ELECTRONX_TOKEN", "ELECTRONX_URL"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_credentials_from_env(no_dotenv):
    no_dotenv.setenv("ELECTRONX_USER", "u")
    no_dotenv.setenv("ELECTRONX_TOKEN", "s3cret-token")
    no_dotenv.setenv("USERNAME", "os-login")
    c = Credentials.resolve()
    assert (c.user, c.token, c.url) == ("u", "s3cret-token", client.DEFAULT_URL)
    assert "s3cret" not in repr(c)


def test_credentials_args_override_env(no_dotenv):
    no_dotenv.setenv("ELECTRONX_USER", "u")
    no_dotenv.setenv("ELECTRONX_TOKEN", "t")
    c = Credentials.resolve(user="x", token="y", url="https://example.test/")
    assert (c.user, c.token, c.url) == ("x", "y", "https://example.test")


def test_credentials_missing(no_dotenv):
    with pytest.raises(ValueError, match="ELECTRONX_USER"):
        Credentials.resolve()
