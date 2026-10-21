import numpy as np
import pandas as pd
import pytest

from electron_x_data_collector.research import (
    add_forward_mids,
    add_quote_state,
    horizon_label,
    ic_summary,
    information_coefficient,
    summarize,
    trade_markouts,
)

S = 1_000_000  # µs per second
A, B = "BFUT500-ERCOT-HB_NORTH-261015-09", "BFUT500-PJM-WESTERN-261015-09"


def l1_frame(rows: list[dict]) -> pd.DataFrame:
    """Normalized-looking L1 frame from a list of partial rows."""
    df = pd.DataFrame(rows)
    df["_seq"] = np.arange(len(df))
    for col in ("best_bid", "best_ask", "trade_px"):
        df[col] = df.get(col, pd.Series(np.nan, index=df.index)).astype("float64")
    for col in ("best_bid_qty", "best_ask_qty", "trade_qty"):
        df[col] = df.get(col, pd.Series(pd.NA, index=df.index)).astype("Int64")
    df["event_type"] = df["event_type"].astype("category")
    df["aggressor_side"] = df.get("aggressor_side", pd.Series(None, index=df.index)).astype("category")
    df["trade_date"] = pd.Timestamp("2026-10-15")
    return df


def quote(sym, t, bid, ask, bq=10, aq=10):
    return dict(symbol=sym, muts=t, event_type="quote", best_bid=bid, best_ask=ask, best_bid_qty=bq, best_ask_qty=aq)


def trade(sym, t, px, qty, side):
    return dict(symbol=sym, muts=t, event_type="trade", trade_px=px, trade_qty=qty, aggressor_side=side)


@pytest.fixture
def l1() -> pd.DataFrame:
    df = l1_frame(
        [
            quote(A, 0, 40.00, 40.50, bq=10, aq=30),
            trade(A, 1 * S, 40.50, 5, "BUY"),
            quote(A, int(1.5 * S), 40.25, 40.75),
            quote(A, 3 * S, np.nan, 41.00, bq=None),  # bid side empty
            quote(A, 5 * S, 40.75, 41.25),
            quote(B, 0, 50.00, 49.75),  # crossed
            quote(B, 2 * S, 50.00, 50.50),
        ]
    )
    return df.sample(frac=1, random_state=0)  # functions must not rely on input order


def test_horizon_label():
    assert horizon_label("10s") == "10s"
    assert horizon_label(5) == "5ev"
    assert horizon_label(pd.Timedelta(milliseconds=250)) == "250ms"
    assert horizon_label(pd.Timedelta(minutes=2)) == "2min"


def test_quote_state(l1):
    q = add_quote_state(l1)
    a = q[q.symbol == A].set_index("muts")
    assert a.loc[0, "mid"] == 40.25
    assert a.loc[0, "spread"] == 2
    assert a.loc[0, "imbalance"] == pytest.approx(-0.5)
    # trade row sees the quote from before it
    assert a.loc[S, "mid"] == 40.25
    # an empty bid side must not inherit the old bid
    assert np.isnan(a.loc[3 * S, "bid"]) and np.isnan(a.loc[3 * S, "mid"])
    b = q[q.symbol == B].set_index("muts")
    assert np.isnan(b.loc[0, "mid"])  # crossed book has no mid
    assert b.loc[2 * S, "mid"] == 50.25


def test_forward_mids(l1):
    q = add_forward_mids(add_quote_state(l1), ["2s", 1, "10s"])
    a = q[q.symbol == A].set_index("muts")
    assert a.loc[0, "mid_2s"] == 40.50
    assert a.loc[0, "ret_2s"] == 1  # 40.25 -> 40.50 is one tick
    assert a.loc[0, "ret_1ev"] == 0  # next event is the trade row, same quote state
    assert np.isnan(a.loc[5 * S, "ret_1ev"])  # no later event
    assert a[["ret_10s"]].isna().all().item()  # past the symbol's last event


def test_forward_mids_accepts_raw_frame(l1):
    q = add_forward_mids(l1, ["2s"])
    assert "imbalance" in q and q.loc[(q.symbol == A) & (q.muts == 0), "ret_2s"].item() == 1


def test_trade_markouts(l1):
    mk = trade_markouts(l1, ["1s", "3s", "4s", "10s"])
    assert len(mk) == 1
    r = mk.iloc[0]
    assert r.sign == 1 and r.mid_before == 40.25 and r.spread_before == 2
    assert r.eff_spread == 2  # paid 1 tick above mid, effective spread 2 ticks
    assert r.markout_1s == 0  # mid at 2s is 40.50 = trade price
    assert np.isnan(r.markout_3s)  # mid at 4s is undefined (bid side empty)
    assert r.markout_4s == 2  # mid at 5s is 41.00
    assert np.isnan(r.markout_10s)


def test_sell_markout_sign():
    df = l1_frame([quote(A, 0, 40.0, 40.5), trade(A, S, 40.0, 1, "SELL"), quote(A, 2 * S, 39.5, 40.0)])
    r = trade_markouts(df, ["1s"]).iloc[0]
    assert r.sign == -1 and r.eff_spread == 2 and r.markout_1s == 1  # mid 39.75 is a tick below the sale


def test_ic_and_summary():
    rng = np.random.default_rng(0)
    frames = []
    for day in range(5):
        n = 500
        sig = rng.normal(size=n)
        frames.append(
            pd.DataFrame(
                {"trade_date": pd.Timestamp("2026-10-01") + pd.Timedelta(days=day), "sig": sig, "ret_1s": sig + rng.normal(size=n)}
            )
        )
    df = pd.concat(frames, ignore_index=True)
    ic = information_coefficient(df, "sig", ["1s"])
    assert len(ic) == 5 and (ic["1s"] > 0.5).all()
    s = ic_summary(ic)
    assert s.loc["1s", "hit_rate"] == 1 and s.loc["1s", "n_groups"] == 5

    st = summarize(df, ["ret_1s"])
    assert st.loc["ret_1s", "n"] == 2500
    by_day = summarize(df, ["ret_1s"], by="trade_date")
    assert len(by_day) == 5
