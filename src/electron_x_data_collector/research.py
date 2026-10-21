"""Tier-A signal research on L1 TAQ: prevailing quotes, forward mid moves, trade markouts, IC.

All price moves are in ticks (``TICK_SIZE``), not percent: power prices can sit
near zero or go negative, so percentage returns are meaningless.

Clock and ordering: events are ordered by ``(symbol, muts, _seq)`` and horizons
are measured on ``muts`` (matching-engine time, epoch µs). A forward value is
NaN when ``t + h`` falls past the symbol's last event in the frame (end of the
query window or contract expiry), so nothing is filled in from data you don't have.

Typical use::

    l1 = ex.l1("2026-10-01", "2026-10-16", markets="ERCOT")
    q = add_forward_mids(add_quote_state(l1), ["1s", "10s", "60s"])
    ic = information_coefficient(q, "imbalance", ["1s", "10s", "60s"])
    ic_summary(ic)

    mk = trade_markouts(l1, ["1s", "10s", "60s"])
    summarize(mk, ["eff_spread", "markout_1s", "markout_10s", "markout_60s"])
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .models import TICK_SIZE

Horizon = str | pd.Timedelta | int
"""A time horizon (``"500ms"``, ``"10s"``, ``pd.Timedelta``) or an event count (``int``)."""

_QUOTE_COLS = ("best_bid", "best_ask", "best_bid_qty", "best_ask_qty")
_SORT = ["symbol", "muts", "_seq"]


def horizon_label(h: Horizon) -> str:
    """Column suffix for a horizon: ``"10s"`` -> ``"10s"``, ``5`` -> ``"5ev"``."""
    if isinstance(h, (int, np.integer)):
        return f"{int(h)}ev"
    if isinstance(h, str):
        return h
    return _format_timedelta(pd.Timedelta(h))


def _format_timedelta(td: pd.Timedelta) -> str:
    us = td // pd.Timedelta(microseconds=1)
    for unit, size in (("h", 3_600_000_000), ("min", 60_000_000), ("s", 1_000_000), ("ms", 1_000)):
        if us % size == 0:
            return f"{us // size}{unit}"
    return f"{us}us"


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in _SORT if c in df]
    return df.sort_values(keys, kind="stable", ignore_index=True)


def _is_trade(df: pd.DataFrame) -> np.ndarray:
    return (df["event_type"].astype("string") == "trade").fillna(False).to_numpy()


def add_quote_state(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the prevailing top of book to every row.

    Adds ``bid``, ``ask``, ``bid_qty``, ``ask_qty``, ``mid``, ``spread`` (ticks)
    and ``imbalance`` = (bid_qty - ask_qty) / (bid_qty + ask_qty).

    The state comes from the latest *quote* row at or before each row, so a trade
    row sees the book as it stood just before the trade. Whole quote rows are carried
    forward, not single columns: a quote with an empty bid side means "no bid", and
    an older bid must not leak through. ``mid`` is NaN when a side is empty or the
    book is locked or crossed.

    Returns a copy sorted by ``(symbol, muts, _seq)``.
    """
    out = _sorted(df)
    is_quote = ~_is_trade(out)
    pos = pd.Series(np.where(is_quote, np.arange(len(out)), np.nan))
    src = pos.groupby(out["symbol"].to_numpy(), sort=False).ffill().to_numpy()
    have = ~np.isnan(src)
    idx = np.where(have, src, 0).astype(np.int64)

    for col, name in zip(_QUOTE_COLS, ("bid", "ask", "bid_qty", "ask_qty"), strict=True):
        vals = out[col].astype("float64").to_numpy()[idx]
        out[name] = np.where(have, vals, np.nan)

    bid, ask = out["bid"].to_numpy(), out["ask"].to_numpy()
    valid = ask > bid
    out["mid"] = np.where(valid, (bid + ask) / 2, np.nan)
    out["spread"] = np.where(valid, (ask - bid) / TICK_SIZE, np.nan)
    depth = out["bid_qty"] + out["ask_qty"]
    out["imbalance"] = ((out["bid_qty"] - out["ask_qty"]) / depth.where(depth > 0)).where(valid)
    return out


def _forward_values(
    df: pd.DataFrame, values: np.ndarray, horizons: Sequence[Horizon]
) -> dict[str, np.ndarray]:
    """``values`` as of ``t + h`` for each row, per symbol. ``df`` must already be sorted."""
    n = len(df)
    result = {horizon_label(h): np.full(n, np.nan) for h in horizons}
    muts = df["muts"].to_numpy(dtype=np.int64)
    groups = df.groupby(df["symbol"].to_numpy(), sort=False).indices

    for rows in groups.values():
        t, v = muts[rows], values[rows]
        for h in horizons:
            fwd = np.full(len(rows), np.nan)
            if isinstance(h, (int, np.integer)):
                if h < len(rows):
                    fwd[: len(rows) - h] = v[h:]
            else:
                target = t + pd.Timedelta(h) // pd.Timedelta(microseconds=1)
                j = np.searchsorted(t, target, side="right") - 1
                ok = target <= t[-1]
                fwd[ok] = v[j[ok]]
            result[horizon_label(h)][rows] = fwd
    return result


def add_forward_mids(df: pd.DataFrame, horizons: Sequence[Horizon]) -> pd.DataFrame:
    """Add ``mid_<h>`` (prevailing mid at ``t + h``) and ``ret_<h>`` (mid change in ticks).

    ``df`` should come from :func:`add_quote_state`. A time horizon includes every
    event up to and including ``t + h``. An ``int`` horizon looks that many events ahead
    in the same symbol.
    """
    # add_mid_spread() also writes "mid", but without the prevailing-quote logic, so key on "imbalance"
    out = _sorted(df) if "imbalance" in df else add_quote_state(df)
    mid = out["mid"].to_numpy(dtype="float64")
    for label, fwd in _forward_values(out, mid, horizons).items():
        out[f"mid_{label}"] = fwd
        out[f"ret_{label}"] = (fwd - mid) / TICK_SIZE
    return out


def trade_markouts(df: pd.DataFrame, horizons: Sequence[Horizon]) -> pd.DataFrame:
    """One row per trade, with costs and markouts from the aggressor's side, in ticks.

    * ``sign``: +1 if the aggressor bought, -1 if they sold.
    * ``eff_spread``: 2 * sign * (trade_px - mid before the trade). This is what the
      aggressor paid to cross. A passive fill earns it.
    * ``markout_<h>``: sign * (mid at t + h - trade_px). This is the aggressor's PnL per
      lot at the horizon. The passive side earns the negative. Persistently positive
      values mean the flow is informed and resting orders get adversely selected.

    Trades with no aggressor side are dropped.
    """
    q = add_quote_state(df)
    mid = q["mid"].to_numpy(dtype="float64")
    fwd = _forward_values(q, mid, horizons)

    side = q["aggressor_side"].astype("string")
    sign = np.select(
        [(side == "BUY").fillna(False).to_numpy(bool), (side == "SELL").fillna(False).to_numpy(bool)],
        [1.0, -1.0],
        default=np.nan,
    )
    keep = _is_trade(q) & ~np.isnan(sign)

    px = q["trade_px"].to_numpy(dtype="float64")
    cols = [c for c in ("symbol", "timestamp", "muts", "_seq", "trade_date", "trade_px", "trade_qty", "aggressor_side") if c in q]
    out = q.loc[keep, cols].copy()
    out["sign"] = sign[keep]
    out["mid_before"] = mid[keep]
    out["spread_before"] = q["spread"].to_numpy()[keep]
    out["eff_spread"] = 2 * sign[keep] * (px[keep] - mid[keep]) / TICK_SIZE
    for label, f in fwd.items():
        out[f"markout_{label}"] = sign[keep] * (f[keep] - px[keep]) / TICK_SIZE
    return out.reset_index(drop=True)


def summarize(
    df: pd.DataFrame,
    cols: Sequence[str],
    by: str | Sequence[str] | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """Mean, standard error, t-stat and count for each column, optionally grouped.

    ``weight`` (e.g. ``"trade_qty"``) gives a size-weighted mean. The standard error
    still treats rows as independent, so it understates uncertainty when events cluster.
    For honest error bars, use per-day means (``by="trade_date"``) and look at their spread.
    """

    def stats(g: pd.DataFrame) -> pd.DataFrame:
        rows = {}
        for c in cols:
            x = g[c].astype("float64")
            m = x.notna()
            if weight is None:
                w = pd.Series(1.0, index=x.index)
            else:
                w = g[weight].astype("float64").fillna(0)
            x, w = x[m], w[m]
            n = int(m.sum())
            wsum = w.sum()
            mean = float((x * w).sum() / wsum) if wsum > 0 else np.nan
            se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            rows[c] = {"mean": mean, "se": se, "t": mean / se if se and se > 0 else np.nan, "n": n}
        return pd.DataFrame(rows).T

    if by is None:
        return stats(df)
    return df.groupby(by, observed=True).apply(stats, include_groups=False)


def information_coefficient(
    df: pd.DataFrame,
    signal: str,
    horizons: Sequence[Horizon],
    by: str | Sequence[str] = "trade_date",
    method: str = "spearman",
    min_obs: int = 30,
) -> pd.DataFrame:
    """Correlation between ``signal`` and ``ret_<h>`` for each group (one row per group).

    Run :func:`add_forward_mids` first. Groups with fewer than ``min_obs`` valid pairs
    get NaN. Consecutive events overlap heavily at long horizons, so treat each
    *group* (by default a day) as one observation, not each row.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', not {method!r}")
    labels = [horizon_label(h) for h in horizons]

    def ic(g: pd.DataFrame) -> pd.Series:
        out = {}
        for label in labels:
            pair = g[[signal, f"ret_{label}"]].astype("float64").dropna()
            if len(pair) < min_obs or (pair.nunique() < 2).any():  # constant column: corr undefined
                out[label] = np.nan
                continue
            if method == "spearman":  # Pearson on ranks; avoids pandas' scipy dependency
                pair = pair.rank()
            out[label] = pair[signal].corr(pair[f"ret_{label}"])
        return pd.Series(out)

    return df.groupby(by, observed=True).apply(ic, include_groups=False)


def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-group IC table into mean, std, t-stat, hit rate and group count."""
    n = ic.notna().sum()
    mean, std = ic.mean(), ic.std(ddof=1)
    return pd.DataFrame(
        {
            "mean_ic": mean,
            "std_ic": std,
            "t": mean / (std / np.sqrt(n)),
            "hit_rate": (ic > 0).sum() / n,
            "n_groups": n,
        }
    )
