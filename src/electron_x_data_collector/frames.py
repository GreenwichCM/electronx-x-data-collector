"""pandas helpers: type the raw API frames and reshape them for research."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from .models import (
    BOOK_DEPTH,
    EVENT_MODELS,
    SYMBOL_RE,
    BaseEvent,
    Dataset,
)

NY = "America/New_York"

_CATEGORICAL = ("event_type", "exchange_flags", "aggressor_side")
_STRING = ("symbol", "execution_id", "trade_id")


def _price_qty_columns(dataset: Dataset) -> tuple[list[str], list[str]]:
    prices, qtys = ["trade_px"], ["trade_qty"]
    if dataset is Dataset.L1_TAQ:
        prices += ["best_bid", "best_ask"]
        qtys += ["best_bid_qty", "best_ask_qty"]
    else:
        for n in range(1, BOOK_DEPTH + 1):
            prices += [f"bid_px_{n}", f"ask_px_{n}"]
            qtys += [f"bid_qty_{n}", f"ask_qty_{n}"]
    return prices, qtys


def empty_frame(dataset: Dataset) -> pd.DataFrame:
    """Zero-row frame with the dataset's columns (used when a query returns nothing)."""
    cols = ["_seq"] + [f for f in EVENT_MODELS[dataset].model_fields if f != "seq"]
    return pd.DataFrame(columns=cols)


def normalize_frame(df: pd.DataFrame, dataset: Dataset | str) -> pd.DataFrame:
    """Convert a raw liberator frame into properly typed columns.

    * ``timestamp``      -> tz-aware datetime (America/New_York)
    * ``transact_time``  -> tz-aware datetime (UTC, nanosecond precision kept)
    * ``publish_date``   -> tz-aware datetime (UTC)
    * ``trade_date``     -> datetime64 (midnight, naive)
    * prices             -> float64 (NaN where not populated)
    * quantities         -> nullable Int64
    * ``""``             -> NA in string / categorical columns
    """
    dataset = Dataset(dataset)
    df = df.copy()

    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(
            NY, ambiguous="NaT", nonexistent="NaT"
        )
    for col in ("transact_time", "publish_date"):
        if col in df:
            df[col] = pd.to_datetime(df[col].replace("", None), utc=True, errors="coerce")
    if "trade_date" in df:
        df["trade_date"] = pd.to_datetime(df["trade_date"].replace("", None), errors="coerce")

    prices, qtys = _price_qty_columns(dataset)
    for col in prices:
        if col in df:
            df[col] = pd.to_numeric(df[col].replace("", None), errors="coerce").astype("float64")
    for col in qtys:
        if col in df:
            df[col] = pd.to_numeric(df[col].replace("", None), errors="coerce").round().astype("Int64")

    for col in _STRING:
        if col in df:
            df[col] = df[col].replace("", None).astype("string")
    for col in _CATEGORICAL:
        if col in df:
            df[col] = df[col].replace("", None).astype("category")
    return df


def add_symbol_columns(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """Append product / product_variant / market / location / expiry parsed from the symbol."""
    uniq = pd.Series(df[symbol_col].dropna().unique())
    parts = uniq.astype(str).str.extract(SYMBOL_RE)
    parts.index = uniq.values
    parts["product_variant"] = pd.to_numeric(parts.pop("variant"), errors="coerce").astype("Int64")
    expiry_date = pd.to_datetime(parts.pop("expiry_date"), format="%y%m%d", errors="coerce")
    parts["expiry_hour"] = pd.to_numeric(parts.pop("expiry_hour"), errors="coerce").astype("Int64")
    parts["expiry"] = expiry_date + pd.to_timedelta(parts["expiry_hour"].astype("float64"), unit="h")
    for col in ("product", "market", "location"):
        parts[col] = parts[col].astype("category")
    joined = parts.reindex(df[symbol_col].to_numpy())
    joined.index = df.index
    return pd.concat([df.drop(columns=parts.columns, errors="ignore"), joined], axis=1)


def add_mid_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``mid`` and ``spread`` from top of book (works for L1 and L2 frames)."""
    bid = df["best_bid"] if "best_bid" in df else df["bid_px_1"]
    ask = df["best_ask"] if "best_ask" in df else df["ask_px_1"]
    return df.assign(mid=(bid + ask) / 2, spread=ask - bid)


def book_to_long(df: pd.DataFrame, id_cols: tuple[str, ...] = ("muts", "_seq", "timestamp", "symbol")) -> pd.DataFrame:
    """Reshape a wide L2 frame to one row per (event, side, level) with a populated price."""
    id_cols = tuple(c for c in id_cols if c in df)
    frames = []
    for side in ("bid", "ask"):
        for n in range(1, BOOK_DEPTH + 1):
            px, qty = f"{side}_px_{n}", f"{side}_qty_{n}"
            if px not in df:
                continue
            part = df.loc[df[px].notna(), [*id_cols, px, qty]].rename(columns={px: "price", qty: "qty"})
            part["side"] = side
            part["level"] = n
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=[*id_cols, "side", "level", "price", "qty"])
    out = pd.concat(frames, ignore_index=True)
    out["side"] = out["side"].astype("category")
    return out.sort_values([*id_cols[:2], "side", "level"], ignore_index=True)


def iter_models(df: pd.DataFrame, dataset: Dataset | str) -> Iterator[BaseEvent]:
    """Validate each row of a raw *or* normalized frame into its pydantic event model."""
    model = EVENT_MODELS[Dataset(dataset)]
    for rec in df.to_dict(orient="records"):
        yield model.model_validate(rec)
