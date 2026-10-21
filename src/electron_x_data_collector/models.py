"""Pydantic models for the ElectronX L1 / L2 TAQ historical datasets.

Mirrors "ElectronX Market Data Historical Dataset Specification" plus what the
live API actually returns:

* ``timestamp`` is New York wall-clock time with no offset (e.g. ``2026-10-15 10:00:00.024``).
* ``transact_time`` / ``publish_date`` are UTC strings with a trailing ``Z``.
* L1 price/qty columns arrive as strings, ``""`` meaning "not populated".
* L2 price/qty columns arrive as doubles, ``NaN`` meaning "no level".
"""

from __future__ import annotations

import math
import re
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Self
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)

NY_TZ = ZoneInfo("America/New_York")
BOOK_DEPTH = 10


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Dataset(StrEnum):
    L1_TAQ = "L1 TAQ"
    L2_TAQ = "L2 TAQ"


class Market(StrEnum):
    ERCOT = "ERCOT"
    PJM = "PJM"
    MISO = "MISO"
    CAISO = "CAISO"
    SPP = "SPP"
    ISO_NE = "ISO-NE"
    NYISO = "NYISO"


class Product(StrEnum):
    BFUT = "BFUT"
    BOPT = "BOPT"


class EventType(StrEnum):
    QUOTE = "quote"
    TRADE = "trade"


class AggressorSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class ExchangeFlag(StrEnum):
    NEW = "NEW"
    CHANGE = "CHANGE"
    DELETE = "DELETE"
    TRADE = "TRADE"


#: First date data exists for each market (spec, "Date Range").
MARKET_LAUNCH_DATES: dict[Market, date] = {
    Market.ERCOT: date(2025, 12, 10),
    Market.PJM: date(2026, 4, 6),
    Market.CAISO: date(2026, 6, 1),
    Market.MISO: date(2026, 6, 1),
    Market.SPP: date(2026, 8, 24),
    Market.ISO_NE: date(2026, 8, 24),
    Market.NYISO: date(2026, 8, 24),
}

#: Earliest date with any data, across all markets.
DATA_START_DATE: date = min(MARKET_LAUNCH_DATES.values())

#: Conventions (spec, "Conventions").
TICK_SIZE = 0.25  # USD / MWh
LOT_SIZE_MWH = 1.0  # 1 lot = 1 contract = 1 MWh


# --------------------------------------------------------------------------- #
# Symbols
# --------------------------------------------------------------------------- #

_MARKET_ALT = "|".join(re.escape(m.value) for m in sorted(Market, key=lambda m: -len(m.value)))
SYMBOL_RE = re.compile(
    rf"^(?P<product>{'|'.join(p.value for p in Product)})(?P<variant>\d*)"
    rf"-(?P<market>{_MARKET_ALT})"
    r"-(?P<location>.+)"
    r"-(?P<expiry_date>\d{6})-(?P<expiry_hour>\d{2})$"
)


class Symbol(BaseModel):
    """A parsed contract symbol: ``<PRODUCT>-<MARKET>-<LOCATION>-<YYMMDD>-<HH>``.

    >>> Symbol.parse("BFUT500-ERCOT-HB_NORTH-260701-18").market
    <Market.ERCOT: 'ERCOT'>
    """

    model_config = ConfigDict(frozen=True)

    raw: str
    product: Product
    product_variant: int | None = Field(
        None, description="Numeric suffix on the product code, e.g. 500 in BFUT500."
    )
    market: Market
    location: str = Field(description="Hub / zone, e.g. HB_NORTH, ZONE_A.")
    expiry_date: date
    expiry_hour: int = Field(ge=0, le=24, description="HH part of the expiry (hour ending).")

    @classmethod
    def parse(cls, symbol: str) -> Self:
        m = SYMBOL_RE.match(symbol)
        if m is None:
            raise ValueError(f"Not a valid ElectronX symbol: {symbol!r}")
        g = m.groupdict()
        return cls(
            raw=symbol,
            product=Product(g["product"]),
            product_variant=int(g["variant"]) if g["variant"] else None,
            market=Market(g["market"]),
            location=g["location"],
            expiry_date=datetime.strptime(g["expiry_date"], "%y%m%d").date(),
            expiry_hour=int(g["expiry_hour"]),
        )

    @property
    def product_code(self) -> str:
        """Product with its variant suffix, e.g. ``BFUT500``."""
        return f"{self.product.value}{self.product_variant or ''}"

    @property
    def expiry(self) -> datetime:
        """Expiry as a naive datetime (``expiry_hour`` 24 rolls to next day 00:00)."""
        return datetime.combine(self.expiry_date, datetime.min.time()) + timedelta(
            hours=self.expiry_hour
        )

    def __str__(self) -> str:
        return self.raw


# --------------------------------------------------------------------------- #
# Value coercion helpers
# --------------------------------------------------------------------------- #


def _blank_to_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):  # drop ns so models from raw and typed frames agree
        return v.floor("us").to_pydatetime()
    return v


def _parse_ny_timestamp(v: Any) -> Any:
    v = _blank_to_none(v)
    if isinstance(v, str):
        v = datetime.fromisoformat(v)
    if isinstance(v, datetime) and v.tzinfo is None:
        v = v.replace(tzinfo=NY_TZ)
    return v


def _parse_utc_timestamp(v: Any) -> Any:
    """Parse ``2026-10-15 14:00:00.024333415Z`` (nanoseconds truncated to µs)."""
    v = _blank_to_none(v)
    if isinstance(v, str):
        s = v.rstrip("Z")
        if "." in s:
            head, frac = s.split(".", 1)
            s = f"{head}.{frac[:6]}"
        v = datetime.fromisoformat(s).replace(tzinfo=UTC)
    elif isinstance(v, datetime) and v.tzinfo is None:
        v = v.replace(tzinfo=UTC)
    return v


OptFloat = Annotated[float | None, BeforeValidator(_blank_to_none)]
OptInt = Annotated[int | None, BeforeValidator(_blank_to_none)]
OptStr = Annotated[str | None, BeforeValidator(_blank_to_none)]
NYDatetime = Annotated[datetime, BeforeValidator(_parse_ny_timestamp)]
UTCDatetime = Annotated[datetime, BeforeValidator(_parse_utc_timestamp)]


# --------------------------------------------------------------------------- #
# Row (event) models
# --------------------------------------------------------------------------- #


class BaseEvent(BaseModel):
    """Fields common to both L1 and L2 TAQ (one row = one quote or trade event)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)

    seq: int | None = Field(
        None,
        validation_alias=AliasChoices("_seq", "seq"),
        description="Server sequence number; tie-breaker within the same muts.",
    )
    timestamp: NYDatetime = Field(description="Event time (America/New_York).")
    muts: int = Field(description="Matching-engine update time, epoch microseconds.")
    transact_time: UTCDatetime = Field(
        description="Transaction time, UTC (source is ns precision; µs kept here)."
    )
    trade_date: date = Field(description="Trading session date (Central time).")
    publish_date: UTCDatetime = Field(description="Batch publication time, UTC.")
    symbol: str
    execution_id: OptStr = None
    exchange_flags: Annotated[ExchangeFlag | None, BeforeValidator(_blank_to_none)] = None
    event_type: EventType

    trade_px: OptFloat = Field(None, description="USD/MWh. Trade rows only.")
    trade_qty: OptInt = Field(None, description="Lots (= MWh). Trade rows only.")
    trade_id: OptStr = Field(None, description="Trade rows only; distinct from execution_id.")
    aggressor_side: Annotated[AggressorSide | None, BeforeValidator(_blank_to_none)] = None

    @field_validator("trade_date", mode="before")
    @classmethod
    def _trade_date(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return v.date()
        return v

    @property
    def is_trade(self) -> bool:
        return self.event_type is EventType.TRADE

    @property
    def parsed_symbol(self) -> Symbol:
        return Symbol.parse(self.symbol)

    @property
    def muts_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.muts / 1_000_000, tz=UTC)


class L1Event(BaseEvent):
    """One row of the ``L1 TAQ`` dataset (top of book + trade prints)."""

    dataset: ClassVar[Dataset] = Dataset.L1_TAQ

    best_bid: OptFloat = Field(None, description="USD/MWh.")
    best_bid_qty: OptInt = Field(None, description="Lots.")
    best_ask: OptFloat = Field(None, description="USD/MWh.")
    best_ask_qty: OptInt = Field(None, description="Lots.")

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


class BookLevel(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: int = Field(ge=1, le=BOOK_DEPTH)
    price: float
    qty: int


class _L2Base(BaseEvent):
    dataset: ClassVar[Dataset] = Dataset.L2_TAQ

    def _side(self, side: str) -> list[BookLevel]:
        out = []
        for n in range(1, BOOK_DEPTH + 1):
            px = getattr(self, f"{side}_px_{n}")
            qty = getattr(self, f"{side}_qty_{n}")
            if px is not None and qty is not None:
                out.append(BookLevel(level=n, price=px, qty=qty))
        return out

    @property
    def bids(self) -> list[BookLevel]:
        """Populated bid levels, best first."""
        return self._side("bid")

    @property
    def asks(self) -> list[BookLevel]:
        """Populated ask levels, best first."""
        return self._side("ask")

    @property
    def mid(self) -> float | None:
        b, a = getattr(self, "bid_px_1"), getattr(self, "ask_px_1")
        return None if b is None or a is None else (a + b) / 2


_l2_fields: dict[str, Any] = {}
for _n in range(1, BOOK_DEPTH + 1):
    for _side in ("bid", "ask"):
        _l2_fields[f"{_side}_px_{_n}"] = (OptFloat, Field(None, description=f"{_side} price, level {_n} (USD/MWh)."))
        _l2_fields[f"{_side}_qty_{_n}"] = (OptInt, Field(None, description=f"{_side} aggregate size, level {_n} (lots)."))

#: One row of the ``L2 TAQ`` dataset (10-level market-by-price book + trade prints).
L2Event = create_model("L2Event", __base__=_L2Base, __module__=__name__, **_l2_fields)
L2Event.__doc__ = "One row of the ``L2 TAQ`` dataset (10-level market-by-price book + trade prints)."

EVENT_MODELS: dict[Dataset, type[BaseEvent]] = {
    Dataset.L1_TAQ: L1Event,
    Dataset.L2_TAQ: L2Event,
}


# --------------------------------------------------------------------------- #
# Dataset catalogue models (``liberator.datasets(schema=True, details=True)``)
# --------------------------------------------------------------------------- #


class ColumnSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    type: str
    group: str | None = None
    description: str | None = None
    display_name: str | None = None


class DatasetStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    first_date: date | None = None
    last_date: date | None = None
    row_count: int | None = None
    symbol_count: int | None = None
    day_count: int | None = None
    updated_at: datetime | None = None


class DatasetStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    message: str | None = None


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    title: str | None = None
    description: str | None = None
    native_timezone: str | None = None
    data_frequency: str | None = None
    stats: DatasetStats = Field(default_factory=DatasetStats)
    status: DatasetStatus = Field(default_factory=DatasetStatus)
    columns: list[ColumnSchema] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _flatten_schema(cls, data: Any) -> Any:
        if isinstance(data, dict) and "columns" not in data:
            schema = (data.get("config") or {}).get("schema")
            if schema is not None:
                data = {**data, "columns": schema}
        return data

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


# --------------------------------------------------------------------------- #
# Query request model
# --------------------------------------------------------------------------- #

DateLike = date | datetime | str


def _to_datetime(v: DateLike) -> datetime:
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime.combine(v, datetime.min.time())
    return datetime.fromisoformat(v)


class QueryParams(BaseModel):
    """A validated TAQ query.

    ``back_to`` / ``as_of`` accept ``date``, ``datetime`` or ISO strings.  Naive
    values are interpreted as America/New_York (as the liberator client does).
    A bare date for ``as_of`` is exclusive, i.e. ``as_of=date(2026, 6, 24)``
    means "up to 2026-06-24 00:00 ET".
    """

    model_config = ConfigDict(frozen=True)

    dataset: Dataset
    back_to: datetime
    as_of: datetime
    symbols: tuple[str, ...] | None = None
    markets: tuple[Market, ...] | None = None
    products: tuple[Product, ...] | None = None
    event_type: EventType | None = None
    fields: tuple[str, ...] | None = None
    where: str | None = Field(None, description="Extra raw SQL WHERE clause, AND-ed on.")

    @field_validator("back_to", "as_of", mode="before")
    @classmethod
    def _coerce_dt(cls, v: Any) -> Any:
        dt = _to_datetime(v) if isinstance(v, (date, str)) else v
        if isinstance(dt, datetime) and dt.tzinfo is None:
            dt = dt.replace(tzinfo=NY_TZ)
        return dt

    @field_validator("symbols", "markets", "products", "fields", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (str, StrEnum)):
            return (v,)
        return tuple(v)

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.as_of <= self.back_to:
            raise ValueError(
                f"as_of ({self.as_of}) must be later than back_to ({self.back_to}); "
                "the API returns nothing otherwise."
            )
        if self.as_of.date() < DATA_START_DATE:
            raise ValueError(f"No data before {DATA_START_DATE} (ERCOT launch).")
        return self

    def where_clause(self) -> str | None:
        """Compile markets / products / event_type / where into one SQL filter."""
        clauses: list[str] = []
        if self.markets:
            clauses.append(
                "(" + " OR ".join(f"contains(symbol, {_sql_str('-' + m.value + '-')})" for m in self.markets) + ")"
            )
        if self.products:
            clauses.append(
                "(" + " OR ".join(f"starts_with(symbol, {_sql_str(p.value)})" for p in self.products) + ")"
            )
        if self.event_type:
            clauses.append(f"event_type = {_sql_str(self.event_type.value)}")
        if self.where:
            clauses.append(f"({self.where})")
        return " AND ".join(clauses) or None

    def to_liberator_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "name": self.dataset.value,
            "back_to": self.back_to,
            "as_of": self.as_of,
        }
        if self.symbols:
            kw["symbols"] = list(self.symbols)
        if self.fields:
            kw["fields"] = list(self.fields)
        if (w := self.where_clause()) is not None:
            kw["where"] = w
        return kw


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"
