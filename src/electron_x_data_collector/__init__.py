"""ElectronX historical market data (L1 / L2 TAQ) for research.

Quick start::

    from electron_x_data_collector import ElectronXClient

    ex = ElectronXClient(cache_dir="~/.cache/electronx")
    df = ex.l1("2026-10-14", "2026-10-16", markets="ERCOT", parse_symbols=True)
"""

from .client import Credentials, ElectronXClient
from .frames import add_mid_spread, add_symbol_columns, book_to_long, iter_models, normalize_frame
from .models import (
    BOOK_DEPTH,
    DATA_START_DATE,
    LOT_SIZE_MWH,
    MARKET_LAUNCH_DATES,
    TICK_SIZE,
    AggressorSide,
    BaseEvent,
    BookLevel,
    ColumnSchema,
    Dataset,
    DatasetInfo,
    DatasetStats,
    EventType,
    ExchangeFlag,
    L1Event,
    L2Event,
    Market,
    Product,
    QueryParams,
    Symbol,
)
from .research import (
    add_forward_mids,
    add_quote_state,
    ic_summary,
    information_coefficient,
    summarize,
    trade_markouts,
)

__all__ = [
    "BOOK_DEPTH",
    "DATA_START_DATE",
    "LOT_SIZE_MWH",
    "MARKET_LAUNCH_DATES",
    "TICK_SIZE",
    "AggressorSide",
    "BaseEvent",
    "BookLevel",
    "ColumnSchema",
    "Credentials",
    "Dataset",
    "DatasetInfo",
    "DatasetStats",
    "ElectronXClient",
    "EventType",
    "ExchangeFlag",
    "L1Event",
    "L2Event",
    "Market",
    "Product",
    "QueryParams",
    "Symbol",
    "add_forward_mids",
    "add_mid_spread",
    "add_quote_state",
    "add_symbol_columns",
    "book_to_long",
    "ic_summary",
    "information_coefficient",
    "iter_models",
    "normalize_frame",
    "summarize",
    "trade_markouts",
]
