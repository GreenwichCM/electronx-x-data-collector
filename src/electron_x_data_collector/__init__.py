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
    "add_mid_spread",
    "add_symbol_columns",
    "book_to_long",
    "iter_models",
    "normalize_frame",
]


def main() -> None:
    ex = ElectronXClient()
    for name, info in ex.dataset_info().items():
        s = info.stats
        print(f"{name}: {s.first_date} -> {s.last_date}, {s.row_count:,} rows, {s.symbol_count:,} symbols")
