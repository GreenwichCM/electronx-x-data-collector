# electron-x-data-collector

A typed Python interface to ElectronX historical market data (L1 and L2 TAQ), built on the vendored `liberator.py` client.

## Credentials

The client looks for credentials in this order:

1. Arguments: `ElectronXClient(user=..., token=...)`
2. Environment variables or `.env`: `ELECTRONX_USER`, `ELECTRONX_TOKEN`, and optionally `ELECTRONX_URL`. It also accepts `LIBERATOR_USER`/`LIBERATOR_TOKEN` and `API_TOKEN`/`API_BASE_URL`.
3. A `liberator.json` file in the working directory, or the path passed as `auth_file=`.

It doesn't read `USERNAME`, because on Windows that variable always holds the OS login name.

## Usage

```python
from electron_x_data_collector import ElectronXClient, Symbol

ex = ElectronXClient(cache_dir=".cache/electronx")   # finished days are cached as Parquet

ex.datasets()                        # ['L1 TAQ', 'L2 TAQ']
info = ex.dataset_info("L1 TAQ")     # coverage stats + column schema (pydantic)

# The range is [back_to, as_of). Naive times are New York time. A bare date means midnight.
l1 = ex.l1("2026-10-14", "2026-10-16", markets="ERCOT", parse_symbols=True)
l2 = ex.l2("2026-10-15 10:00", "2026-10-15 12:00", symbols="BFUT500-ERCOT-HB_NORTH-261015-09")
trades = ex.trades("2026-10-01", "2026-10-16", products="BFUT", markets=["PJM", "MISO"])
quotes = ex.quotes("2026-10-15", "2026-10-16", where="contains(symbol, 'HB_NORTH')")

# Validated row objects
for ev in ex.iter_events("L1 TAQ", "2026-10-15 10:00", "2026-10-15 10:05", markets="CAISO"):
    print(ev.symbol, ev.mid, ev.is_trade)

Symbol.parse("BFUT500-ERCOT-HB_NORTH-260701-18").expiry   # datetime(2026, 7, 1, 18, 0)
```

The filters (`symbols`, `markets`, `products`, `event_type`, `fields`, `where`) all run on the server. Unfiltered queries are slow, at roughly 3–4 minutes for 2 hours of all symbols, so filter whenever you can.

### Returned columns

When `normalize=True` (the default), columns come back typed:

| Column | dtype |
|---|---|
| `timestamp` | datetime64, America/New_York |
| `transact_time`, `publish_date` | datetime64, UTC (`transact_time` keeps nanoseconds) |
| `trade_date` | datetime64 |
| prices | float64 in USD/MWh, NaN when not populated |
| quantities | nullable `Int64`, in lots (1 lot = 1 MWh) |
| `event_type`, `exchange_flags`, `aggressor_side` | category |

### Helpers

- `add_symbol_columns(df)` adds `product`, `product_variant`, `market`, `location`, `expiry_hour` and `expiry`.
- `add_mid_spread(df)` adds `mid` and `spread` to L1 or L2 frames.
- `book_to_long(l2_df)` reshapes the wide L2 book to one row per (event, side, level).
- `iter_models(df, dataset)` turns rows into `L1Event` / `L2Event` pydantic models.

## Models (`electron_x_data_collector.models`)

| Model | What it holds |
|---|---|
| `L1Event` | One L1 row |
| `L2Event` | One L2 row: `bid_px_1..10`, `bid_qty_1..10`, `ask_*`, plus `.bids` / `.asks` as `BookLevel` lists |
| `Symbol` | A parsed `<PRODUCT>-<MARKET>-<LOCATION>-<YYMMDD>-<HH>` symbol (handles `ISO-NE`) |
| `QueryParams` | A validated request: checks `as_of > back_to` and that the range isn't before the first launch date |
| `DatasetInfo` | Dataset stats and `ColumnSchema` list |
| Enums | `Dataset`, `Market`, `Product`, `EventType`, `AggressorSide`, `ExchangeFlag` |
| Constants | `MARKET_LAUNCH_DATES`, `TICK_SIZE`, `LOT_SIZE_MWH` |
