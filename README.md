# electron-x-data-collector

A typed Python interface to ElectronX historical market data (L1 and L2 TAQ), plus signal-research helpers. It's built on CloudQuant's `liberator.py` client, vendored unchanged.

## Layout

```
docs/                            ElectronX data guide and spec, Liberator API reference (PDF)*
notebooks/exploration.ipynb      Worked examples of every method, from retrieval to research
src/electron_x_data_collector/
  client.py                      ElectronXClient: credentials, queries, day-by-day Parquet cache
  models.py                      Pydantic models, enums, symbol parsing, query validation
  frames.py                      Typing and reshaping of raw DataFrames
  research.py                    Forward returns, trade markouts, information coefficient
  cli.py                         `electron-x-data-collector` console script (dataset coverage)
  _vendor/liberator.py           Third-party Liberator client; don't edit*
tests/                           Run with `uv run pytest`
```

\* Vendor material is kept local only and is git-ignored. After cloning, get `liberator.py` from CloudQuant/ElectronX and put it at `src/electron_x_data_collector/_vendor/liberator.py`; the package won't import without it. The PDFs go in `docs/`.

## Credentials

Put them in a `.env` file in the working directory (it's git-ignored):

```
ELECTRONX_USER=...
ELECTRONX_TOKEN=...
ELECTRONX_URL=...        # optional, defaults to https://marketdata.electronx.com
```

Real environment variables take precedence over `.env`, and `ElectronXClient(user=..., token=...)` overrides both. The names need the `ELECTRONX_` prefix: on Windows, `USERNAME` always holds the OS login name.

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

## Signal research (`electron_x_data_collector.research`)

Tools for checking whether an L1 signal predicts anything before you build a full backtest. All price moves are in ticks, not percent, because power prices can be near zero or negative. Horizons are either time (`"1s"`, `"500ms"`, `pd.Timedelta`) measured on `muts`, or event counts (`int`).

```python
from electron_x_data_collector import (
    add_quote_state, add_forward_mids, trade_markouts,
    information_coefficient, ic_summary, summarize,
)

l1 = ex.l1("2026-10-01", "2026-10-16", markets="ERCOT")
H = ["1s", "10s", "60s", "300s"]

q = add_forward_mids(add_quote_state(l1), H)       # adds bid/ask/mid/spread/imbalance, then mid_<h>, ret_<h>
ic = information_coefficient(q, "imbalance", H)    # Spearman IC per trade_date (or by="symbol", ...)
ic_summary(ic)                                     # mean IC, t-stat, hit rate across groups

mk = trade_markouts(l1, H)                         # one row per trade: eff_spread, markout_<h> (aggressor's view)
summarize(mk, ["eff_spread", *[f"markout_{h}" for h in H]], by="trade_date")
```

- `add_quote_state(df)` gives every row the latest quote at or before it, so a trade row sees the book just before the trade. Trade rows carry no quote columns of their own. A quote with an empty side, or a locked or crossed book, has `mid = NaN`.
- Forward values are NaN past a symbol's last event in the frame (end of the window or expiry), so nothing is extrapolated.
- `markout_<h>` is the aggressor's PnL per lot in ticks. The passive side earns the negative. Positive markouts mean resting orders get adversely selected.
- Treat each day (or symbol) as one observation when you judge significance, not each event: overlapping horizons make per-event t-stats look far stronger than they are.

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
