"""High-level ElectronX market data client built on the vendored ``liberator.py``."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from . import liberator
from .frames import add_symbol_columns, empty_frame, iter_models, normalize_frame
from .models import (
    NY_TZ,
    BaseEvent,
    ColumnSchema,
    Dataset,
    DateLike,
    DatasetInfo,
    EventType,
    Market,
    Product,
    QueryParams,
)

log = logging.getLogger(__name__)

DEFAULT_URL = "https://marketdata.electronx.com"


@dataclass(frozen=True)
class Credentials:
    user: str
    token: str
    url: str = DEFAULT_URL

    def __repr__(self) -> str:
        return f"Credentials(user={self.user!r}, token='***', url={self.url!r})"

    @classmethod
    def resolve(
        cls,
        user: str | None = None,
        token: str | None = None,
        url: str | None = None,
        auth_file: str | os.PathLike | None = None,
    ) -> Credentials:
        """Explicit args > environment / .env > ``liberator.json``.

        Environment variables: ``ELECTRONX_USER``, ``ELECTRONX_TOKEN``,
        ``ELECTRONX_URL`` (also ``LIBERATOR_USER`` / ``LIBERATOR_TOKEN``,
        ``API_TOKEN`` / ``API_BASE_URL``).  ``USERNAME`` is deliberately *not*
        read: on Windows it is always the OS login name.
        """
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover
            pass

        env = os.environ.get
        user = user or env("ELECTRONX_USER") or env("LIBERATOR_USER")
        token = token or env("ELECTRONX_TOKEN") or env("LIBERATOR_TOKEN") or env("API_TOKEN")
        url = url or env("ELECTRONX_URL") or env("API_BASE_URL") or DEFAULT_URL

        if not (user and token):
            path = Path(auth_file or env("ELECTRONX_AUTH_FILE") or "liberator.json")
            if path.is_file():
                data = json.loads(path.read_text())
                user = user or data.get("user")
                token = token or data.get("token")
        if not (user and token):
            raise ValueError(
                "ElectronX credentials not found. Pass user=/token=, set ELECTRONX_USER and "
                "ELECTRONX_TOKEN (e.g. in .env), or point auth_file= at liberator.json."
            )
        return cls(user=user, token=token, url=url.rstrip("/"))


class ElectronXClient:
    """Research-friendly access to the ElectronX L1 / L2 TAQ datasets.

    >>> ex = ElectronXClient()                       # creds from .env / liberator.json
    >>> ex.datasets()
    ['L1 TAQ', 'L2 TAQ']
    >>> df = ex.l1("2026-10-14", "2026-10-16", markets="ERCOT")
    >>> trades = ex.trades("2026-10-15", "2026-10-16", products="BFUT")

    Long ranges are fetched one New York calendar day at a time; set
    ``cache_dir`` to keep completed days as Parquet so re-runs are instant.
    """

    def __init__(
        self,
        user: str | None = None,
        token: str | None = None,
        url: str | None = None,
        *,
        auth_file: str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
    ) -> None:
        self.credentials = Credentials.resolve(user, token, url, auth_file)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._info: dict[str, DatasetInfo] | None = None

    def __repr__(self) -> str:
        return f"ElectronXClient(user={self.credentials.user!r}, url={self.credentials.url!r}, cache_dir={self.cache_dir!r})"

    @property
    def _auth(self) -> dict[str, str]:
        c = self.credentials
        return {"user": c.user, "token": c.token, "url": c.url}

    # ------------------------------------------------------------------ #
    # Catalogue
    # ------------------------------------------------------------------ #

    def datasets(self) -> list[str]:
        """Names of datasets your account is entitled to."""
        res = liberator.datasets(entitled=True, **self._auth)
        return list(res.get("data", [])) if isinstance(res, dict) else list(res)

    def dataset_info(self, dataset: Dataset | str | None = None, *, refresh: bool = False) -> Any:
        """Coverage stats + column schema. One :class:`DatasetInfo`, or a dict of all of them."""
        if self._info is None or refresh:
            raw = liberator.datasets(entitled=True, schema=True, details=True, **self._auth)
            self._info = {k: DatasetInfo.model_validate(v) for k, v in raw.items()}
        if dataset is None:
            return dict(self._info)
        return self._info[Dataset(dataset).value]

    def schema(self, dataset: Dataset | str) -> list[ColumnSchema]:
        return self.dataset_info(dataset).columns

    def ping(self) -> tuple[str, str, int, str]:
        return liberator.ping(self.credentials.url)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #

    def query(
        self,
        dataset: Dataset | str,
        back_to: DateLike,
        as_of: DateLike,
        *,
        symbols: str | Iterable[str] | None = None,
        markets: Market | str | Iterable[Market | str] | None = None,
        products: Product | str | Iterable[Product | str] | None = None,
        event_type: EventType | str | None = None,
        fields: Iterable[str] | None = None,
        where: str | None = None,
        normalize: bool = True,
        parse_symbols: bool = False,
        by_day: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch ``[back_to, as_of)`` from a TAQ dataset as a DataFrame.

        Args:
            dataset: ``"L1 TAQ"`` / ``"L2 TAQ"`` or a :class:`Dataset`.
            back_to: Earliest time (date, datetime or ISO string; naive = New York time).
            as_of: Latest time; must be after ``back_to``.
            symbols: Exact symbol(s), e.g. ``"BFUT500-ERCOT-HB_NORTH-260701-18"``.
            markets: Restrict to market(s), e.g. ``"ERCOT"`` or ``["PJM", "ISO-NE"]``.
            products: Restrict to product(s): ``"BFUT"`` / ``"BOPT"``.
            event_type: ``"trade"`` or ``"quote"``.
            fields: Column subset (mandatory columns are always returned).
            where: Extra server-side SQL filter, AND-ed with the above.
            normalize: Return typed columns (see :func:`normalize_frame`); False gives raw API output.
            parse_symbols: Add product / market / location / expiry columns.
            by_day: Split into one request per New York calendar day.
            use_cache: Read/write the Parquet cache (only if ``cache_dir`` is set).

        All filters run server-side, so they cut download time as well as size.
        """
        params = QueryParams(
            dataset=dataset,
            back_to=back_to,
            as_of=as_of,
            symbols=symbols,
            markets=markets,
            products=products,
            event_type=event_type,
            fields=fields,
            where=where,
        )
        windows = _day_windows(params.back_to, params.as_of) if by_day else [(params.back_to, params.as_of)]
        frames = []
        for i, (lo, hi) in enumerate(windows, 1):
            log.info("%s %s -> %s (%d/%d)", params.dataset.value, lo, hi, i, len(windows))
            frames.append(self._fetch_window(params.model_copy(update={"back_to": lo, "as_of": hi}), use_cache))
        frames = [f for f in frames if len(f)]
        df = pd.concat(frames, ignore_index=True) if frames else empty_frame(params.dataset)

        if normalize:
            df = normalize_frame(df, params.dataset)
        if parse_symbols and len(df):
            df = add_symbol_columns(df)
        return df

    def l1(self, back_to: DateLike, as_of: DateLike, **kwargs: Any) -> pd.DataFrame:
        """Top-of-book quotes and trades. Same keyword arguments as :meth:`query`."""
        return self.query(Dataset.L1_TAQ, back_to, as_of, **kwargs)

    def l2(self, back_to: DateLike, as_of: DateLike, **kwargs: Any) -> pd.DataFrame:
        """10-level market-by-price book and trades. Same keyword arguments as :meth:`query`."""
        return self.query(Dataset.L2_TAQ, back_to, as_of, **kwargs)

    def trades(
        self, back_to: DateLike, as_of: DateLike, *, dataset: Dataset | str = Dataset.L1_TAQ, **kwargs: Any
    ) -> pd.DataFrame:
        """Trade prints only (``event_type == 'trade'``), trimmed to trade columns."""
        kwargs.setdefault("fields", TRADE_FIELDS)
        return self.query(dataset, back_to, as_of, event_type=EventType.TRADE, **kwargs)

    def quotes(
        self, back_to: DateLike, as_of: DateLike, *, dataset: Dataset | str = Dataset.L1_TAQ, **kwargs: Any
    ) -> pd.DataFrame:
        """Book updates only (``event_type == 'quote'``)."""
        return self.query(dataset, back_to, as_of, event_type=EventType.QUOTE, **kwargs)

    def iter_batches(self, dataset: Dataset | str, back_to: DateLike, as_of: DateLike, **filters: Any) -> Iterator[pa.RecordBatch]:
        """Stream raw Arrow record batches (one request, no cache) for very large pulls."""
        params = QueryParams(dataset=dataset, back_to=back_to, as_of=as_of, **filters)
        yield from liberator.query(**params.to_liberator_kwargs(), **self._auth)

    def iter_events(self, dataset: Dataset | str, back_to: DateLike, as_of: DateLike, **kwargs: Any) -> Iterator[BaseEvent]:
        """Yield each row as a validated :class:`L1Event` / :class:`L2Event`."""
        df = self.query(dataset, back_to, as_of, normalize=False, **kwargs)
        yield from iter_models(df, dataset)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _fetch_window(self, params: QueryParams, use_cache: bool) -> pd.DataFrame:
        path = self._cache_path(params) if use_cache else None
        if path is not None and path.is_file():
            return pd.read_parquet(path)

        batches = list(liberator.query(**params.to_liberator_kwargs(), **self._auth))
        if not batches:
            df = empty_frame(params.dataset)
        else:
            df = pa.Table.from_batches(batches).to_pandas()
            sort = [c for c in ("muts", "_seq") if c in df]
            if sort:
                df = df.sort_values(sort, ignore_index=True)

        if path is not None and _is_final(params.as_of):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        return df

    def _cache_path(self, params: QueryParams) -> Path | None:
        if self.cache_dir is None:
            return None
        key = params.model_dump(mode="json", exclude={"back_to", "as_of", "dataset"})
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
        stamp = f"{params.back_to:%Y%m%dT%H%M%S}_{params.as_of:%Y%m%dT%H%M%S}"
        return self.cache_dir / params.dataset.name / digest / f"{stamp}.parquet"


TRADE_FIELDS = (
    "timestamp", "muts", "transact_time", "trade_date", "publish_date", "symbol",
    "execution_id", "exchange_flags", "event_type",
    "trade_px", "trade_qty", "trade_id", "aggressor_side",
)


def _day_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split ``[start, end)`` on New York midnights."""
    start, end = start.astimezone(NY_TZ), end.astimezone(NY_TZ)
    out = []
    lo = start
    while lo < end:
        next_midnight = datetime.combine(lo.date() + timedelta(days=1), datetime.min.time(), NY_TZ)
        hi = min(next_midnight, end)
        out.append((lo, hi))
        lo = hi
    return out


def _is_final(as_of: datetime) -> bool:
    """Only cache windows whose data is fully published (T-1 lands ~03:00 ET; 06:00 for margin)."""
    now = datetime.now(NY_TZ)
    cutoff = datetime.combine(now.date(), datetime.min.time(), NY_TZ)
    if now.hour < 6:
        cutoff -= timedelta(days=1)
    return as_of <= cutoff
