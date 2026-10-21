"""``electron-x-data-collector`` console script: print dataset coverage."""

from .client import ElectronXClient


def main() -> None:
    ex = ElectronXClient()
    for name, info in ex.dataset_info().items():
        s = info.stats
        print(f"{name}: {s.first_date} -> {s.last_date}, {s.row_count:,} rows, {s.symbol_count:,} symbols")
