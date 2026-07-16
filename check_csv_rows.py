#!/usr/bin/env python3
"""Count rows in CSV files in a folder and print the average."""

import argparse
import sys
from pathlib import Path


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV sorok számlálása egy mappában, átlag kiírásával."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="A mappa, amiben a CSV fájlok vannak (pl. data_bmw_cutted)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Fejléc nélküli adatsorok száma (összes sor - 1)",
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder.is_dir():
        print(f"Hiba: nem létező mappa: {folder}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"Nincs CSV fájl ebben a mappában: {folder}")
        sys.exit(0)

    counts = []
    for path in csv_files:
        n = count_lines(path)
        if args.data_only:
            n = max(0, n - 1)
        counts.append(n)

    avg = sum(counts) / len(counts)
    min_n = min(counts)
    max_n = max(counts)

    label = "adatsor" if args.data_only else "sor (fejlécel együtt)"
    print(f"Mappa: {folder.resolve()}")
    print(f"CSV fájlok száma: {len(counts)}")
    print(f"Min {label}: {min_n}")
    print(f"Max {label}: {max_n}")
    print(f"Átlag {label}: {avg:.2f}")


if __name__ == "__main__":
    main()
