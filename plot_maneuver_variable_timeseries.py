#!/usr/bin/env python3
"""
Plot every variable of one maneuver as a simple line chart vs time.

Time axis uses the measurement sampling interval (default 0.01 s).
The first ``skip_rows`` samples are dropped (same convention as text-embed averaging).

Example:
    python3 plot_maneuver_variable_timeseries.py \\
        --data-dir data_bmw_cutted \\
        --maneuver allando_v_chirp_a5_v100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def disambiguate_columns(header: list[str]) -> list[str]:
    """Rename duplicate CSV headers (e.g. signal -> signal, signal_1)."""
    seen: dict[str, int] = {}
    cols: list[str] = []
    for name in header:
        if name in seen:
            seen[name] += 1
            cols.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            cols.append(name)
    return cols


def load_combined_csv(csv_path: Path) -> tuple[list[str], list[list[float]]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = disambiguate_columns(header)
        rows: list[list[float]] = []
        for raw in reader:
            if not raw or all(cell.strip() == "" for cell in raw):
                continue
            rows.append([float(cell) if cell != "" else float("nan") for cell in raw])
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(f"Inconsistent column count in {csv_path}")
    return columns, rows


def resolve_csv_path(data_dir: Path, maneuver: str) -> Path:
    candidates = [
        data_dir / f"{maneuver}_combined.csv",
        data_dir / f"{maneuver}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"CSV not found for maneuver '{maneuver}' under {data_dir}. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def plot_variables(
    columns: list[str],
    rows: list[list[float]],
    *,
    maneuver: str,
    out_dir: Path,
    skip_rows: int,
    dt: float,
    dpi: int,
) -> int:
    if skip_rows < 0:
        raise ValueError("skip_rows must be >= 0")
    if skip_rows >= len(rows):
        raise ValueError(
            f"skip_rows={skip_rows} leaves no points (n_rows={len(rows)})"
        )
    if dt <= 0:
        raise ValueError("dt must be > 0")

    kept = rows[skip_rows:]
    # Time axis restarts at 0 after skip; sampling interval remains dt.
    time_s = [i * dt for i in range(len(kept))]

    out_dir.mkdir(parents=True, exist_ok=True)
    for col_idx, col in enumerate(columns):
        values = [row[col_idx] for row in kept]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(time_s, values, linewidth=0.8, color="#1f4e79")
        ax.set_title(f"{maneuver} — {col}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", dpi=dpi)
        plt.close(fig)

    return len(columns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot all variables of one maneuver as time-series line charts."
    )
    parser.add_argument(
        "--data-dir",
        default="data_bmw_cutted",
        help="Directory with *_combined.csv files (default: data_bmw_cutted).",
    )
    parser.add_argument(
        "--maneuver",
        default="allando_v_chirp_a5_v100",
        help="Maneuver stem without _combined.csv (default: allando_v_chirp_a5_v100).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for PNGs "
            "(default: Results/maneuver_variable_plots/<maneuver>)."
        ),
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=2500,
        help="Number of leading samples to drop (default: 2500).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Sampling interval in seconds (default: 0.01).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="PNG resolution (default: 120).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    maneuver = args.maneuver
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("Results") / "maneuver_variable_plots" / maneuver
    )

    csv_path = resolve_csv_path(data_dir, maneuver)
    columns, rows = load_combined_csv(csv_path)
    n_plots = plot_variables(
        columns,
        rows,
        maneuver=maneuver,
        out_dir=out_dir,
        skip_rows=args.skip_rows,
        dt=args.dt,
        dpi=args.dpi,
    )

    n_plotted = len(rows) - args.skip_rows
    t1 = (n_plotted - 1) * args.dt
    print(f"maneuver={maneuver}")
    print(f"csv={csv_path}")
    print(f"rows_total={len(rows)} skip_rows={args.skip_rows} rows_plotted={n_plotted}")
    print(f"dt={args.dt}s time_range=[0.00, {t1:.2f}] s")
    print(f"n_plots={n_plots}")
    print(f"out_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
