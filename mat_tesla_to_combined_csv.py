#!/usr/bin/env python3
"""
Convert Tesla CarMaker .mat maneuvers under data_tesla/ into BMW-compatible
*_combined.csv files (same column names / length as data_bmw_cutted).

Mirrors the historical data_preprocess.mat_to_csv + merge_csv_for_manoeuvres
flow, but reads the already-bundled per-maneuver .mat directly.

Example:
    .venv/bin/python mat_tesla_to_combined_csv.py \\
        --mat-dir data_tesla \\
        --out-dir data3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

# Match data_bmw_cutted / data_process truncation.
TARGET_N_ROWS = 10805

# Tesla mat key -> BMW combined CSV column name.
RENAME: dict[str, str] = {
    "wheelspeed_FL": "FL",
    "wheelspeed_FR": "FR",
    "wheelspeed_RL": "RL",
    "wheelspeed_RR": "RR",
    "side_slip": "slip",
    "side_slip1": "slip1",
}

# Direct 1:1 channels shared with BMW CSVs.
DIRECT_CHANNELS: tuple[str, ...] = (
    "sideslipFL",
    "sideslipFR",
    "sideslipRL",
    "sideslipRR",
    "longslipFL",
    "longslipFR",
    "longslipRL",
    "longslipRR",
    "FxFL",
    "FxFR",
    "FxRL",
    "FxRR",
    "FyFL",
    "FyFR",
    "FyRL",
    "FyRR",
    "vx",
    "vy",
    "ax",
    "ay",
    "az",
    "yaw",
    "yawrate",
    "roll",
    "rollrate",
    "pitch",
    "pitchrate",
    "x",
    "y",
    "steeringangel",
    "gas",
    "brake",
)

# BMW CSVs contain two columns both named "signal". Map Tesla excitation
# channels the same way (pandas will show the second as signal.1 on reload).
SIGNAL_SOURCES: tuple[str, ...] = ("chirp_signal", "sin_signal")

# Preferred column order = data_bmw_cutted header (incl. duplicate signal).
BMW_COLUMN_ORDER: list[str] = [
    "sideslipFL",
    "yawrate",
    "FyRL",
    "longslipFL",
    "sideslipRL",
    "FxRR",
    "vy",
    "FL",
    "ay",
    "roll",
    "rollrate",
    "slip",
    "pitch",
    "sideslipRR",
    "RL",
    "vx",
    "signal",
    "brake",
    "signal",
    "FxRL",
    "slip1",
    "FR",
    "FyFL",
    "longslipFR",
    "gas",
    "ax",
    "FyRR",
    "longslipRR",
    "FxFL",
    "x",
    "y",
    "steeringangel",
    "RR",
    "yaw",
    "az",
    "FyFR",
    "pitchrate",
    "FxFR",
    "sideslipFR",
    "longslipRL",
]


def maneuver_stem(mat_path: Path) -> str:
    """Strip accidental '_mat' / '_csv' suffixes from Tesla filenames."""
    stem = mat_path.stem
    stem = re.sub(r"_(mat|csv)$", "", stem)
    return stem


def _as_1d(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr


def load_maneuver_series(mat_path: Path) -> dict[str, np.ndarray]:
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    series: dict[str, np.ndarray] = {}

    for src in DIRECT_CHANNELS:
        if src not in mat:
            raise KeyError(f"{mat_path.name}: missing channel '{src}'")
        series[src] = _as_1d(mat[src])

    for src, dst in RENAME.items():
        if src not in mat:
            raise KeyError(f"{mat_path.name}: missing channel '{src}' (-> {dst})")
        series[dst] = _as_1d(mat[src])

    for src in SIGNAL_SOURCES:
        if src not in mat:
            raise KeyError(f"{mat_path.name}: missing channel '{src}' (-> signal)")
        # Keep both under the BMW duplicate name "signal"; order matters for
        # the two header slots below.
        key = f"__signal_from_{src}"
        series[key] = _as_1d(mat[src])

    lengths = {name: len(values) for name, values in series.items()}
    n = min(lengths.values())
    if n < TARGET_N_ROWS:
        raise ValueError(
            f"{mat_path.name}: shortest channel has {n} samples "
            f"(need >= {TARGET_N_ROWS})"
        )
    mismatched = {k: v for k, v in lengths.items() if v != max(lengths.values())}
    if mismatched:
        # Still ok if all >= TARGET; truncate uniformly below.
        pass

    n_keep = TARGET_N_ROWS
    return {name: values[:n_keep] for name, values in series.items()}


def series_to_dataframe(series: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a DataFrame with BMW column order, including duplicate 'signal'."""
    signal_cols = [
        series["__signal_from_chirp_signal"],
        series["__signal_from_sin_signal"],
    ]
    signal_iter = iter(signal_cols)

    columns: list[str] = []
    data: list[np.ndarray] = []
    for name in BMW_COLUMN_ORDER:
        if name == "signal":
            data.append(next(signal_iter))
            columns.append("signal")
        else:
            data.append(series[name])
            columns.append(name)

    # pandas forbids duplicate column names on construction via dict; use
    # ndarray + explicit columns list.
    return pd.DataFrame(np.column_stack(data), columns=columns)


def convert_one(mat_path: Path, out_dir: Path) -> Path:
    stem = maneuver_stem(mat_path)
    series = load_maneuver_series(mat_path)
    df = series_to_dataframe(series)
    out_path = out_dir / f"{stem}_combined.csv"
    df.to_csv(out_path, index=False)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert data_tesla/*.mat to data3/*_combined.csv (BMW-compatible)."
    )
    p.add_argument("--mat-dir", default="data_tesla")
    p.add_argument("--out-dir", default="data3")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of mats to convert (smoke test).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    mat_dir = Path(args.mat_dir)
    out_dir = Path(args.out_dir)
    if not mat_dir.is_dir():
        raise SystemExit(f"mat dir not found: {mat_dir}")

    mats = sorted(mat_dir.glob("*.mat"))
    if args.limit is not None:
        mats = mats[: max(0, args.limit)]
    if not mats:
        raise SystemExit(f"No .mat files under {mat_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(mats)} mats from {mat_dir} -> {out_dir}")
    print(f"target_rows={TARGET_N_ROWS}")

    written: list[str] = []
    for i, mat_path in enumerate(mats, start=1):
        out_path = convert_one(mat_path, out_dir)
        written.append(out_path.name)
        if i == 1 or i % 50 == 0 or i == len(mats):
            print(f"  [{i}/{len(mats)}] {mat_path.name} -> {out_path.name}")

    summary = out_dir / "conversion_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"mat_dir={mat_dir}",
                f"out_dir={out_dir}",
                f"n={len(written)}",
                f"target_rows={TARGET_N_ROWS}",
                *written,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Done. Wrote {len(written)} CSVs under {out_dir.resolve()}")
    print(f"summary={summary.resolve()}")


if __name__ == "__main__":
    main()
