#!/usr/bin/env python3
"""
Generate structured multi-signal figures + numerical features for one maneuver.

Creates synchronized panel / complementary plot variants for the maneuver-level
vision pipeline, plus deterministic feature extraction.

Does NOT generate text descriptions or call an LLM.

Example:
    .venv/bin/python plot_maneuver_structured_panels.py \\
        --data-dir data_bmw_cutted \\
        --maneuver allando_v_chirp_a5_v100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


WHEEL_ORDER = ("FL", "FR", "RL", "RR")
WHEEL_COLORS = {
    "FL": "#1f77b4",
    "FR": "#ff7f0e",
    "RL": "#2ca02c",
    "RR": "#d62728",
}

# Units used on plot axes (best-effort CarMaker / dataset convention).
SIGNAL_UNITS: dict[str, str] = {
    "steeringangel": "deg",
    "gas": "-",
    "brake": "-",
    "vx": "m/s",
    "vy": "m/s",
    "ax": "m/s²",
    "ay": "m/s²",
    "az": "m/s²",
    "yaw": "rad",
    "pitch": "rad",
    "roll": "rad",
    "yawrate": "rad/s",
    "pitchrate": "rad/s",
    "rollrate": "rad/s",
    "x": "m",
    "y": "m",
    "slip": "rad",
    "slip1": "rad",
    "signal": "-",
    "signal_1": "-",
    "FL": "-",
    "FR": "-",
    "RL": "-",
    "RR": "-",
}
for _w in WHEEL_ORDER:
    SIGNAL_UNITS[f"sideslip{_w}"] = "rad"
    SIGNAL_UNITS[f"longslip{_w}"] = "-"
    SIGNAL_UNITS[f"Fx{_w}"] = "N"
    SIGNAL_UNITS[f"Fy{_w}"] = "N"


def unit_of(name: str) -> str:
    if name in SIGNAL_UNITS:
        return SIGNAL_UNITS[name]
    if name.startswith("sideslip"):
        return "rad"
    if name.startswith("longslip"):
        return "-"
    if name.startswith("Fx") or name.startswith("Fy"):
        return "N"
    return "-"


def ylabel_for(name: str) -> str:
    u = unit_of(name)
    return f"{name} [{u}]"


def disambiguate_columns(header: list[str]) -> list[str]:
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


def load_series(
    csv_path: Path, skip_rows: int, dt: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = disambiguate_columns(next(reader))
        rows: list[list[float]] = []
        for raw in reader:
            if not raw or all(cell.strip() == "" for cell in raw):
                continue
            rows.append([float(cell) if cell != "" else float("nan") for cell in raw])

    if skip_rows < 0:
        raise ValueError("skip_rows must be >= 0")
    if skip_rows >= len(rows):
        raise ValueError(f"skip_rows={skip_rows} leaves no points (n={len(rows)})")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    kept = np.asarray(rows[skip_rows:], dtype=np.float64)
    series = {col: kept[:, i] for i, col in enumerate(columns)}
    time_s = np.arange(kept.shape[0], dtype=np.float64) * dt
    return time_s, series


def require(series: dict[str, np.ndarray], names: Iterable[str]) -> list[str]:
    missing = [n for n in names if n not in series]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return list(names)


def savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_stacked(
    time_s: np.ndarray,
    series: dict[str, np.ndarray],
    names: list[str],
    *,
    title: str,
    out_path: Path,
    dpi: int,
    colors: dict[str, str] | None = None,
) -> None:
    n = len(names)
    fig_h = max(2.8, 1.85 * n)
    fig, axes = plt.subplots(
        n, 1, figsize=(11, fig_h), sharex=True, squeeze=False, constrained_layout=True
    )
    for ax, name in zip(axes[:, 0], names):
        color = "#1f4e79"
        if colors and name in colors:
            color = colors[name]
        elif name[-2:] in WHEEL_COLORS and (
            name.startswith("sideslip")
            or name.startswith("longslip")
            or name.startswith("Fx")
            or name.startswith("Fy")
            or name in WHEEL_ORDER
        ):
            color = WHEEL_COLORS[name[-2:]]
        ax.plot(time_s, series[name], linewidth=0.9, color=color)
        ax.set_ylabel(ylabel_for(name), fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=12)
    savefig(fig, out_path, dpi)


def plot_trajectory_xy(
    time_s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    out_path: Path,
    dpi: int,
    min_span_m: float = 5.0,
) -> None:
    """XY path relative to start, zoomed so small lateral motion stays visible."""
    x_rel = x - x[0]
    y_rel = y - y[0]

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    points = np.column_stack([x_rel, y_rel]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", linewidth=2.0)
    lc.set_array(time_s[:-1])
    line = ax.add_collection(lc)
    ax.scatter(x_rel[0], y_rel[0], c="green", s=55, zorder=5, label="start")
    ax.scatter(x_rel[-1], y_rel[-1], c="red", s=55, zorder=5, label="end")

    x_min, x_max = float(np.nanmin(x_rel)), float(np.nanmax(x_rel))
    y_min, y_max = float(np.nanmin(y_rel)), float(np.nanmax(y_rel))
    span_x = max(x_max - x_min, min_span_m)
    span_y = max(y_max - y_min, min_span_m)
    pad_x = 0.08 * span_x
    pad_y = 0.08 * span_y
    if span_x / span_y > 4.0 or span_y / span_x > 4.0:
        ax.set_aspect("auto")
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel("x − x₀ [m]")
    ax.set_ylabel("y − y₀ [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Time [s]")
    savefig(fig, out_path, dpi)


def plot_scatter_timecolor(
    xs: dict[str, np.ndarray],
    ys: dict[str, np.ndarray],
    time_s: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
    subsample: int = 5,
) -> None:
    keys = list(xs.keys())
    n = len(keys)
    idx = np.arange(0, len(time_s), max(1, subsample))

    if n == 1:
        fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
        key = keys[0]
        sc = ax.scatter(
            xs[key][idx],
            ys[key][idx],
            c=time_s[idx],
            cmap="viridis",
            s=8,
            linewidths=0,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.12)
        cbar.set_label("Time [s]")
        savefig(fig, out_path, dpi)
        return

    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 4.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for i, key in enumerate(keys):
        ax = axes[i // ncols, i % ncols]
        sc = ax.scatter(
            xs[key][idx],
            ys[key][idx],
            c=time_s[idx],
            cmap="viridis",
            s=6,
            linewidths=0,
        )
        ax.set_title(key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Time [s]")
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(title, fontsize=12)
    savefig(fig, out_path, dpi)


def axle_mean(series: dict[str, np.ndarray], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    front = 0.5 * (series[f"{prefix}FL"] + series[f"{prefix}FR"])
    rear = 0.5 * (series[f"{prefix}RL"] + series[f"{prefix}RR"])
    return front, rear


def zero_crossings(y: np.ndarray) -> int:
    s = np.sign(y)
    s[s == 0] = 1
    return int(np.sum(s[1:] * s[:-1] < 0))


def signal_features(y: np.ndarray, time_s: np.ndarray, dt: float) -> dict:
    finite = np.isfinite(y)
    if not np.any(finite):
        return {"all_nan": True}
    yy = y[finite]
    i_max = int(np.nanargmax(y))
    i_min = int(np.nanargmin(y))
    rms = float(np.sqrt(np.nanmean(y * y)))
    yd = yy - np.mean(yy)
    spectrum = np.abs(np.fft.rfft(yd))
    freqs = np.fft.rfftfreq(len(yd), d=dt)
    if len(spectrum) > 1:
        k = int(np.argmax(spectrum[1:]) + 1)
        dom_f = float(freqs[k])
    else:
        dom_f = 0.0
    return {
        "min": float(np.nanmin(y)),
        "max": float(np.nanmax(y)),
        "mean": float(np.nanmean(y)),
        "std": float(np.nanstd(y)),
        "rms": rms,
        "peak_abs": float(np.nanmax(np.abs(y))),
        "t_at_max": float(time_s[i_max]),
        "t_at_min": float(time_s[i_min]),
        "zero_crossings": zero_crossings(np.nan_to_num(y, nan=0.0)),
        "dominant_freq_hz": dom_f,
        "start": float(y[0]),
        "end": float(y[-1]),
    }


def extract_features(
    time_s: np.ndarray, series: dict[str, np.ndarray], dt: float, maneuver: str
) -> dict:
    per_signal = {}
    for name, values in series.items():
        if name in EXCLUDED_FEATURE_CHANNELS:
            continue
        feats = signal_features(values, time_s, dt)
        feats["unit"] = unit_of(name)
        per_signal[name] = feats

    derived: dict[str, object] = {}
    if all(k in series for k in ("ax", "ay", "az")):
        a_mag = np.sqrt(series["ax"] ** 2 + series["ay"] ** 2 + series["az"] ** 2)
        feats = signal_features(a_mag, time_s, dt)
        feats["unit"] = "m/s²"
        derived["accel_magnitude"] = feats
    if all(k in series for k in ("vx", "vy")):
        v_mag = np.sqrt(series["vx"] ** 2 + series["vy"] ** 2)
        feats = signal_features(v_mag, time_s, dt)
        feats["unit"] = "m/s"
        derived["speed_magnitude"] = feats
        feats_kmh = signal_features(series["vx"] * 3.6, time_s, dt)
        feats_kmh["unit"] = "km/h"
        derived["speed_kmh"] = feats_kmh

    for prefix, label in (
        ("sideslip", "sideslip"),
        ("longslip", "longslip"),
        ("Fx", "Fx"),
        ("Fy", "Fy"),
    ):
        keys = [f"{prefix}{w}" for w in WHEEL_ORDER]
        if all(k in series for k in keys):
            front, rear = axle_mean(series, prefix)
            uf = unit_of(keys[0])
            ff = signal_features(front, time_s, dt)
            ff["unit"] = uf
            rf = signal_features(rear, time_s, dt)
            rf["unit"] = uf
            derived[f"{label}_front_mean"] = ff
            derived[f"{label}_rear_mean"] = rf

    if "x" in series and "y" in series:
        dx = np.diff(series["x"], prepend=series["x"][0])
        dy = np.diff(series["y"], prepend=series["y"][0])
        path_len = float(np.nansum(np.sqrt(dx * dx + dy * dy)))
        derived["trajectory"] = {
            "path_length_m": path_len,
            "x_start": float(series["x"][0]),
            "y_start": float(series["y"][0]),
            "x_end": float(series["x"][-1]),
            "y_end": float(series["y"][-1]),
            "x_range_m": float(np.nanmax(series["x"]) - np.nanmin(series["x"])),
            "y_range_m": float(np.nanmax(series["y"]) - np.nanmin(series["y"])),
        }

    return {
        "maneuver": maneuver,
        "dt": dt,
        "n_samples": int(time_s.size),
        "duration_s": float(time_s[-1]) if time_s.size else 0.0,
        "signals": per_signal,
        "derived": derived,
    }


def generate_all_figures(
    time_s: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    maneuver: str,
    out_dir: Path,
    dpi: int,
) -> list[str]:
    written: list[str] = []

    def note(path: Path) -> None:
        written.append(str(path.relative_to(out_dir)))

    # 2. Vehicle velocity
    path = out_dir / "02_velocity_stacked.png"
    plot_stacked(
        time_s,
        series,
        require(series, ["vx", "vy"]),
        title=f"{maneuver} — 2. Vehicle velocity",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    path = out_dir / "02c_vy_vs_vx_timecolor.png"
    plot_scatter_timecolor(
        {"vy_vs_vx": series["vx"]},
        {"vy_vs_vx": series["vy"]},
        time_s,
        title=f"{maneuver} — 2c. vy vs vx (time-colored)",
        xlabel="vx [m/s]",
        ylabel="vy [m/s]",
        out_path=path,
        dpi=dpi,
        subsample=3,
    )
    note(path)

    # 3. Translational dynamics
    path = out_dir / "03_translational_accel_stacked.png"
    plot_stacked(
        time_s,
        series,
        require(series, ["ax", "ay", "az"]),
        title=f"{maneuver} — 3. Translational accelerations",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    # 4. Rotational dynamics
    path = out_dir / "04_rotational_dynamics_stacked.png"
    plot_stacked(
        time_s,
        series,
        require(series, ["yaw", "yawrate", "roll", "rollrate", "pitch", "pitchrate"]),
        title=f"{maneuver} — 4. Rotational dynamics",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    # 5. Trajectory (relative XY map only)
    require(series, ["x", "y"])
    path = out_dir / "05_trajectory_xy_timecolor.png"
    plot_trajectory_xy(
        time_s,
        series["x"],
        series["y"],
        title=f"{maneuver} — 5. XY trajectory (relative, time-colored)",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    # 6. Tire sideslip — stacked (not overlay)
    path = out_dir / "06_sideslip_stacked.png"
    plot_stacked(
        time_s,
        series,
        require(series, [f"sideslip{w}" for w in WHEEL_ORDER]),
        title=f"{maneuver} — 6. Tire sideslip",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    # 7. Longitudinal slip — stacked
    path = out_dir / "07_longslip_stacked.png"
    plot_stacked(
        time_s,
        series,
        require(series, [f"longslip{w}" for w in WHEEL_ORDER]),
        title=f"{maneuver} — 7. Longitudinal slip",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    # 8. Tire forces — stacked
    path = out_dir / "08_Fx_stacked.png"
    plot_stacked(
        time_s,
        series,
        require(series, [f"Fx{w}" for w in WHEEL_ORDER]),
        title=f"{maneuver} — 8. Longitudinal tire forces Fx",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    path = out_dir / "08b_Fy_stacked.png"
    plot_stacked(
        time_s,
        series,
        require(series, [f"Fy{w}" for w in WHEEL_ORDER]),
        title=f"{maneuver} — 8b. Lateral tire forces Fy",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    path = out_dir / "08d_Fx_vs_Fy_timecolor.png"
    plot_scatter_timecolor(
        {w: series[f"Fx{w}"] for w in WHEEL_ORDER},
        {w: series[f"Fy{w}"] for w in WHEEL_ORDER},
        time_s,
        title=f"{maneuver} — 8d. Fx–Fy per wheel (time-colored)",
        xlabel="Fx [N]",
        ylabel="Fy [N]",
        out_path=path,
        dpi=dpi,
        subsample=4,
    )
    note(path)

    path = out_dir / "08e_sideslip_vs_Fy_timecolor.png"
    plot_scatter_timecolor(
        {w: series[f"sideslip{w}"] for w in WHEEL_ORDER},
        {w: series[f"Fy{w}"] for w in WHEEL_ORDER},
        time_s,
        title=f"{maneuver} — 8e. sideslip–Fy per wheel (time-colored)",
        xlabel="sideslip [rad]",
        ylabel="Fy [N]",
        out_path=path,
        dpi=dpi,
        subsample=4,
    )
    note(path)

    # 9. Wheel channels only (slip/signal raw channels are omitted)
    path = out_dir / "09_wheel_speeds_stacked.png"
    plot_stacked(
        time_s,
        series,
        require(series, list(WHEEL_ORDER)),
        title=f"{maneuver} — 9. Wheel channels FL/FR/RL/RR",
        out_path=path,
        dpi=dpi,
        colors=WHEEL_COLORS,
    )
    note(path)

    path = out_dir / "10b_longitudinal_control_response.png"
    plot_stacked(
        time_s,
        series,
        require(series, ["gas", "brake", "vx", "ax"]),
        title=f"{maneuver} — 10b. Longitudinal control → response",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    path = out_dir / "10c_lateral_control_response.png"
    plot_stacked(
        time_s,
        series,
        require(series, ["steeringangel", "yawrate", "vy", "ay"]),
        title=f"{maneuver} — 10c. Lateral control → response",
        out_path=path,
        dpi=dpi,
    )
    note(path)

    return written


OBSOLETE_FIGURES = [
    "01_driver_inputs_stacked.png",
    "01b_driver_inputs_overlay.png",
    "01c_driver_pedals_step.png",
    "01d_steering_with_yawrate_stacked.png",
    "02b_velocity_overlay.png",
    "03b_translational_accel_overlay.png",
    "03c_accel_magnitude.png",
    "05b_xy_as_separate_timeseries_stacked.png",
    "06_sideslip_overlay.png",
    "06b_sideslip_axle_means.png",
    "07_longslip_overlay.png",
    "07b_longslip_axle_means.png",
    "08_Fx_overlay.png",
    "08b_Fy_overlay.png",
    "08c_Fx_Fy_stacked_groups.png",
    "09_wheel_speeds_overlay.png",
    "09b_remaining_signals_stacked.png",
    "08f_sideslip_vs_Fx_timecolor.png",
    "10_vehicle_response_overview.png",
]


# Raw channels intentionally excluded from plotting / feature summaries.
EXCLUDED_FEATURE_CHANNELS = {"slip", "slip1", "signal", "signal_1"}


def cleanup_obsolete(out_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in OBSOLETE_FIGURES:
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


GROUP_PREFIXES = [
    "allando_v_savvaltas",
    "allando_v_chirp",
    "allando_v_sin",
    "valtozo_v_savvaltas_gas",
    "valtozo_v_savvaltas_fek",
    "valtozo_v_sin_gas",
    "valtozo_v_sin_fek",
]

# Preferred representative per group (fallback: middle alphabetical file).
PREFERRED_PER_GROUP = {
    "allando_v_savvaltas": "allando_v_savvaltas_magas_v100",
    "allando_v_chirp": "allando_v_chirp_a5_v100",
    "allando_v_sin": "allando_v_sin_a8_f7_v100",
    "valtozo_v_savvaltas_gas": "valtozo_v_savvaltas_gas_kozepes_pedal0_5",
    "valtozo_v_savvaltas_fek": "valtozo_v_savvaltas_fek_kozepes_pedal0_5",
    "valtozo_v_sin_gas": "valtozo_v_sin_gas_a8_f1_pedal0_2",
    "valtozo_v_sin_fek": "valtozo_v_sin_fek_a8_f1_pedal0_2",
}


def list_maneuvers_in_group(data_dir: Path, prefix: str) -> list[str]:
    names: list[str] = []
    for path in sorted(data_dir.glob("*_combined.csv")):
        stem = path.name.replace("_combined.csv", "")
        if stem.startswith(prefix + "_") or stem == prefix:
            names.append(stem)
    return names


def choose_one_per_group(data_dir: Path) -> list[tuple[str, str]]:
    """Return [(group_prefix, maneuver_stem), ...] for all groups that exist."""
    chosen: list[tuple[str, str]] = []
    for prefix in GROUP_PREFIXES:
        opts = list_maneuvers_in_group(data_dir, prefix)
        if not opts:
            print(f"WARNING: no maneuvers for group {prefix} under {data_dir}")
            continue
        preferred = PREFERRED_PER_GROUP.get(prefix)
        if preferred and preferred in opts:
            pick = preferred
        else:
            pick = opts[len(opts) // 2]
        chosen.append((prefix, pick))
    return chosen


def process_maneuver(
    *,
    data_dir: Path,
    maneuver: str,
    out_dir: Path,
    skip_rows: int,
    dt: float,
    dpi: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_csv_path(data_dir, maneuver)
    time_s, series = load_series(csv_path, skip_rows=skip_rows, dt=dt)

    removed = cleanup_obsolete(out_dir)
    written = generate_all_figures(
        time_s, series, maneuver=maneuver, out_dir=out_dir, dpi=dpi
    )
    features = extract_features(time_s, series, dt=dt, maneuver=maneuver)
    features_path = out_dir / "features.json"
    with features_path.open("w", encoding="utf-8") as handle:
        json.dump(features, handle, indent=2)

    manifest = {
        "maneuver": maneuver,
        "csv": str(csv_path),
        "skip_rows": skip_rows,
        "dt": dt,
        "n_samples": int(time_s.size),
        "time_range_s": [0.0, float(time_s[-1]) if time_s.size else 0.0],
        "figures": written,
        "removed_obsolete": removed,
        "features_file": features_path.name,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"maneuver={maneuver}")
    print(f"csv={csv_path}")
    print(f"rows_plotted={time_s.size} dt={dt} time=[0.00, {time_s[-1]:.2f}] s")
    print(f"n_figures={len(written)}")
    if removed:
        print(f"removed_obsolete={len(removed)}")
    print(f"features={features_path}")
    print(f"out_dir={out_dir.resolve()}")
    for name in written:
        print(f"  - {name}")
    print("")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Structured multi-signal maneuver plots + numerical feature extraction "
            "(no text generation)."
        )
    )
    p.add_argument("--data-dir", default="data_bmw_cutted")
    p.add_argument(
        "--maneuver",
        default="allando_v_chirp_a5_v100",
        help="Single maneuver stem (ignored if --one-per-group is set).",
    )
    p.add_argument(
        "--one-per-group",
        action="store_true",
        help="Run one representative maneuver from each of the 7 maneuver groups.",
    )
    p.add_argument(
        "--out-root",
        default="Results/maneuver_structured_plots",
        help="Root output directory (each maneuver gets its own subfolder).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Exact output dir for a single maneuver (default: <out-root>/<maneuver>).",
    )
    p.add_argument("--skip-rows", type=int, default=2500)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--dpi", type=int, default=120)
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)

    if args.one_per_group:
        selected = choose_one_per_group(data_dir)
        if not selected:
            raise SystemExit(f"No maneuver groups found under {data_dir}")
        print(f"Running {len(selected)} group representatives from {data_dir}:")
        for group, maneuver in selected:
            print(f"  [{group}] {maneuver}")
        print("")
        out_root.mkdir(parents=True, exist_ok=True)
        for group, maneuver in selected:
            print(f"=== {group} ===")
            process_maneuver(
                data_dir=data_dir,
                maneuver=maneuver,
                out_dir=out_root / maneuver,
                skip_rows=args.skip_rows,
                dt=args.dt,
                dpi=args.dpi,
            )
        summary = {
            "data_dir": str(data_dir),
            "maneuvers": [{"group": g, "maneuver": m} for g, m in selected],
        }
        summary_path = out_root / "one_per_group_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Done. Processed {len(selected)} maneuvers.")
        print(f"summary={summary_path.resolve()}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else out_root / args.maneuver
    process_maneuver(
        data_dir=data_dir,
        maneuver=args.maneuver,
        out_dir=out_dir,
        skip_rows=args.skip_rows,
        dt=args.dt,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
