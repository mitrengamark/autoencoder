#!/usr/bin/env python3
"""
Generate English vehicle-state descriptions from dynamical CSV rows.

Reads every row of *_combined.csv files in data3 and data_bmw_cutted,
writes one .txt per CSV under texts/{dataset}/ with one description per line.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

INTENSITY_LABELS = ("negligible", "slight", "moderate", "strong", "extreme")

# Upper bounds for each intensity bin (exclusive except last).
# First bound is the negligible cutoff: values below it are treated as zero/absent.
THRESHOLDS = {
    "pedal_brake": (0.01, 0.15, 0.4, 0.7, 1.0),
    "pedal_gas": (0.05, 0.3, 0.8, 1.5, 5.0),
    "steer": (0.05, 0.5, 2.0, 5.0, 8.0),
    "accel": (0.2, 1.0, 3.0, 6.0, 10.0),
    "rate": (0.02, 0.1, 0.4, 1.0, 2.5),
    "attitude": (0.005, 0.02, 0.05, 0.1, 0.2),
    "slip_angle": (0.01, 0.05, 0.15, 0.5, 1.5),
    "long_slip": (0.005, 0.02, 0.08, 0.3, 1.0),
    "force": (200.0, 1000.0, 3000.0, 6000.0, 10000.0),
    "speed": (1.0, 5.0, 15.0, 25.0, 40.0),
    "position": (50.0, 200.0, 500.0, 1000.0, 5000.0),
    "load": (0.05, 0.5, 2.0, 10.0, 40.0),
    "load_imbalance": (0.05, 0.15, 0.3, 0.5, 0.8),
}

WHEEL_ORDER = ("front left", "front right", "rear left", "rear right")
WHEEL_KEYS = {
    "front left": "FL",
    "front right": "FR",
    "rear left": "RL",
    "rear right": "RR",
}


def is_absent(value: float, family: str) -> bool:
    """True when the signal is effectively zero for this family."""
    return abs(value) < THRESHOLDS[family][0]


def intensity_level(value: float, family: str) -> str:
    """Map absolute value to intensity label using family thresholds."""
    thresholds = THRESHOLDS[family]
    abs_val = abs(value)
    for label, upper in zip(INTENSITY_LABELS, thresholds):
        if abs_val < upper:
            return label
    return INTENSITY_LABELS[-1]


def direction_word(
    value: float,
    negative: str,
    positive: str,
    neutral: str = "straight",
    family: str = "steer",
) -> str:
    if is_absent(value, family):
        return neutral
    return negative if value < 0 else positive


def fmt_float(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}"


def with_article(phrase: str) -> str:
    """Prefix a/an for phrases starting with intensity adjectives."""
    first = phrase.lstrip().split(" ", 1)[0].lower()
    article = "an" if first[:1] in "aeiou" else "a"
    return f"{article} {phrase}"


def level_adverb(level: str) -> str:
    if level == "extreme":
        return "extremely"
    if level == "moderate":
        return "moderately"
    if level == "slight":
        return "slightly"
    if level == "strong":
        return "strongly"
    if level == "negligible":
        return "negligibly"
    return f"{level}ly"


def get_row_float(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def join_clauses(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def describe_longitudinal(row: dict) -> str:
    vx = get_row_float(row, "vx")
    ax = get_row_float(row, "ax")
    gas = get_row_float(row, "gas")
    brake = get_row_float(row, "brake")

    speed_kmh = vx * 3.6
    speed_level = intensity_level(vx, "speed")
    ax_level = intensity_level(ax, "accel")
    gas_clamped = min(abs(gas), THRESHOLDS["pedal_gas"][-1])
    gas_level = intensity_level(gas_clamped if gas != 0 else 0.0, "pedal_gas")
    brake_level = intensity_level(brake, "pedal_brake")

    if is_absent(vx, "speed"):
        speed_bit = "The car is nearly stopped"
    else:
        speed_bit = (
            f"The car is moving at about {fmt_float(abs(speed_kmh))} km/h "
            f"({speed_level} forward speed"
            f"{', but currently sliding backward' if vx < 0 else ''})"
        )

    bits = [speed_bit]

    if not is_absent(ax, "accel"):
        if ax > 0:
            bits.append(
                f"it is speeding up with {ax_level} forward acceleration "
                f"({fmt_float(ax)} m/s²)"
            )
        else:
            bits.append(
                f"it is slowing down with {ax_level} deceleration "
                f"({fmt_float(abs(ax))} m/s²)"
            )

    if not is_absent(gas, "pedal_gas"):
        bits.append(
            f"the driver is pressing the accelerator with {gas_level} throttle"
        )

    if not is_absent(brake, "pedal_brake"):
        bits.append(f"the brake pedal is pressed with {brake_level} force")

    if (
        is_absent(ax, "accel")
        and is_absent(gas, "pedal_gas")
        and is_absent(brake, "pedal_brake")
    ):
        bits.append("speed is being held roughly steady with no meaningful pedal input")

    return join_clauses(bits) + "."


def describe_lateral(row: dict) -> str:
    steering = get_row_float(row, "steeringangel")
    signal = get_row_float(row, "signal")
    ay = get_row_float(row, "ay")
    yawrate = get_row_float(row, "yawrate")
    yaw = get_row_float(row, "yaw")

    steer_val = steering if abs(steering) >= abs(signal) else signal
    steer_level = intensity_level(steer_val, "steer")
    steer_dir = direction_word(steer_val, "left", "right", "centered", "steer")
    ay_level = intensity_level(ay, "accel")
    ay_dir = direction_word(ay, "left", "right", "neutral", "accel")
    yawrate_level = intensity_level(yawrate, "rate")
    yaw_turn = direction_word(yawrate, "left", "right", "neutral", "rate")
    yaw_level = intensity_level(yaw, "attitude")

    bits = []

    if steer_dir == "centered":
        bits.append("the steering wheel is essentially straight")
    else:
        bits.append(
            f"the driver is turning the steering wheel {level_adverb(steer_level)} to the {steer_dir}"
        )

    if not is_absent(ay, "accel"):
        bits.append(
            f"the car is being pulled sideways to the {ay_dir} with {ay_level} "
            f"lateral acceleration ({fmt_float(abs(ay))} m/s²), like the feeling in a curve"
        )

    if not is_absent(yawrate, "rate"):
        bits.append(
            f"the nose of the car is rotating {yaw_turn}ward at a {yawrate_level} rate"
        )

    if not is_absent(yaw, "attitude"):
        bits.append(
            f"overall the car is pointing in {with_article(level_adverb(yaw_level) + ' rotated')} "
            f"heading relative to its start"
        )

    if len(bits) == 1 and steer_dir == "centered":
        return "The car is traveling straight with no meaningful sideways motion."

    return "Regarding turning: " + join_clauses(bits) + "."


def describe_attitude(row: dict) -> str:
    roll = get_row_float(row, "roll")
    rollrate = get_row_float(row, "rollrate")
    pitch = get_row_float(row, "pitch")
    pitchrate = get_row_float(row, "pitchrate")

    bits = []

    if not is_absent(roll, "attitude"):
        roll_dir = direction_word(roll, "left", "right", "neutral", "attitude")
        bits.append(
            f"the body is leaning {level_adverb(intensity_level(roll, 'attitude'))} toward the {roll_dir} "
            f"(body roll)"
        )
    if not is_absent(rollrate, "rate"):
        bits.append(
            f"that lean is changing at a {intensity_level(rollrate, 'rate')} rate"
        )

    if not is_absent(pitch, "attitude"):
        pitch_dir = direction_word(pitch, "nose-down", "nose-up", "level", "attitude")
        if pitch_dir == "nose-down":
            bits.append(
                f"the nose is dipping {level_adverb(intensity_level(pitch, 'attitude'))} "
                f"(typical under braking)"
            )
        elif pitch_dir == "nose-up":
            bits.append(
                f"the nose is lifting {level_adverb(intensity_level(pitch, 'attitude'))} "
                f"(typical under hard acceleration)"
            )
        else:
            bits.append(f"pitch attitude is {intensity_level(pitch, 'attitude')}")

    if not is_absent(pitchrate, "rate"):
        bits.append(
            f"pitch is changing at a {intensity_level(pitchrate, 'rate')} rate"
        )

    if not bits:
        return "The car body is sitting level, with no noticeable lean or nose dive."

    return "Body motion: " + join_clauses(bits) + "."


def _wheel_slip_phrase(name: str, side: float, long_: float) -> str | None:
    side_absent = is_absent(side, "slip_angle")
    long_absent = is_absent(long_, "long_slip")
    if side_absent and long_absent:
        return None

    bits = []
    if not side_absent:
        side_level = intensity_level(side, "slip_angle")
        bits.append(
            f"{side_level} sideways scrubbing (the tire is sliding a bit across the road)"
        )
    if not long_absent:
        long_level = intensity_level(long_, "long_slip")
        if long_ > 0:
            bits.append(
                f"{long_level} spin-up slip (the wheel is spinning faster than the road speed)"
            )
        else:
            bits.append(
                f"{long_level} lock-up tendency (the wheel is rotating slower than the road speed, as in braking)"
            )

    return f"on the {name} wheel there is " + join_clauses(bits)


def describe_slip(row: dict) -> str:
    phrases = []
    for name, key in WHEEL_KEYS.items():
        phrase = _wheel_slip_phrase(
            name,
            get_row_float(row, f"sideslip{key}"),
            get_row_float(row, f"longslip{key}"),
        )
        if phrase:
            phrases.append(phrase)

    slip = get_row_float(row, "slip")
    if not is_absent(slip, "slip_angle"):
        phrases.append(
            f"overall vehicle slip is {intensity_level(slip, 'slip_angle')}, "
            f"meaning the car's path is not perfectly aligned with where it is pointing"
        )

    if not phrases:
        return "All four tires have good grip with no meaningful slipping."

    return "Tire grip: " + join_clauses(phrases) + "."


def _wheel_force_phrase(name: str, fx: float, fy: float) -> str | None:
    fx_absent = is_absent(fx, "force")
    fy_absent = is_absent(fy, "force")
    if fx_absent and fy_absent:
        return None

    bits = []
    if not fx_absent:
        fx_level = intensity_level(fx, "force")
        if fx > 0:
            bits.append(
                f"{fx_level} forward push from the tire (driving/tractive force)"
            )
        else:
            bits.append(
                f"{fx_level} backward pull from the tire (braking force)"
            )
    if not fy_absent:
        fy_level = intensity_level(fy, "force")
        fy_dir = direction_word(fy, "left", "right", "neutral", "force")
        bits.append(
            f"{fy_level} sideways cornering force toward the {fy_dir}"
        )

    return f"the {name} tire has " + join_clauses(bits)


def describe_tire_forces(row: dict) -> str:
    phrases = []
    for name, key in WHEEL_KEYS.items():
        phrase = _wheel_force_phrase(
            name,
            get_row_float(row, f"Fx{key}"),
            get_row_float(row, f"Fy{key}"),
        )
        if phrase:
            phrases.append(phrase)

    if not phrases:
        return "None of the tires are producing meaningful driving, braking, or cornering forces."

    return "Tire forces: " + join_clauses(phrases) + "."


def describe_wheel_loads(row: dict) -> str:
    loads = {
        name: get_row_float(row, key) for name, key in WHEEL_KEYS.items()
    }
    # Use absolute load magnitude for "how loaded" description
    abs_loads = {n: abs(v) for n, v in loads.items()}
    mean_load = sum(abs_loads.values()) / len(abs_loads)

    if mean_load < THRESHOLDS["load"][0]:
        return "Wheel load signals are effectively zero, so weight distribution is not informative here."

    phrases = []
    for name in WHEEL_ORDER:
        level = intensity_level(abs_loads[name], "load")
        if level == "negligible":
            phrases.append(f"the {name} wheel carries almost no load")
        else:
            phrases.append(
                f"the {name} wheel carries {with_article(level + ' share')} of the weight"
            )

    max_name = max(abs_loads, key=abs_loads.get)
    min_name = min(abs_loads, key=abs_loads.get)
    imbalance = (abs_loads[max_name] - abs_loads[min_name]) / max(mean_load, 1e-9)
    if not is_absent(imbalance, "load_imbalance"):
        phrases.append(
            f"weight has shifted so the {max_name} is most loaded and the {min_name} is least loaded "
            f"({intensity_level(imbalance, 'load_imbalance')} load transfer)"
        )

    return "Weight on the wheels: " + join_clauses(phrases) + "."


def describe_position(row: dict) -> str:
    x = get_row_float(row, "x")
    y = get_row_float(row, "y")
    if is_absent(x, "position") and is_absent(y, "position"):
        return "The car is still near its starting position on the map."
    return (
        f"On the map the car is about {fmt_float(x)} m along X and {fmt_float(y)} m along Y "
        f"from the origin."
    )


def describe_vertical(row: dict) -> str:
    az = get_row_float(row, "az")
    if is_absent(az, "accel"):
        return "There is no noticeable up-or-down bounce."
    az_dir = direction_word(az, "downward", "upward", "neutral", "accel")
    level = intensity_level(az, "accel")
    if az_dir == "upward":
        sense = "as if going over a bump upward"
    elif az_dir == "downward":
        sense = "as if dropping into a dip"
    else:
        sense = "vertically"
    return (
        f"There is {level} vertical acceleration {sense} ({fmt_float(az)} m/s²)."
    )


def describe_row(row: dict) -> str:
    """Build a fixed-schema English description from dynamical values only."""
    clauses = [
        describe_longitudinal(row),
        describe_lateral(row),
        describe_attitude(row),
        describe_slip(row),
        describe_tire_forces(row),
        describe_wheel_loads(row),
        describe_position(row),
        describe_vertical(row),
    ]
    return " ".join(c for c in clauses if c)


def stream_csv_rows(csv_path: Path):
    """Yield row dicts from a CSV, keeping the first occurrence of duplicate columns."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        seen = set()
        indices = []
        fieldnames = []
        for idx, name in enumerate(header):
            if name in seen:
                continue
            seen.add(name)
            indices.append(idx)
            fieldnames.append(name)

        for raw_row in reader:
            yield {name: raw_row[i] for name, i in zip(fieldnames, indices)}


def describe_csv(csv_path: Path, out_path: Path) -> int:
    """Stream one CSV to one TXT file. Returns number of rows written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as out_handle:
        for row in stream_csv_rows(csv_path):
            out_handle.write(describe_row(row) + "\n")
            count += 1
    return count


def collect_csv_files(dataset_dir: Path) -> list[Path]:
    files = sorted(
        p
        for p in dataset_dir.glob("*_combined.csv")
        if p.is_file() and not p.name.startswith("._")
    )
    return files


def process_file(args: tuple[Path, Path]) -> tuple[str, int]:
    csv_path, out_path = args
    count = describe_csv(csv_path, out_path)
    return csv_path.name, count


def run_generation(
    datasets: Sequence[str],
    out_dir: Path,
    root: Path,
    workers: int = 1,
) -> None:
    all_jobs: list[tuple[Path, Path]] = []

    for dataset in datasets:
        dataset_dir = root / dataset
        if not dataset_dir.is_dir():
            print(f"Warning: dataset folder not found: {dataset_dir}", file=sys.stderr)
            continue
        csv_files = collect_csv_files(dataset_dir)
        for csv_path in csv_files:
            rel_txt = csv_path.with_suffix(".txt").name
            out_path = out_dir / dataset / rel_txt
            all_jobs.append((csv_path, out_path))

    total_files = len(all_jobs)
    if total_files == 0:
        print("No CSV files found to process.")
        return

    print(f"Processing {total_files} CSV files with {workers} worker(s)...")

    completed = 0
    total_rows = 0

    if workers <= 1:
        for csv_path, out_path in all_jobs:
            name, count = process_file((csv_path, out_path))
            completed += 1
            total_rows += count
            print(f"[{completed}/{total_files}] {name} -> {count} lines")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_file, job): job for job in all_jobs
            }
            for future in as_completed(futures):
                name, count = future.result()
                completed += 1
                total_rows += count
                print(f"[{completed}/{total_files}] {name} -> {count} lines")

    print(f"Done. Wrote {total_rows} descriptions across {total_files} files.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate English text descriptions from dynamical CSV rows."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data3", "data_bmw_cutted"],
        help="Dataset folders under project root (default: data3 data_bmw_cutted).",
    )
    parser.add_argument(
        "--out-dir",
        default="texts",
        help="Output directory for .txt files (default: texts).",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root containing dataset folders (default: current directory).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers (one CSV per worker, default: 1).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Print N sample descriptions from the first CSV and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    if args.sample:
        for dataset in args.datasets:
            dataset_dir = root / dataset
            files = collect_csv_files(dataset_dir)
            if not files:
                continue
            csv_path = files[0]
            print(f"=== Samples from {csv_path.name} ===")
            for i, row in enumerate(stream_csv_rows(csv_path)):
                if i >= args.sample:
                    break
                print(f"\n--- Row {i} ---")
                print(describe_row(row))
            return

    run_generation(args.datasets, out_dir, root, workers=max(1, args.workers))


if __name__ == "__main__":
    main()
