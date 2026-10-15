#!/usr/bin/env python3
"""
Embed each maneuver's 14 structured plots into one 2048-d vector via
Qwen3-VL-Embedding-2B (multi-image joint encoding).

Uses the project root .venv (see pyproject.toml). Activate and run:

  source .venv/bin/activate
  python embed_maneuver_plots_qwen3vl.py --smoke
  python embed_maneuver_plots_qwen3vl.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
QWEN_ROOT = ROOT / "Qwen3-VL-Embedding"
DEFAULT_MODEL_DIR = QWEN_ROOT / "models" / "Qwen3-VL-Embedding-2B"
DEFAULT_PLOT_ROOT = ROOT / "Results" / "maneuver_structured_plots_tesla"
DEFAULT_OUT_ROOT = ROOT / "Results" / "qwen3_vl_maneuver_embeddings"

# Same panel order as the Tesla VLM A–D fusion inputs (14 figures → 1 vector).
PANEL_NAMES: tuple[str, ...] = (
    "02_velocity_stacked.png",
    "03_translational_accel_stacked.png",
    "10b_longitudinal_control_response.png",
    "10c_lateral_control_response.png",
    "02c_vy_vs_vx_timecolor.png",
    "04_rotational_dynamics_stacked.png",
    "05_trajectory_xy_timecolor.png",
    "06_sideslip_stacked.png",
    "07_longslip_stacked.png",
    "09_wheel_speeds_stacked.png",
    "08_Fx_stacked.png",
    "08b_Fy_stacked.png",
    "08d_Fx_vs_Fy_timecolor.png",
    "08e_sideslip_vs_Fy_timecolor.png",
)


def _ensure_qwen_on_path() -> None:
    qwen = str(QWEN_ROOT)
    if qwen not in sys.path:
        sys.path.insert(0, qwen)


def resolve_panel_paths(plot_dir: Path) -> list[Path]:
    paths = [(plot_dir / name).resolve() for name in PANEL_NAMES]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{plot_dir.name}: missing {len(missing)}/{len(PANEL_NAMES)} panels: "
            + ", ".join(Path(m).name for m in missing[:5])
        )
    return paths


def discover_maneuvers(plot_root: Path) -> list[Path]:
    dirs = sorted(
        p for p in plot_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    ok: list[Path] = []
    for d in dirs:
        try:
            resolve_panel_paths(d)
            ok.append(d)
        except FileNotFoundError:
            continue
    return ok


def load_embedder(
    model_dir: Path,
    max_length: int,
    max_pixels: int,
    dtype: str,
):
    _ensure_qwen_on_path()
    import torch
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL-Embedding-2B")

    print(
        f"Loading {model_dir}  dtype={dtype}  device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    t0 = time.time()
    model = Qwen3VLEmbedder(
        model_name_or_path=str(model_dir),
        max_length=max_length,
        max_pixels=max_pixels,
        torch_dtype=torch_dtype,
    )
    print(f"Model ready in {time.time() - t0:.1f}s", flush=True)
    return model


def embed_maneuver(model, plot_dir: Path) -> np.ndarray:
    images = [str(p) for p in resolve_panel_paths(plot_dir)]
    embeddings = model.process([{"image": images}])
    vector = embeddings[0].float().cpu().numpy().astype(np.float32, copy=False)
    if vector.ndim != 1:
        raise RuntimeError(f"Expected 1-D embedding, got shape={vector.shape}")
    return vector


def write_embedding(
    out_dir: Path,
    maneuver: str,
    vector: np.ndarray,
    meta: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{maneuver}.npy", vector)
    (out_dir / f"{maneuver}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_batch(
    model,
    maneuvers: list[Path],
    out_root: Path,
    *,
    resume: bool,
    model_dir: Path,
    max_length: int,
    max_pixels: int,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": str(model_dir),
        "dim": None,
        "n_panels": len(PANEL_NAMES),
        "panels": list(PANEL_NAMES),
        "max_length": max_length,
        "max_pixels": max_pixels,
        "processed": [],
        "skipped": [],
        "failed": [],
    }

    for i, plot_dir in enumerate(maneuvers, 1):
        name = plot_dir.name
        npy_path = out_root / f"{name}.npy"
        if resume and npy_path.is_file():
            print(f"[{i}/{len(maneuvers)}] skip (exists): {name}", flush=True)
            summary["skipped"].append(name)
            continue

        print(f"[{i}/{len(maneuvers)}] embed: {name}", flush=True)
        t0 = time.time()
        try:
            vector = embed_maneuver(model, plot_dir)
            dt = time.time() - t0
            meta = {
                "maneuver": name,
                "plot_dir": str(plot_dir.resolve()),
                "n_panels": len(PANEL_NAMES),
                "panels": list(PANEL_NAMES),
                "dim": int(vector.shape[0]),
                "l2_norm": float(np.linalg.norm(vector)),
                "elapsed_s": round(dt, 3),
                "model": str(model_dir),
            }
            write_embedding(out_root, name, vector, meta)
            summary["dim"] = int(vector.shape[0])
            summary["processed"].append(
                {"maneuver": name, "dim": int(vector.shape[0]), "elapsed_s": round(dt, 3)}
            )
            print(
                f"  → dim={vector.shape[0]}  norm={meta['l2_norm']:.4f}  {dt:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — batch continues
            print(f"  FAIL: {exc}", flush=True)
            summary["failed"].append({"maneuver": name, "error": str(exc)})

    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint-embed 14 maneuver plot panels with Qwen3-VL-Embedding-2B."
    )
    p.add_argument(
        "--plot-root",
        type=Path,
        default=DEFAULT_PLOT_ROOT,
        help="Root with one subdirectory per maneuver (default: Tesla structured plots).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Where to write <maneuver>.npy + .json + summary.json.",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local HF snapshot of Qwen/Qwen3-VL-Embedding-2B.",
    )
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--max-pixels", type=int, default=768 * 768)
    p.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    p.add_argument(
        "--maneuver",
        action="append",
        default=[],
        help="Maneuver folder name under --plot-root (repeatable).",
    )
    p.add_argument("--all", action="store_true", help="Process every complete maneuver.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Only allando_v_chirp_a1_v10 (14-image capability check).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip maneuvers that already have a .npy (default: on).",
    )
    p.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Overwrite existing embeddings.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N maneuvers (0 = no limit).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not QWEN_ROOT.is_dir():
        print(f"Missing {QWEN_ROOT}", file=sys.stderr)
        return 1
    if not args.model_dir.is_dir():
        print(
            f"Missing model dir {args.model_dir}\n"
            "  cd Qwen3-VL-Embedding && hf download Qwen/Qwen3-VL-Embedding-2B "
            "--local-dir ./models/Qwen3-VL-Embedding-2B",
            file=sys.stderr,
        )
        return 1
    if not args.plot_root.is_dir():
        print(f"Missing plot root {args.plot_root}", file=sys.stderr)
        return 1

    if args.smoke:
        selected = [args.plot_root / "allando_v_chirp_a1_v10"]
    elif args.maneuver:
        selected = [args.plot_root / m for m in args.maneuver]
    elif args.all:
        selected = discover_maneuvers(args.plot_root)
    else:
        print("Specify --smoke, --all, or --maneuver NAME", file=sys.stderr)
        return 2

    for d in selected:
        if not d.is_dir():
            print(f"Not a maneuver dir: {d}", file=sys.stderr)
            return 1
        resolve_panel_paths(d)

    if args.limit > 0:
        selected = selected[: args.limit]

    print(
        f"maneuvers={len(selected)}  panels={len(PANEL_NAMES)}  "
        f"out={args.out_root}",
        flush=True,
    )

    model = load_embedder(
        args.model_dir,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        dtype=args.dtype,
    )
    summary = run_batch(
        model,
        selected,
        args.out_root,
        resume=args.resume,
        model_dir=args.model_dir,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
    )
    print(
        f"Done. processed={len(summary['processed'])} "
        f"skipped={len(summary['skipped'])} failed={len(summary['failed'])}",
        flush=True,
    )
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
