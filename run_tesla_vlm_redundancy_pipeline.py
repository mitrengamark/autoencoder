#!/usr/bin/env python3
"""
Tesla VLM text → embedding → cosine redundancy → filtered .mat export.

Stages: text, embed, cosine, mat (select with --stages).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from build_filtered_mat_from_removing_json import (
    copy_or_link,
    find_removal_json,
    index_source_mats,
    load_removed,
)
from cosine_similarity import CosineSimilarity
from pca_baseline import GROUP_PREFIXES, THRESHOLDS

DEFAULT_OLLAMA_HOST = "127.0.0.1:11434"
DEFAULT_VLM_MODEL = "qwen3-vl:32b"
DEFAULT_EMBED_MODEL = "qwen3-embedding:latest"
MODEL_NAME = "text_vlm_tesla"

WHEELS = ("FL", "FR", "RL", "RR")

TASK_JSON_KEYS = frozenset(
    {
        "initial_state",
        "temporal_evolution",
        "main_events",
        "signal_relationships",
        "symmetry_asymmetry",
        "oscillatory_behavior",
        "final_state",
        "important_numeric_features",
        "technical_summary",
    }
)

TASK_SPECS: dict[str, dict[str, Any]] = {
    "A": {
        "prompt_file": "task_a.toml",
        "images": [
            "02_velocity_stacked.png",
            "03_translational_accel_stacked.png",
            "10b_longitudinal_control_response.png",
            "10c_lateral_control_response.png",
        ],
        "signals": [
            "vx",
            "vy",
            "ax",
            "ay",
            "az",
            "gas",
            "brake",
            "steeringangel",
            "yawrate",
        ],
        "derived": ["speed_kmh", "accel_magnitude"],
        "json_output": True,
    },
    "B": {
        "prompt_file": "task_b.toml",
        "images": [
            "02c_vy_vs_vx_timecolor.png",
            "04_rotational_dynamics_stacked.png",
            "05_trajectory_xy_timecolor.png",
        ],
        "signals": [
            "vx",
            "vy",
            "yaw",
            "yawrate",
            "roll",
            "rollrate",
            "pitch",
            "pitchrate",
        ],
        "derived": ["trajectory"],
        "json_output": True,
    },
    "C": {
        "prompt_file": "task_c.toml",
        "images": [
            "06_sideslip_stacked.png",
            "07_longslip_stacked.png",
            "09_wheel_speeds_stacked.png",
        ],
        "signals": [f"sideslip{w}" for w in WHEELS]
        + [f"longslip{w}" for w in WHEELS]
        + list(WHEELS),
        "derived": [
            "sideslip_front_mean",
            "sideslip_rear_mean",
            "longslip_front_mean",
            "longslip_rear_mean",
        ],
        "json_output": True,
    },
    "D": {
        "prompt_file": "task_d.toml",
        "images": [
            "08_Fx_stacked.png",
            "08b_Fy_stacked.png",
            "08d_Fx_vs_Fy_timecolor.png",
            "08e_sideslip_vs_Fy_timecolor.png",
        ],
        "signals": [f"Fx{w}" for w in WHEELS]
        + [f"Fy{w}" for w in WHEELS]
        + [f"sideslip{w}" for w in WHEELS],
        "derived": [
            "Fx_front_mean",
            "Fx_rear_mean",
            "Fy_front_mean",
            "Fy_rear_mean",
        ],
        "json_output": True,
    },
}

MAT_FILTER_THRESHOLDS = (90, 95, 98)


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def ollama_base(host: str) -> str:
    return host if host.startswith("http") else f"http://{host}"


def ollama_post(host: str, path: str, payload: dict, timeout: float) -> dict:
    url = f"{ollama_base(host).rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def encode_image_b64(path: Path, max_side: int = 1280) -> str:
    """Encode PNG for Ollama; optionally downscale large plots."""
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                img = img.resize(
                    (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
                )
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:
        return base64.b64encode(path.read_bytes()).decode("ascii")


def ollama_chat(
    host: str,
    model: str,
    system: str,
    user_text: str,
    image_paths: list[Path],
    *,
    json_format: bool,
    timeout: float,
) -> str:
    images = [encode_image_b64(p) for p in image_paths]
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text, "images": images},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    if json_format:
        payload["format"] = "json"
    body = ollama_post(host, "/api/chat", payload, timeout)
    msg = body.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"Empty Ollama response: {body!r}")
    return content.strip()


def ollama_embed(host: str, model: str, text: str, timeout: float) -> list[float]:
    base = f"{ollama_base(host).rstrip('/')}/v1"
    payload = {"model": model, "input": text}
    url = f"{base}/embeddings"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    items = body.get("data") or []
    if not items:
        raise RuntimeError(f"No embedding data: {body!r}")
    emb = items[0].get("embedding")
    if emb is None:
        raise RuntimeError(f"Missing embedding vector: {body!r}")
    return list(emb)


def subset_features(
    features: dict,
    signals: list[str],
    derived: list[str],
) -> dict:
    out: dict[str, Any] = {}
    for key in ("dt", "duration_s", "n_samples", "maneuver", "truncate_after_s"):
        if key in features:
            out[key] = features[key]
    sig_block = features.get("signals") or {}
    out["signals"] = {k: sig_block[k] for k in signals if k in sig_block}
    der_block = features.get("derived") or {}
    out["derived"] = {k: der_block[k] for k in derived if k in der_block}
    return out


def validate_task_json(obj: dict) -> None:
    if not isinstance(obj, dict):
        raise ValueError("Task output is not a JSON object")
    missing = TASK_JSON_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"Missing JSON keys: {sorted(missing)}")


def load_json_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    validate_task_json(data)
    return data


def list_maneuver_dirs(plots_root: Path) -> list[str]:
    names = sorted(
        p.name for p in plots_root.iterdir() if p.is_dir() and (p / "features.json").is_file()
    )
    return names


def maneuver_group(maneuver: str) -> str:
    for prefix in GROUP_PREFIXES:
        if maneuver.startswith(prefix + "_") or maneuver == prefix:
            return prefix
    return "unknown"


def append_error(errors_path: Path, record: dict) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_task_with_retry(
    host: str,
    model: str,
    system: str,
    user_text: str,
    images: list[Path],
    *,
    max_attempts: int,
    timeout: float,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = ollama_chat(
                host,
                model,
                system,
                user_text,
                images,
                json_format=True,
                timeout=timeout,
            )
            obj = json.loads(raw)
            validate_task_json(obj)
            return obj
        except (
            json.JSONDecodeError,
            ValueError,
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_err = exc
            if attempt < max_attempts:
                time.sleep(2.0 * attempt)
    assert last_err is not None
    raise last_err


def stage_text(
    *,
    plots_root: Path,
    texts_root: Path,
    prompts_dir: Path,
    host: str,
    vlm_model: str,
    maneuvers: list[str],
    overwrite: bool,
    max_attempts: int,
    timeout: float,
    errors_path: Path,
) -> None:
    system = read_prompt(prompts_dir / "systempromt.toml")
    fusion_prompt = read_prompt(prompts_dir / "description_fusion.toml")

    for i, maneuver in enumerate(maneuvers, start=1):
        plot_dir = plots_root / maneuver
        out_dir = texts_root / maneuver
        out_dir.mkdir(parents=True, exist_ok=True)

        features_path = plot_dir / "features.json"
        with features_path.open(encoding="utf-8") as handle:
            features_full = json.load(handle)

        print(f"[text {i}/{len(maneuvers)}] {maneuver}", flush=True)

        task_outputs: dict[str, dict] = {}
        try:
            for task_id, spec in TASK_SPECS.items():
                out_path = out_dir / f"{task_id}.json"
                if out_path.is_file() and not overwrite:
                    task_outputs[task_id] = load_json_file(out_path)
                    continue

                task_prompt = read_prompt(prompts_dir / spec["prompt_file"])
                images = [plot_dir / name for name in spec["images"]]
                missing_img = [str(p) for p in images if not p.is_file()]
                if missing_img:
                    raise FileNotFoundError(f"Missing images: {missing_img}")

                feat_subset = subset_features(
                    features_full, spec["signals"], spec["derived"]
                )
                user_text = (
                    f"Task {task_id} for maneuver: {maneuver}\n"
                    f"Analyze ONLY the plots and metadata for Task {task_id}. "
                    "Do not summarize the whole maneuver or duplicate content meant for other tasks.\n\n"
                    "Numerical metadata (subset for this task):\n"
                    f"{json.dumps(feat_subset, indent=2)}"
                )
                obj = run_task_with_retry(
                    host,
                    vlm_model,
                    system,
                    task_prompt + "\n\n" + user_text,
                    images,
                    max_attempts=max_attempts,
                    timeout=timeout,
                )
                with out_path.open("w", encoding="utf-8") as handle:
                    json.dump(obj, handle, indent=2)
                task_outputs[task_id] = obj

            desc_path = out_dir / "description.txt"
            if desc_path.is_file() and not overwrite:
                pass
            else:
                fusion_user = (
                    "Structured analyses A, B, C, D (JSON):\n\n"
                    f"A:\n{json.dumps(task_outputs['A'], indent=2)}\n\n"
                    f"B:\n{json.dumps(task_outputs['B'], indent=2)}\n\n"
                    f"C:\n{json.dumps(task_outputs['C'], indent=2)}\n\n"
                    f"D:\n{json.dumps(task_outputs['D'], indent=2)}"
                )
                desc = ollama_chat(
                    host,
                    vlm_model,
                    fusion_prompt,
                    fusion_user,
                    [],
                    json_format=False,
                    timeout=timeout,
                )
                desc_path.write_text(desc.strip() + "\n", encoding="utf-8")
                digit_count = len(re.findall(r"\d", desc))
                if digit_count:
                    print(
                        f"  WARNING description.txt contains {digit_count} digit chars "
                        "(fusion prompt asks for words-only numbers)",
                        flush=True,
                    )
        except Exception as exc:
            print(f"  ERROR {maneuver}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            append_error(
                errors_path,
                {"stage": "text", "maneuver": maneuver, "error": str(exc)},
            )


def stage_embed(
    *,
    texts_root: Path,
    chroma_path: Path,
    collection_name: str,
    host: str,
    embed_model: str,
    maneuvers: list[str],
    overwrite: bool,
    timeout: float,
    errors_path: Path,
) -> None:
    try:
        import chromadb
    except ImportError as exc:
        raise SystemExit("pip install chromadb") from exc

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for i, maneuver in enumerate(maneuvers, start=1):
        desc_path = texts_root / maneuver / "description.txt"
        if not desc_path.is_file():
            print(f"[embed {i}/{len(maneuvers)}] skip (no description): {maneuver}")
            continue

        if not overwrite:
            got = collection.get(ids=[maneuver], include=[])
            if got.get("ids"):
                print(f"[embed {i}/{len(maneuvers)}] skip (exists): {maneuver}")
                continue

        text = desc_path.read_text(encoding="utf-8").strip()
        if not text:
            append_error(
                errors_path,
                {"stage": "embed", "maneuver": maneuver, "error": "empty description"},
            )
            continue

        try:
            vec = ollama_embed(host, embed_model, text, timeout)
            collection.upsert(
                ids=[maneuver],
                embeddings=[vec],
                documents=[text],
                metadatas=[{"maneuver": maneuver, "group": maneuver_group(maneuver)}],
            )
            print(f"[embed {i}/{len(maneuvers)}] {maneuver} dim={len(vec)}", flush=True)
        except Exception as exc:
            print(f"  ERROR embed {maneuver}: {exc}", file=sys.stderr)
            append_error(
                errors_path,
                {"stage": "embed", "maneuver": maneuver, "error": str(exc)},
            )


def build_groups(maneuvers: list[str]) -> list[list[str]]:
    buckets: dict[str, list[str]] = {p: [] for p in GROUP_PREFIXES}
    for m in maneuvers:
        g = maneuver_group(m)
        if g in buckets:
            buckets[g].append(m)
    return [buckets[p] for p in GROUP_PREFIXES]


def stage_cosine(
    *,
    chroma_path: Path,
    collection_name: str,
    matrices_dir: Path,
    maneuvers: list[str],
    plot: bool,
) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)

    ids = list(maneuvers)
    if not ids:
        print("cosine: no maneuvers", flush=True)
        return

    result = collection.get(ids=ids, include=["embeddings"])
    id_to_emb = {
        i: np.asarray(e, dtype=np.float64)
        for i, e in zip(result["ids"], result["embeddings"])
    }
    present = [m for m in maneuvers if m in id_to_emb]
    missing = sorted(set(maneuvers) - set(present))
    if missing:
        print(f"cosine: missing embeddings for {len(missing)} maneuvers", flush=True)

    groups = build_groups(present)
    similarity_matrices: dict[int, tuple[list[str], np.ndarray]] = {}

    for idx, group in enumerate(groups, start=1):
        if len(group) < 2:
            print(f"cosine group {idx}: skip (n={len(group)})", flush=True)
            continue
        vectors = np.stack([id_to_emb[m] for m in group], axis=0)
        sim = cosine_similarity(vectors)
        similarity_matrices[idx] = (group, sim)

    matrices_dir.mkdir(parents=True, exist_ok=True)
    if plot:
        plotter = CosineSimilarity("", str(matrices_dir), threshold=90, model_name=MODEL_NAME)
        for _idx, (names, matrix) in similarity_matrices.items():
            plotter.plot_confusion_matrix(names, matrix)

    for threshold in THRESHOLDS:
        cos_sim = CosineSimilarity(
            "", str(matrices_dir), threshold=threshold, model_name=MODEL_NAME
        )
        cos_sim.similarity_matrices = similarity_matrices
        redundant = cos_sim.detect_redundancy()
        removed = cos_sim.remove_redundancy(redundant)
        n_removed = sum(len(v) for v in removed.values())
        print(
            f"  threshold {threshold}%: groups_with_pairs={len(redundant)}, "
            f"removed={n_removed}",
            flush=True,
        )


def stage_mat_filter(
    *,
    root: Path,
    source_dir: Path,
    matrices_dir: Path,
    out_root: Path,
    mode: str,
) -> None:
    source_mats = index_source_mats(source_dir)
    if not source_mats:
        raise SystemExit(f"No .mat files in {source_dir}")

    universe = set(source_mats.keys())
    method_dir = matrices_dir
    if not method_dir.is_dir():
        raise SystemExit(f"Missing matrices dir: {method_dir}")

    for thr in MAT_FILTER_THRESHOLDS:
        json_path = find_removal_json(method_dir, thr, MODEL_NAME)
        if json_path is None:
            print(f"mat filter {thr}: no removal JSON, skip", flush=True)
            continue

        removed = load_removed(json_path)
        kept_names = sorted(universe - removed)
        out_dir = out_root / f"{MODEL_NAME}_{thr}"
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "method": MODEL_NAME,
            "threshold": thr,
            "source_json": str(json_path),
            "n_universe": len(universe),
            "n_removed": len(removed),
            "n_kept": len(kept_names),
            "kept": kept_names,
            "removed": sorted(removed),
        }
        (out_dir / "filter_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        copied = 0
        for name in kept_names:
            src = source_mats.get(name)
            if src is None:
                continue
            copy_or_link(src, out_dir / src.name, mode)
            copied += 1
        print(
            f"mat filter {thr}: kept={len(kept_names)} copied={copied} -> {out_dir}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stages",
        nargs="+",
        default=["text", "embed", "cosine", "mat"],
        choices=["text", "embed", "cosine", "mat"],
    )
    p.add_argument(
        "--plots-root",
        default="Results/maneuver_structured_plots_tesla",
    )
    p.add_argument("--texts-root", default="texts/tesla_vlm")
    p.add_argument("--prompts-dir", default="prompts")
    p.add_argument("--chroma-path", default="chroma_db/tesla_vlm")
    p.add_argument("--collection", default="tesla_vlm_descriptions")
    p.add_argument(
        "--matrices-dir",
        default="cosine_similarity_matrices/text_vlm_tesla",
    )
    p.add_argument("--mat-source", default="data_tesla")
    p.add_argument("--mat-out-root", default="data_filtered_mat")
    p.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    p.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--vlm-timeout", type=float, default=1800.0)
    p.add_argument("--embed-timeout", type=float, default=120.0)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--maneuvers", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--mat-link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
    )
    p.add_argument("--root", default=".")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    plots_root = resolve(args.plots_root)
    texts_root = resolve(args.texts_root)
    prompts_dir = resolve(args.prompts_dir)
    chroma_path = resolve(args.chroma_path)
    matrices_dir = resolve(args.matrices_dir)
    mat_source = resolve(args.mat_source)
    mat_out_root = resolve(args.mat_out_root)
    errors_path = texts_root / "errors.jsonl"

    all_maneuvers = list_maneuver_dirs(plots_root)
    if args.maneuvers:
        maneuvers = [m for m in args.maneuvers if m in all_maneuvers]
        unknown = set(args.maneuvers) - set(maneuvers)
        if unknown:
            print(f"WARNING: unknown maneuvers skipped: {sorted(unknown)}", file=sys.stderr)
    else:
        maneuvers = all_maneuvers
    if args.limit is not None:
        maneuvers = maneuvers[: max(0, args.limit)]

    print(f"Manoeuvers: {len(maneuvers)} (from {plots_root})", flush=True)

    stages = set(args.stages)

    if "text" in stages:
        print("=== stage: text ===", flush=True)
        stage_text(
            plots_root=plots_root,
            texts_root=texts_root,
            prompts_dir=prompts_dir,
            host=args.ollama_host,
            vlm_model=args.vlm_model,
            maneuvers=maneuvers,
            overwrite=args.overwrite,
            max_attempts=args.max_attempts,
            timeout=args.vlm_timeout,
            errors_path=errors_path,
        )

    if "embed" in stages:
        print("=== stage: embed ===", flush=True)
        stage_embed(
            texts_root=texts_root,
            chroma_path=chroma_path,
            collection_name=args.collection,
            host=args.ollama_host,
            embed_model=args.embed_model,
            maneuvers=maneuvers,
            overwrite=args.overwrite,
            timeout=args.embed_timeout,
            errors_path=errors_path,
        )

    if "cosine" in stages:
        print("=== stage: cosine ===", flush=True)
        # Use all maneuvers with descriptions in texts_root for grouping
        embed_maneuvers = sorted(
            p.name
            for p in texts_root.iterdir()
            if p.is_dir() and (p / "description.txt").is_file()
        )
        stage_cosine(
            chroma_path=chroma_path,
            collection_name=args.collection,
            matrices_dir=matrices_dir,
            maneuvers=embed_maneuvers,
            plot=not args.no_plot,
        )

    if "mat" in stages:
        print("=== stage: mat ===", flush=True)
        stage_mat_filter(
            root=root,
            source_dir=mat_source,
            matrices_dir=matrices_dir,
            out_root=mat_out_root,
            mode=args.mat_link_mode,
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
