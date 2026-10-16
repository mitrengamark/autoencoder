#!/usr/bin/env python3
"""
text_vlm_tesla_v2: ID-focused VLM (A–D) → 4 paragraphs (no fusion) →
Qwen3-VL-Embedding-8B multimodal embed → cosine → filtered .mat.

Does not touch v1 paths (texts/tesla_vlm, chroma_db/tesla_vlm, prompts/).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

import run_tesla_vlm_redundancy_pipeline as v1
from build_filtered_mat_from_removing_json import (
    copy_or_link,
    find_removal_json,
    index_source_mats,
    load_removed,
)
from cosine_similarity import CosineSimilarity
from pca_baseline import GROUP_PREFIXES, THRESHOLDS

MODEL_NAME = "text_vlm_tesla_v2"
DEFAULT_TEXTS_ROOT = "texts/tesla_vlm_v2"
DEFAULT_PROMPTS_DIR = "prompts_v2"
DEFAULT_CHROMA_PATH = "chroma_db/text_vlm_tesla_v2"
DEFAULT_COLLECTION = "text_vlm_tesla_v2"
DEFAULT_MATRICES_DIR = "cosine_similarity_matrices/text_vlm_tesla_v2"
DEFAULT_QWEN_MODEL_DIR = "Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-8B"
QWEN_ROOT = Path(__file__).resolve().parent / "Qwen3-VL-Embedding"

WHEELS = v1.WHEELS

TASK_SPECS: dict[str, dict[str, Any]] = {
    "A": {
        "prompt_file": "task_a.toml",
        "images": [
            "02_velocity_stacked.png",
            "03_translational_accel_stacked.png",
            "10c_lateral_control_response.png",
        ],
        "signals": ["vx", "vy", "ax", "ay", "steeringangel", "yawrate"],
        "derived": ["speed_kmh", "accel_magnitude"],
        "json_output": True,
    },
    "B": {
        "prompt_file": "task_b.toml",
        "images": [
            "02c_vy_vs_vx_timecolor.png",
            "04_rotational_dynamics_stacked.png",
        ],
        "signals": ["vx", "vy", "yaw", "yawrate"],
        "derived": [],
        "json_output": True,
    },
    "C": {
        "prompt_file": "task_c.toml",
        "images": [
            "06_sideslip_stacked.png",
            "07_longslip_stacked.png",
        ],
        "signals": [f"sideslip{w}" for w in WHEELS]
        + [f"longslip{w}" for w in WHEELS],
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
            "08b_Fy_stacked.png",
            "08e_sideslip_vs_Fy_timecolor.png",
            "08_Fx_stacked.png",
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

# Stable order for multimodal embed (10 panels).
EMBED_PANEL_NAMES: tuple[str, ...] = tuple(
    name
    for spec in ("A", "B", "C", "D")
    for name in TASK_SPECS[spec]["images"]
)


def build_description_from_tasks(task_outputs: dict[str, dict]) -> str:
    parts: list[str] = []
    for task_id in ("A", "B", "C", "D"):
        summary = (task_outputs[task_id].get("technical_summary") or "").strip()
        if not summary:
            raise ValueError(f"Task {task_id}: empty technical_summary")
        parts.append(summary)
    return "\n\n".join(parts) + "\n"


def stage_text_v2(
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
    num_ctx: int,
    num_predict: int,
) -> None:
    v1.log(f"text v2: loading prompts from {prompts_dir}")
    system = v1.read_prompt(prompts_dir / "systempromt.toml")
    v1.log(
        f"text v2: model={vlm_model} maneuvers={len(maneuvers)} "
        f"(no fusion; concat A–D technical_summary)"
    )

    for i, maneuver in enumerate(maneuvers, start=1):
        plot_dir = plots_root / maneuver
        out_dir = texts_root / maneuver
        out_dir.mkdir(parents=True, exist_ok=True)
        t_maneuver = time.monotonic()

        with (plot_dir / "features.json").open(encoding="utf-8") as handle:
            features_full = json.load(handle)

        v1.log(f"===== [text v2 {i}/{len(maneuvers)}] {maneuver} =====")
        task_outputs: dict[str, dict] = {}
        try:
            for task_id, spec in TASK_SPECS.items():
                out_path = out_dir / f"{task_id}.json"
                if out_path.is_file() and not overwrite:
                    v1.log(f"  Task {task_id}: skip (exists)")
                    task_outputs[task_id] = v1.load_json_file(out_path)
                    continue

                task_prompt = v1.read_prompt(prompts_dir / spec["prompt_file"])
                images = [plot_dir / name for name in spec["images"]]
                missing = [str(p) for p in images if not p.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing images: {missing}")

                feat_subset = v1.subset_features(
                    features_full, spec["signals"], spec["derived"]
                )
                user_text = (
                    f"Task {task_id}.\n"
                    f"Analyze ONLY the plots and metadata for Task {task_id}. "
                    "Focus on lateral vehicle-model identification (vy, yaw rate, "
                    "sideslip, Fy).\n"
                    "Do not invent or mention any maneuver name or file stem.\n\n"
                    "Numerical metadata (subset for this task):\n"
                    f"{json.dumps(feat_subset, ensure_ascii=False)}"
                )
                obj = v1.run_task_with_retry(
                    host,
                    vlm_model,
                    system,
                    task_prompt + "\n\n" + user_text,
                    images,
                    max_attempts=max_attempts,
                    timeout=timeout,
                    label=task_id,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                )
                with out_path.open("w", encoding="utf-8") as handle:
                    json.dump(obj, handle, indent=2)
                task_outputs[task_id] = obj
                v1.log(f"  Task {task_id}: DONE → {out_path}")

            desc_path = out_dir / "description.txt"
            if desc_path.is_file() and not overwrite:
                v1.log(f"  concat: skip (exists) → {desc_path}")
            else:
                description = build_description_from_tasks(task_outputs)
                desc_path.write_text(description, encoding="utf-8")
                v1.log(
                    f"  concat: wrote {desc_path} "
                    f"({len(description)} chars, 4 paragraphs)"
                )

            v1.log(
                f"===== [text v2 {i}/{len(maneuvers)}] DONE "
                f"in {time.monotonic() - t_maneuver:.1f}s ====="
            )
        except Exception as exc:
            v1.log(f"  ERROR {maneuver}: {exc}")
            traceback.print_exc()
            v1.append_error(
                errors_path,
                {"stage": "text", "maneuver": maneuver, "error": str(exc)},
            )


def _load_qwen_embedder(model_dir: Path):
    qwen = str(QWEN_ROOT)
    if qwen not in sys.path:
        sys.path.insert(0, qwen)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Qwen3-VL-Embedding-8B")
    v1.log(f"Loading Qwen embedder from {model_dir} (bf16)")
    t0 = time.monotonic()
    model = Qwen3VLEmbedder(
        model_name_or_path=str(model_dir),
        max_length=16384,
        max_pixels=768 * 768,
        torch_dtype=torch.bfloat16,
    )
    v1.log(f"Qwen embedder ready in {time.monotonic() - t0:.1f}s")
    return model


def resolve_embed_panels(plot_dir: Path) -> list[str]:
    paths = [(plot_dir / name).resolve() for name in EMBED_PANEL_NAMES]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing embed panels: {missing[:3]}…")
    return [str(p) for p in paths]


def stage_embed_v2(
    *,
    plots_root: Path,
    texts_root: Path,
    chroma_path: Path,
    collection_name: str,
    model_dir: Path,
    maneuvers: list[str],
    overwrite: bool,
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

    embedder = _load_qwen_embedder(model_dir)

    for i, maneuver in enumerate(maneuvers, start=1):
        desc_path = texts_root / maneuver / "description.txt"
        if not desc_path.is_file():
            v1.log(f"[embed v2 {i}/{len(maneuvers)}] skip (no description): {maneuver}")
            continue

        if not overwrite:
            got = collection.get(ids=[maneuver], include=[])
            if got.get("ids"):
                v1.log(f"[embed v2 {i}/{len(maneuvers)}] skip (exists): {maneuver}")
                continue

        text = desc_path.read_text(encoding="utf-8").strip()
        if not text:
            v1.append_error(
                errors_path,
                {"stage": "embed", "maneuver": maneuver, "error": "empty description"},
            )
            continue

        try:
            v1.log(f"[embed v2 {i}/{len(maneuvers)}] START {maneuver}")
            t0 = time.monotonic()
            images = resolve_embed_panels(plots_root / maneuver)
            embeddings = embedder.process([{"text": text, "image": images}])
            vector = (
                embeddings[0].float().cpu().numpy().astype(np.float32, copy=False)
            )
            collection.upsert(
                ids=[maneuver],
                embeddings=[vector.tolist()],
                documents=[text],
                metadatas=[
                    {
                        "maneuver": maneuver,
                        "group": v1.maneuver_group(maneuver),
                        "model": "Qwen3-VL-Embedding-8B",
                        "n_panels": len(images),
                        "dim": int(vector.shape[0]),
                    }
                ],
            )
            dt = time.monotonic() - t0
            v1.log(
                f"[embed v2 {i}/{len(maneuvers)}] DONE {maneuver} "
                f"dim={vector.shape[0]} panels={len(images)} {dt:.1f}s"
            )
        except Exception as exc:
            v1.log(f"  ERROR embed {maneuver}: {exc}")
            traceback.print_exc()
            v1.append_error(
                errors_path,
                {"stage": "embed", "maneuver": maneuver, "error": str(exc)},
            )


def build_groups(maneuvers: list[str]) -> list[list[str]]:
    buckets: dict[str, list[str]] = {p: [] for p in GROUP_PREFIXES}
    for m in maneuvers:
        g = v1.maneuver_group(m)
        if g in buckets:
            buckets[g].append(m)
    return [buckets[p] for p in GROUP_PREFIXES]


def stage_cosine_v2(
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
    v1.log(f"cosine v2: fetching {len(ids)} embeddings")
    result = collection.get(ids=ids, include=["embeddings"])
    id_to_emb = {
        i: np.asarray(e, dtype=np.float64)
        for i, e in zip(result["ids"], result["embeddings"])
    }
    present = [m for m in maneuvers if m in id_to_emb]

    groups = build_groups(present)
    similarity_matrices: dict[int, tuple[list[str], np.ndarray]] = {}

    for idx, group in enumerate(groups, start=1):
        if len(group) < 2:
            continue
        names = sorted(group)
        vectors = np.stack([id_to_emb[m] for m in names], axis=0)
        sim = cosine_similarity(vectors)
        similarity_matrices[idx] = (names, sim)
        v1.log(f"cosine v2 group {idx}: n={len(names)}")

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
        v1.log(
            f"  threshold {threshold}%: removed={n_removed} "
            f"kept={len(present) - n_removed}"
        )


def stage_mat_filter_v2(
    *,
    root: Path,
    source_dir: Path,
    matrices_dir: Path,
    out_root: Path,
    mode: str,
    thresholds: tuple[int, ...] = (90, 95, 98),
) -> None:
    source_mats = index_source_mats(source_dir)
    if not source_mats:
        raise SystemExit(f"No .mat files in {source_dir}")
    universe = set(source_mats.keys())

    for thr in thresholds:
        json_path = find_removal_json(matrices_dir, thr, MODEL_NAME)
        if json_path is None:
            v1.log(f"mat filter v2 {thr}: no removal JSON, skip")
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
        v1.log(f"mat filter v2 {thr}: kept={len(kept_names)} copied={copied} → {out_dir}")


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
    p.add_argument("--texts-root", default=DEFAULT_TEXTS_ROOT)
    p.add_argument("--prompts-dir", default=DEFAULT_PROMPTS_DIR)
    p.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--matrices-dir", default=DEFAULT_MATRICES_DIR)
    p.add_argument("--qwen-model-dir", default=DEFAULT_QWEN_MODEL_DIR)
    p.add_argument("--mat-source", default="data_tesla")
    p.add_argument("--mat-out-root", default="data_filtered_mat")
    p.add_argument("--ollama-host", default=v1.DEFAULT_OLLAMA_HOST)
    p.add_argument("--vlm-model", default=v1.DEFAULT_VLM_MODEL)
    p.add_argument("--vlm-timeout", type=float, default=1800.0)
    p.add_argument("--num-ctx", type=int, default=v1.DEFAULT_NUM_CTX)
    p.add_argument("--num-predict", type=int, default=v1.DEFAULT_NUM_PREDICT)
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
    qwen_model_dir = resolve(args.qwen_model_dir)
    mat_source = resolve(args.mat_source)
    mat_out_root = resolve(args.mat_out_root)
    errors_path = texts_root / "errors.jsonl"

    all_maneuvers = v1.list_maneuver_dirs(plots_root)
    if args.maneuvers:
        maneuvers = [m for m in args.maneuvers if m in all_maneuvers]
    else:
        maneuvers = all_maneuvers
    if args.limit is not None:
        maneuvers = maneuvers[: max(0, args.limit)]

    v1.log(f"text_vlm_tesla_v2  maneuvers={len(maneuvers)}  stages={args.stages}")
    stages = set(args.stages)

    if "text" in stages:
        v1.log("=== stage: text (v2) ===")
        stage_text_v2(
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
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )

    if "embed" in stages:
        v1.log("=== stage: embed (v2 multimodal) ===")
        if not qwen_model_dir.is_dir():
            raise SystemExit(f"Missing Qwen model dir: {qwen_model_dir}")
        stage_embed_v2(
            plots_root=plots_root,
            texts_root=texts_root,
            chroma_path=chroma_path,
            collection_name=args.collection,
            model_dir=qwen_model_dir,
            maneuvers=maneuvers,
            overwrite=args.overwrite,
            errors_path=errors_path,
        )

    if "cosine" in stages:
        v1.log("=== stage: cosine (v2) ===")
        embed_maneuvers = sorted(
            p.name
            for p in texts_root.iterdir()
            if p.is_dir() and (p / "description.txt").is_file()
        )
        stage_cosine_v2(
            chroma_path=chroma_path,
            collection_name=args.collection,
            matrices_dir=matrices_dir,
            maneuvers=embed_maneuvers,
            plot=not args.no_plot,
        )

    if "mat" in stages:
        v1.log("=== stage: mat (v2) ===")
        stage_mat_filter_v2(
            root=root,
            source_dir=mat_source,
            matrices_dir=matrices_dir,
            out_root=mat_out_root,
            mode=args.mat_link_mode,
        )

    v1.log("Done (text_vlm_tesla_v2).")


if __name__ == "__main__":
    main()
