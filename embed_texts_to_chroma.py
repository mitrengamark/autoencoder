#!/usr/bin/env python3
"""
Embed row-level vehicle-state text descriptions into ChromaDB via SGLang.

Uses OpenAI-compatible /v1/embeddings (gte-Qwen2 on port 30001 by default).
Supports resume, batching, and separate Tesla/BMW databases.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator, Sequence, TextIO

# Default dataset -> Chroma persist path mapping
DEFAULT_MAPPINGS: tuple[tuple[str, str, str], ...] = (
    ("data3", "chroma_db/tesla", "tesla"),
    ("data_bmw_cutted", "chroma_db/bmw", "bmw"),
)

DEFAULT_LOG_FILE = "logs/embed_texts_to_chroma.log"
DEFAULT_OOM_SCORE_ADJ = 700


def apply_oom_victim_score(score: int = DEFAULT_OOM_SCORE_ADJ) -> None:
    """Prefer OOM-killing this process over SSH/shells when memory is exhausted."""
    import os

    raw = os.environ.get("OOM_SCORE_ADJ")
    if raw is not None:
        try:
            score = int(raw)
        except ValueError:
            pass
    path = Path("/proc/self/oom_score_adj")
    try:
        path.write_text(str(score), encoding="utf-8")
        print(f"oom_score_adj={path.read_text(encoding='utf-8').strip()} (higher = killed first)")
    except OSError as exc:
        print(f"Could not set oom_score_adj: {exc}", file=sys.stderr)


class Tee:
    """Write the same output to multiple streams (terminal + log file)."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def setup_run_log(log_path: Path) -> TextIO:
    """Truncate log file for a fresh run and tee stdout/stderr into it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log_handle.write(f"=== embed_texts_to_chroma start {started} ===\n")
    log_handle.write(f"command: {' '.join(sys.argv)}\n\n")
    log_handle.flush()
    sys.stdout = Tee(sys.__stdout__, log_handle)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, log_handle)  # type: ignore[assignment]
    print(f"Logging to {log_path} (cleared for this run)")
    return log_handle


def iter_text_lines(txt_path: Path) -> Iterator[tuple[int, str]]:
    """Yield (row_idx, line) from a text file."""
    with txt_path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            text = line.rstrip("\n")
            if text:
                yield row_idx, text


def make_doc_id(maneuver: str, row_idx: int) -> str:
    return f"{maneuver}__{row_idx}"


def maneuver_from_txt(txt_path: Path) -> str:
    return txt_path.name.replace("_combined.txt", "")


def collect_txt_files(texts_root: Path, dataset: str) -> list[Path]:
    dataset_dir = texts_root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Texts directory not found: {dataset_dir}")
    return sorted(
        p
        for p in dataset_dir.glob("*_combined.txt")
        if p.is_file() and not p.name.startswith("._")
    )


def load_progress(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    with progress_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return set(data.get("completed_files", []))


def save_progress(progress_path: Path, completed_files: set[str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_files": sorted(completed_files),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def get_existing_ids(collection, ids: list[str]) -> set[str]:
    """Return subset of ids already stored in Chroma."""
    if not ids:
        return set()
    try:
        result = collection.get(ids=ids, include=[])
        existing = result.get("ids") or []
        return set(existing)
    except Exception:
        return set()


def embed_batch(client, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    # Preserve input order
    by_index = {item.index: item.embedding for item in response.data}
    return [by_index[i] for i in range(len(texts))]


def process_txt_file(
    txt_path: Path,
    collection,
    client,
    model: str,
    dataset: str,
    vehicle: str,
    batch_size: int,
    chroma_batch_size: int,
) -> int:
    maneuver = maneuver_from_txt(txt_path)
    source_txt = str(txt_path)

    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_meta: list[dict] = []
    pending_texts: list[str] = []
    rows_written = 0

    def flush_pending() -> None:
        nonlocal rows_written
        if not pending_texts:
            return

        existing = get_existing_ids(collection, pending_ids)
        if existing:
            keep = [
                i
                for i, doc_id in enumerate(pending_ids)
                if doc_id not in existing
            ]
            if not keep:
                pending_ids.clear()
                pending_docs.clear()
                pending_meta.clear()
                pending_texts.clear()
                return
            ids = [pending_ids[i] for i in keep]
            docs = [pending_docs[i] for i in keep]
            metas = [pending_meta[i] for i in keep]
            texts = [pending_texts[i] for i in keep]
        else:
            ids, docs, metas, texts = (
                pending_ids,
                pending_docs,
                pending_meta,
                pending_texts,
            )

        embeddings = embed_batch(client, model, texts)
        collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )
        rows_written += len(ids)
        pending_ids.clear()
        pending_docs.clear()
        pending_meta.clear()
        pending_texts.clear()

    for row_idx, text in iter_text_lines(txt_path):
        doc_id = make_doc_id(maneuver, row_idx)
        pending_ids.append(doc_id)
        pending_docs.append(text)
        pending_meta.append(
            {
                "dataset": dataset,
                "vehicle": vehicle,
                "maneuver": maneuver,
                "row_idx": row_idx,
                "source_txt": source_txt,
            }
        )
        pending_texts.append(text)

        if len(pending_texts) >= batch_size:
            flush_pending()

    flush_pending()
    return rows_written


def embed_dataset(
    texts_root: Path,
    dataset: str,
    chroma_path: Path,
    collection_name: str,
    vehicle: str,
    base_url: str,
    model: str,
    batch_size: int,
    chroma_batch_size: int,
    resume: bool,
    limit_files: int | None,
) -> None:
    try:
        import chromadb
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with: pip install -r requirements-embeddings.txt"
        ) from exc

    txt_files = collect_txt_files(texts_root, dataset)
    if limit_files is not None:
        txt_files = txt_files[:limit_files]

    total_files = len(txt_files)
    if total_files == 0:
        print(f"No text files found under {texts_root / dataset}", file=sys.stderr)
        return

    chroma_path.mkdir(parents=True, exist_ok=True)
    progress_path = chroma_path / ".embed_progress.json"
    completed_files: set[str] = load_progress(progress_path) if resume else set()

    client_db = chromadb.PersistentClient(path=str(chroma_path))
    collection = client_db.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embed_client = OpenAI(base_url=base_url, api_key="None")

    print(
        f"Embedding {total_files} files from {dataset} -> {chroma_path} "
        f"(collection={collection_name}, resume={resume})"
    )

    total_rows = 0
    t0 = time.time()

    for file_idx, txt_path in enumerate(txt_files, start=1):
        rel = txt_path.name
        if resume and rel in completed_files:
            print(f"[{file_idx}/{total_files}] skip (done): {rel}")
            continue

        file_t0 = time.time()
        try:
            rows = process_txt_file(
                txt_path,
                collection,
                embed_client,
                model,
                dataset,
                vehicle,
                batch_size,
                chroma_batch_size,
            )
        except Exception as exc:
            print(f"[{file_idx}/{total_files}] ERROR {rel}: {exc}", file=sys.stderr)
            traceback.print_exc()
            raise

        total_rows += rows
        completed_files.add(rel)
        save_progress(progress_path, completed_files)

        elapsed = time.time() - file_t0
        rate = rows / elapsed if elapsed > 0 else 0.0
        print(
            f"[{file_idx}/{total_files}] {rel} -> {rows} rows "
            f"({rate:.1f} rows/s, collection count={collection.count()})"
        )

    elapsed_total = time.time() - t0
    print(
        f"Done {dataset}: {total_rows} new rows in {elapsed_total:.1f}s, "
        f"collection count={collection.count()}"
    )


def resolve_mappings(args) -> list[tuple[str, str, str, str]]:
    """Return list of (dataset, chroma_path, vehicle, collection_name)."""
    if args.all:
        mappings = []
        for dataset, chroma_rel, vehicle in DEFAULT_MAPPINGS:
            mappings.append((dataset, chroma_rel, vehicle, args.collection))
        return mappings

    if not args.dataset or not args.chroma_path:
        raise SystemExit("Provide --dataset and --chroma-path, or use --all")

    vehicle = args.vehicle
    if not vehicle:
        if args.dataset == "data3":
            vehicle = "tesla"
        elif args.dataset == "data_bmw_cutted":
            vehicle = "bmw"
        else:
            vehicle = args.dataset

    return [(args.dataset, args.chroma_path, vehicle, args.collection)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed vehicle-state texts into ChromaDB via SGLang."
    )
    parser.add_argument(
        "--texts-root",
        default="texts",
        help="Root folder containing dataset subfolders (default: texts).",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset subfolder under texts-root (e.g. data3, data_bmw_cutted).",
    )
    parser.add_argument(
        "--chroma-path",
        help="Chroma persist directory (e.g. chroma_db/tesla).",
    )
    parser.add_argument(
        "--vehicle",
        help="Vehicle label stored in metadata (default: tesla/bmw from dataset).",
    )
    parser.add_argument(
        "--collection",
        default="vehicle_states",
        help="Chroma collection name (default: vehicle_states).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Embed both data3->chroma_db/tesla and data_bmw_cutted->chroma_db/bmw.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30001/v1",
        help="SGLang OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--model",
        default="Alibaba-NLP/gte-Qwen2-7B-instruct",
        help="Embedding model id passed to /v1/embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Texts per embedding API call (default: 64).",
    )
    parser.add_argument(
        "--chroma-batch-size",
        type=int,
        default=512,
        help="Reserved for future chunking; batch-size drives API calls.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip files listed in .embed_progress.json.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        metavar="N",
        help="Process only the first N text files (smoke test).",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root for relative paths (default: current directory).",
    )
    parser.add_argument(
        "--log-file",
        default=DEFAULT_LOG_FILE,
        help=f"Run log path; truncated at each start (default: {DEFAULT_LOG_FILE}).",
    )
    return parser


def main() -> None:
    apply_oom_victim_score()
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    texts_root = Path(args.texts_root)
    if not texts_root.is_absolute():
        texts_root = root / texts_root

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = root / log_path
    log_handle = setup_run_log(log_path)

    try:
        mappings = resolve_mappings(args)
        for dataset, chroma_rel, vehicle, collection_name in mappings:
            chroma_path = Path(chroma_rel)
            if not chroma_path.is_absolute():
                chroma_path = root / chroma_path
            embed_dataset(
                texts_root=texts_root,
                dataset=dataset,
                chroma_path=chroma_path,
                collection_name=collection_name,
                vehicle=vehicle,
                base_url=args.base_url,
                model=args.model,
                batch_size=args.batch_size,
                chroma_batch_size=args.chroma_batch_size,
                resume=not args.no_resume,
                limit_files=args.limit_files,
            )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"=== embed_texts_to_chroma end {finished} ===")
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_handle.close()


if __name__ == "__main__":
    main()
