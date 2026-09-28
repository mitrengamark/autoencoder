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
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
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
# qwen3-vl ships with 262k context + thinking; both make /api/chat glacial.
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = 4096
# Fusion prose is ~200-300 words; avoid burning the full 4096 budget on CoT.
DEFAULT_FUSION_NUM_PREDICT = 1024
# qwen3-vl ignores think=False unless the chat template sees /no_think.
NO_THINK_PREFIX = "/no_think\n"
MODEL_NAME = "text_vlm_tesla"

# region agent log
_DEBUG_LOG_PATH = Path("/home/mark_mitrenga/codes/autoencoder/.cursor/debug-9f4061.log")
_DEBUG_SESSION_ID = "9f4061"


def _agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    *,
    run_id: str = "post-fix",
) -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion

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


def log(msg: str) -> None:
    """Timestamped progress line (always flushed)."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def ollama_base(host: str) -> str:
    return host if host.startswith("http") else f"http://{host}"


class _Heartbeat:
    """Prints waiting status while a blocking Ollama call runs."""

    def __init__(self, label: str, interval_s: float = 15.0) -> None:
        self.label = label
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="ollama-heartbeat", daemon=True)

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            elapsed = time.monotonic() - self._t0
            log(f"  ... still waiting on Ollama ({self.label}) — {elapsed:.0f}s elapsed")


def ollama_post(host: str, path: str, payload: dict, timeout: float, label: str) -> dict:
    url = f"{ollama_base(host).rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    payload_mb = len(data) / (1024 * 1024)
    log(
        f"  → POST {url}  model={payload.get('model')}  "
        f"payload={payload_mb:.1f} MiB  timeout={timeout:.0f}s  ({label})"
    )
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with _Heartbeat(label):
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    elapsed = time.monotonic() - t0
    log(f"  ← Ollama OK in {elapsed:.1f}s  response={len(raw) / 1024:.1f} KiB  ({label})")
    return json.loads(raw)


def encode_image_b64(path: Path, max_side: int = 1280) -> str:
    """Encode PNG for Ollama; optionally downscale large plots."""
    src_kb = path.stat().st_size / 1024
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
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            log(
                f"  image {path.name}: {w}x{h} → JPEG "
                f"{len(buf.getvalue()) / 1024:.0f} KiB (src {src_kb:.0f} KiB)"
            )
            return encoded
    except ImportError:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        log(f"  image {path.name}: raw PNG {src_kb:.0f} KiB (no Pillow)")
        return encoded


def _looks_like_cot(text: str) -> bool:
    """Detect chain-of-thought dumps mistakenly used as the final answer."""
    head = text.lstrip()[:400].lower()
    markers = (
        "<think>",
        "we are given",
        "we must fuse",
        "let's break down",
        "steps:",
        "step 1",
        "now, we must write",
        "conceptual order",
    )
    return any(m in head for m in markers)


def _ensure_no_think(system: str) -> str:
    s = system.lstrip()
    if s.startswith("/no_think"):
        return system
    return NO_THINK_PREFIX + system


def _unwrap_fusion_description(text: str) -> str:
    """If fusion returned JSON {\"description\": \"...\"}, unwrap to prose."""
    raw = text.strip()
    if not raw.startswith("{"):
        return raw
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(obj, dict) and isinstance(obj.get("description"), str):
        return obj["description"].strip()
    return raw


def ollama_chat(
    host: str,
    model: str,
    system: str,
    user_text: str,
    image_paths: list[Path],
    *,
    json_format: bool,
    timeout: float,
    label: str = "chat",
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    think: bool = False,
) -> str:
    system = _ensure_no_think(system)
    log(
        f"  encode {len(image_paths)} image(s) for {label} "
        f"(system={len(system)} chars, user={len(user_text)} chars)"
    )
    user_msg: dict[str, Any] = {"role": "user", "content": user_text}
    if image_paths:
        user_msg["images"] = [encode_image_b64(p) for p in image_paths]
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            user_msg,
        ],
        "stream": False,
        # Qwen3-VL defaults to thinking=on; think=False alone is unreliable —
        # /no_think in the system prompt is what actually suppresses CoT.
        "think": think,
        "options": {
            "temperature": 0,
            # Model default context is 262144 — keep this modest.
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
        },
    }
    if json_format:
        payload["format"] = "json"
    log(
        f"  chat options: think={think} num_ctx={num_ctx} "
        f"num_predict={num_predict} format_json={json_format} "
        f"no_think_prefix={system.lstrip().startswith('/no_think')}"
    )
    # region agent log
    _agent_debug_log(
        "H1",
        "run_tesla_vlm_redundancy_pipeline.py:ollama_chat:request",
        "ollama chat request options",
        {
            "label": label,
            "think": think,
            "json_format": json_format,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "has_no_think_prefix": system.lstrip().startswith("/no_think"),
            "n_images": len(image_paths),
            "system_chars": len(system),
            "user_chars": len(user_text),
        },
    )
    # endregion
    body = ollama_post(host, "/api/chat", payload, timeout, label=label)
    msg = body.get("message") or {}
    content = (msg.get("content") or "").strip()
    thinking = (msg.get("thinking") or body.get("thinking") or "").strip()
    eval_count = body.get("eval_count")
    prompt_eval_count = body.get("prompt_eval_count")
    used_thinking_fallback = False
    # qwen3-vl often leaves content empty and puts the answer in "thinking"
    # even when think=False; with /no_think that field is usually the answer
    # itself (not a multi-page CoT plan).
    if not content and thinking:
        log(
            f"  WARNING: empty content, using thinking field "
            f"({len(thinking)} chars)  ({label})"
        )
        content = thinking
        used_thinking_fallback = True
    elif thinking:
        log(f"  (model also returned thinking={len(thinking)} chars)")
    if not content:
        raise RuntimeError(f"Empty Ollama response: {body!r}")
    cot_like = _looks_like_cot(content)
    digit_count = len(re.findall(r"\d", content))
    # region agent log
    _agent_debug_log(
        "H2",
        "run_tesla_vlm_redundancy_pipeline.py:ollama_chat:response",
        "ollama chat response shape",
        {
            "label": label,
            "content_len": len(msg.get("content") or ""),
            "thinking_len": len(thinking),
            "used_thinking_fallback": used_thinking_fallback,
            "cot_like": cot_like,
            "digit_count": digit_count,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "hit_num_predict": (
                isinstance(eval_count, int) and eval_count >= int(num_predict)
            ),
            "preview": content[:180],
        },
    )
    # endregion
    if cot_like:
        log(
            f"  WARNING: reply looks like chain-of-thought, not final answer "
            f"({label})"
        )
    log(f"  reply length={len(content)} chars  ({label})")
    return content


def ollama_embed(host: str, model: str, text: str, timeout: float) -> list[float]:
    base = f"{ollama_base(host).rstrip('/')}/v1"
    payload = {"model": model, "input": text}
    url = f"{base}/embeddings"
    data = json.dumps(payload).encode("utf-8")
    log(
        f"  → POST {url}  model={model}  text={len(text)} chars  "
        f"timeout={timeout:.0f}s  (embed)"
    )
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
        method="POST",
    )
    t0 = time.monotonic()
    with _Heartbeat("embed"):
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.monotonic() - t0
    items = body.get("data") or []
    if not items:
        raise RuntimeError(f"No embedding data: {body!r}")
    emb = items[0].get("embedding")
    if emb is None:
        raise RuntimeError(f"Missing embedding vector: {body!r}")
    log(f"  ← embed OK in {elapsed:.1f}s  dim={len(emb)}")
    return list(emb)


def subset_features(
    features: dict,
    signals: list[str],
    derived: list[str],
) -> dict:
    out: dict[str, Any] = {}
    # Never forward maneuver/file identifiers to the VLM.
    for key in ("dt", "duration_s", "n_samples", "truncate_after_s"):
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
    label: str,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        log(f"  Task {label}: attempt {attempt}/{max_attempts}")
        try:
            raw = ollama_chat(
                host,
                model,
                system,
                user_text,
                images,
                json_format=True,
                timeout=timeout,
                label=f"task_{label}",
                num_ctx=num_ctx,
                num_predict=num_predict,
                think=False,
            )
            if _looks_like_cot(raw):
                raise ValueError(
                    f"Task {label}: model returned chain-of-thought instead of JSON"
                )
            obj = json.loads(raw)
            validate_task_json(obj)
            log(f"  Task {label}: JSON OK")
            return obj
        except (
            json.JSONDecodeError,
            ValueError,
            RuntimeError,
            urllib.error.URLError,
            TimeoutError,
        ) as exc:
            last_err = exc
            log(f"  Task {label}: attempt {attempt} FAILED: {exc}")
            if attempt < max_attempts:
                wait = 2.0 * attempt
                log(f"  Task {label}: retry in {wait:.0f}s")
                time.sleep(wait)
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
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> None:
    log(f"text stage: loading prompts from {prompts_dir}")
    system = read_prompt(prompts_dir / "systempromt.toml")
    fusion_prompt = read_prompt(prompts_dir / "description_fusion.toml")
    log(
        f"text stage: model={vlm_model} host={host} timeout={timeout:.0f}s "
        f"num_ctx={num_ctx} num_predict={num_predict} think=False "
        f"maneuvers={len(maneuvers)} overwrite={overwrite}"
    )

    for i, maneuver in enumerate(maneuvers, start=1):
        plot_dir = plots_root / maneuver
        out_dir = texts_root / maneuver
        out_dir.mkdir(parents=True, exist_ok=True)
        t_maneuver = time.monotonic()

        features_path = plot_dir / "features.json"
        with features_path.open(encoding="utf-8") as handle:
            features_full = json.load(handle)

        log(f"===== [text {i}/{len(maneuvers)}] {maneuver} =====")

        task_outputs: dict[str, dict] = {}
        try:
            for task_id, spec in TASK_SPECS.items():
                out_path = out_dir / f"{task_id}.json"
                if out_path.is_file() and not overwrite:
                    log(f"  Task {task_id}: skip (exists) → {out_path}")
                    task_outputs[task_id] = load_json_file(out_path)
                    continue

                log(
                    f"  Task {task_id}: START  images={spec['images']}  "
                    f"signals={len(spec['signals'])} derived={len(spec['derived'])}"
                )
                t_task = time.monotonic()
                task_prompt = read_prompt(prompts_dir / spec["prompt_file"])
                images = [plot_dir / name for name in spec["images"]]
                missing_img = [str(p) for p in images if not p.is_file()]
                if missing_img:
                    raise FileNotFoundError(f"Missing images: {missing_img}")

                feat_subset = subset_features(
                    features_full, spec["signals"], spec["derived"]
                )
                user_text = (
                    f"Task {task_id}.\n"
                    f"Analyze ONLY the plots and metadata for Task {task_id}. "
                    "Do not summarize the whole recording or duplicate content meant for other tasks.\n"
                    "Do not invent or mention any maneuver name, file stem or run identifier.\n\n"
                    "Numerical metadata (subset for this task):\n"
                    f"{json.dumps(feat_subset, ensure_ascii=False)}"
                )
                obj = run_task_with_retry(
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
                log(
                    f"  Task {task_id}: DONE in {time.monotonic() - t_task:.1f}s → {out_path}"
                )

            desc_path = out_dir / "description.txt"
            if desc_path.is_file() and not overwrite:
                log(f"  fusion: skip (exists) → {desc_path}")
            else:
                log("  fusion: START (description_fusion, no images)")
                t_fus = time.monotonic()
                # Compact JSON (no indent) cuts prompt tokens; VL still needed
                # for consistency with A–D but fusion is text-only.
                fusion_user = (
                    "Structured analyses A, B, C, D (JSON):\n\n"
                    f"A:\n{json.dumps(task_outputs['A'], ensure_ascii=False)}\n\n"
                    f"B:\n{json.dumps(task_outputs['B'], ensure_ascii=False)}\n\n"
                    f"C:\n{json.dumps(task_outputs['C'], ensure_ascii=False)}\n\n"
                    f"D:\n{json.dumps(task_outputs['D'], ensure_ascii=False)}\n\n"
                    "Return ONLY JSON of the form "
                    '{"description":"<one continuous 200-300 word paragraph>"}. '
                    "Inside description: spell every number and unit in words; "
                    "no digits; no planning notes; no lists."
                )
                # Plain-text fusion lets qwen3-vl burn num_predict on CoT.
                # format=json + /no_think yields the paragraph directly.
                raw_desc = ollama_chat(
                    host,
                    vlm_model,
                    fusion_prompt,
                    fusion_user,
                    [],
                    json_format=True,
                    timeout=timeout,
                    label="fusion",
                    num_ctx=num_ctx,
                    num_predict=min(num_predict, DEFAULT_FUSION_NUM_PREDICT),
                    think=False,
                )
                desc = _unwrap_fusion_description(raw_desc)
                if _looks_like_cot(desc):
                    # region agent log
                    _agent_debug_log(
                        "H3",
                        "run_tesla_vlm_redundancy_pipeline.py:fusion:cot_reject",
                        "fusion output rejected as CoT",
                        {
                            "raw_len": len(raw_desc),
                            "desc_len": len(desc),
                            "preview": desc[:240],
                        },
                    )
                    # endregion
                    raise RuntimeError(
                        "Fusion returned chain-of-thought instead of the "
                        "final description paragraph"
                    )
                desc_path.write_text(desc.strip() + "\n", encoding="utf-8")
                digit_count = len(re.findall(r"\d", desc))
                word_count = len(desc.split())
                # region agent log
                _agent_debug_log(
                    "H3",
                    "run_tesla_vlm_redundancy_pipeline.py:fusion:done",
                    "fusion description quality",
                    {
                        "chars": len(desc),
                        "words": word_count,
                        "digits": digit_count,
                        "cot_like": _looks_like_cot(desc),
                        "preview": desc[:240],
                    },
                )
                # endregion
                log(
                    f"  fusion: DONE in {time.monotonic() - t_fus:.1f}s → {desc_path} "
                    f"({len(desc)} chars, words={word_count}, digits={digit_count})"
                )
                if digit_count:
                    log(
                        "  WARNING: description.txt contains digit chars "
                        "(fusion prompt asks for words-only numbers)"
                    )
            log(
                f"===== [text {i}/{len(maneuvers)}] {maneuver} DONE "
                f"in {time.monotonic() - t_maneuver:.1f}s ====="
            )
        except Exception as exc:
            log(f"  ERROR {maneuver}: {exc}")
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
            log(f"[embed {i}/{len(maneuvers)}] skip (no description): {maneuver}")
            continue

        if not overwrite:
            got = collection.get(ids=[maneuver], include=[])
            if got.get("ids"):
                log(f"[embed {i}/{len(maneuvers)}] skip (exists): {maneuver}")
                continue

        text = desc_path.read_text(encoding="utf-8").strip()
        if not text:
            append_error(
                errors_path,
                {"stage": "embed", "maneuver": maneuver, "error": "empty description"},
            )
            continue

        try:
            log(f"[embed {i}/{len(maneuvers)}] START {maneuver}")
            vec = ollama_embed(host, embed_model, text, timeout)
            collection.upsert(
                ids=[maneuver],
                embeddings=[vec],
                documents=[text],
                metadatas=[{"maneuver": maneuver, "group": maneuver_group(maneuver)}],
            )
            log(f"[embed {i}/{len(maneuvers)}] DONE {maneuver} dim={len(vec)}")
        except Exception as exc:
            log(f"  ERROR embed {maneuver}: {exc}")
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
        log("cosine: no maneuvers")
        return

    log(f"cosine: fetching {len(ids)} embeddings from Chroma")
    result = collection.get(ids=ids, include=["embeddings"])
    id_to_emb = {
        i: np.asarray(e, dtype=np.float64)
        for i, e in zip(result["ids"], result["embeddings"])
    }
    present = [m for m in maneuvers if m in id_to_emb]
    missing = sorted(set(maneuvers) - set(present))
    if missing:
        log(f"cosine: missing embeddings for {len(missing)} maneuvers")

    groups = build_groups(present)
    similarity_matrices: dict[int, tuple[list[str], np.ndarray]] = {}

    for idx, group in enumerate(groups, start=1):
        if len(group) < 2:
            log(f"cosine group {idx}: skip (n={len(group)})")
            continue
        log(f"cosine group {idx}: n={len(group)} computing matrix")
        vectors = np.stack([id_to_emb[m] for m in group], axis=0)
        sim = cosine_similarity(vectors)
        similarity_matrices[idx] = (group, sim)

    matrices_dir.mkdir(parents=True, exist_ok=True)
    if plot:
        log("cosine: writing heatmaps / npy / csv")
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
        log(
            f"  threshold {threshold}%: groups_with_pairs={len(redundant)}, "
            f"removed={n_removed}"
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
            log(f"mat filter {thr}: no removal JSON, skip")
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
        log(
            f"mat filter {thr}: kept={len(kept_names)} copied={copied} → {out_dir}"
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
    p.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help="Ollama num_ctx (default 16384; model ships with 262144 which is tiny-slow).",
    )
    p.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help="Max generated tokens per call (default 4096).",
    )
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
            log(f"WARNING: unknown maneuvers skipped: {sorted(unknown)}")
    else:
        maneuvers = all_maneuvers
    if args.limit is not None:
        maneuvers = maneuvers[: max(0, args.limit)]

    log(f"Manoeuvers: {len(maneuvers)} (from {plots_root})")
    log(f"Stages: {args.stages}")

    stages = set(args.stages)

    if "text" in stages:
        log("=== stage: text ===")
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
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )

    if "embed" in stages:
        log("=== stage: embed ===")
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
        log("=== stage: cosine ===")
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
        log("=== stage: mat ===")
        stage_mat_filter(
            root=root,
            source_dir=mat_source,
            matrices_dir=matrices_dir,
            out_root=mat_out_root,
            mode=args.mat_link_mode,
        )

    log("Done.")


if __name__ == "__main__":
    main()
