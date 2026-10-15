# Qwen3-VL-Embedding: 14 plot → 1 vector

Joint multimodal embedding of each Tesla maneuver’s 14 structured panels with
`Qwen/Qwen3-VL-Embedding-2B` (2048-d, L2-normalized).

## Layout

| Path | Role |
|---|---|
| `pyproject.toml` / `uv.lock` / `.venv/` | Single project env (includes VL-embedding deps) |
| `Qwen3-VL-Embedding/` | Upstream code + weights only (gitignored; no own venv/lock) |
| `Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B/` | Local HF weights (~4 GB) |
| `embed_maneuver_plots_qwen3vl.py` | Batch embedder for `Results/maneuver_structured_plots_tesla` |
| `Results/qwen3_vl_maneuver_embeddings/` | `<maneuver>.npy` + `.json` + `summary.json` |

## Setup

```bash
# project root
uv sync
source .venv/bin/activate

# weights only (code tree already under Qwen3-VL-Embedding/)
hf download Qwen/Qwen3-VL-Embedding-2B \
  --local-dir ./Qwen3-VL-Embedding/models/Qwen3-VL-Embedding-2B
```

`decord` is intentionally omitted (no aarch64 wheel); image-only embedding does not need it.
`flash-attn` is optional — SDPA works on GB10.

## Smoke test (14 images → one vector)

```bash
source .venv/bin/activate
python embed_maneuver_plots_qwen3vl.py --smoke
```

## Batch all maneuvers

```bash
source .venv/bin/activate
python embed_maneuver_plots_qwen3vl.py --all
```

## API shape

```python
embeddings = model.process([{"image": [path1, ..., path14]}])
vector = embeddings[0]  # (2048,)
```

Do **not** use Ollama for this model: multimodal embedding is not supported there.
