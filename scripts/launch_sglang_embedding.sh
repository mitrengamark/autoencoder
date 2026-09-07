#!/usr/bin/env bash
# Launch SGLang embedding server (OpenAI-compatible /v1/embeddings).
# Requires: pip install sglang (in the active Python environment)
#
# Memory protection:
# - raises oom_score_adj so this job is preferred for OOM kill over SSH
# - re-execs under a systemd user scope with MemoryMax (default 100G)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oom_protect.sh"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
  echo "Using Hugging Face token from ${ENV_FILE}"
fi

export OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-800}"
export MEMORY_MAX="${SGLANG_MEMORY_MAX:-100G}"
export MEMORY_HIGH="${SGLANG_MEMORY_HIGH:-90G}"

oom_set_victim_score "${OOM_SCORE_ADJ}"
oom_reexec_under_memory_scope SGLANG_UNDER_MEMORY_SCOPE "$@"
oom_set_victim_score "${OOM_SCORE_ADJ}"

MODEL="${SGLANG_EMBED_MODEL:-Alibaba-NLP/gte-Qwen2-7B-instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30001}"

echo "Starting SGLang embedding server: ${MODEL} on ${HOST}:${PORT}"
echo "Protection: oom_score_adj=$(cat /proc/self/oom_score_adj 2>/dev/null || echo '?') MemoryMax=${MEMORY_MAX}"
exec python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --is-embedding \
  --host "${HOST}" \
  --port "${PORT}"
