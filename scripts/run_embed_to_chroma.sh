#!/usr/bin/env bash
# One-shot: ensure Ollama is up, then embed texts into ChromaDB.
# Protection: system-wide MemAvailable watchdog inside the Python process
# (default: stop if < 5 GiB free on the whole machine).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/oom_protect.sh"

cd "${ROOT_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -f "${ROOT_DIR}/.venv-embeddings/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv-embeddings/bin/activate"
fi

"${SCRIPT_DIR}/ensure_ollama_embed.sh"

# Prefer being OOM-killed before SSH if the kernel runs out of memory.
export OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-700}"
oom_set_victim_score "${OOM_SCORE_ADJ}"

OLLAMA_HOST_VAL="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_EMBED_MODEL:-qwen3-embedding:latest}"
MIN_AVAIL_GB="${EMBED_MIN_AVAIL_MEM_GB:-5}"

echo "Protection: system MemAvailable watchdog < ${MIN_AVAIL_GB} GiB; oom_score_adj=$(cat /proc/self/oom_score_adj 2>/dev/null || echo '?')"
echo "Ollama embed: model=${OLLAMA_MODEL} base-url=http://${OLLAMA_HOST_VAL}/v1"

extra=()
joined=" $* "
if [[ "${joined}" != *" --base-url "* && "${joined}" != *" --base-url="* ]]; then
  extra+=(--base-url "http://${OLLAMA_HOST_VAL}/v1")
fi
if [[ "${joined}" != *" --model "* && "${joined}" != *" --model="* ]]; then
  extra+=(--model "${OLLAMA_MODEL}")
fi
if [[ "${joined}" != *" --min-avail-mem-gb "* && "${joined}" != *" --min-avail-mem-gb="* ]]; then
  extra+=(--min-avail-mem-gb "${MIN_AVAIL_GB}")
fi

exec python embed_texts_to_chroma.py "${extra[@]}" "$@"
