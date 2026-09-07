#!/usr/bin/env bash
# Run embed_texts_to_chroma.py with OOM victim score + optional memory cap.
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

export OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-700}"
# Embed client is lighter than SGLang; still cap to leave headroom.
export MEMORY_MAX="${EMBED_MEMORY_MAX:-20G}"
export MEMORY_HIGH="${EMBED_MEMORY_HIGH:-16G}"

oom_set_victim_score "${OOM_SCORE_ADJ}"
oom_reexec_under_memory_scope EMBED_UNDER_MEMORY_SCOPE "$@"
oom_set_victim_score "${OOM_SCORE_ADJ}"

echo "Protection: oom_score_adj=$(cat /proc/self/oom_score_adj 2>/dev/null || echo '?') MemoryMax=${MEMORY_MAX}"
exec python embed_texts_to_chroma.py "$@"
