#!/usr/bin/env bash
# Ensure the system Ollama service is up and the embedding model exists.
# Does NOT invent a second terminal — uses the normal systemd ollama.service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MODEL="${OLLAMA_EMBED_MODEL:-qwen3-embedding:latest}"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
BASE_URL="http://${HOST}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found in PATH. Install from https://ollama.com" >&2
  exit 1
fi

if ! curl -sf --connect-timeout 2 "${BASE_URL}/api/tags" >/dev/null; then
  echo "Ollama daemon is down (systemctl: ollama.service inactive)." >&2
  echo "This is the normal machine service — start it once:" >&2
  echo "  sudo systemctl start ollama" >&2
  echo "Then re-run. No screen needed; afterwards 'ollama list' works as before." >&2
  # Try non-interactive start in case passwordless sudo / polkit allows it.
  if systemctl start ollama 2>/dev/null || sudo -n systemctl start ollama 2>/dev/null; then
    sleep 1
  fi
  if ! curl -sf --connect-timeout 2 "${BASE_URL}/api/tags" >/dev/null; then
    exit 1
  fi
fi

if ! ollama show "${MODEL}" >/dev/null 2>&1; then
  echo "Pulling embedding model: ${MODEL}"
  ollama pull "${MODEL}"
fi

echo "Ollama ready (${BASE_URL}, model=${MODEL})"
