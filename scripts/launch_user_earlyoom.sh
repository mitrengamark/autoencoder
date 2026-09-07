#!/usr/bin/env bash
# User-space earlyoom: kills *our* heavy processes when free RAM/swap is low,
# while avoiding sshd / interactive shells / screen / tmux.
# Does not need root for killing our own jobs (system earlyoom is currently failed).
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

if ! command -v earlyoom >/dev/null 2>&1; then
  echo "earlyoom not installed" >&2
  exit 1
fi

MEM_MIN="${EARLYOOM_MEM_MIN:-8}"
MEM_KILL="${EARLYOOM_MEM_KILL:-4}"
SWAP_MIN="${EARLYOOM_SWAP_MIN:-10}"
SWAP_KILL="${EARLYOOM_SWAP_KILL:-5}"

echo "Starting user earlyoom: kill when avail mem < ${MEM_KILL}% (notify < ${MEM_MIN}%), prefer python/sglang, avoid sshd/screen/bash"
exec earlyoom \
  -m "${MEM_MIN},${MEM_KILL}" \
  -s "${SWAP_MIN},${SWAP_KILL}" \
  -r 30 \
  --avoid '^(sshd|systemd|bash|tmux|screen|zsh|fish|cursor-server|node)$' \
  --prefer '^(python3|python|sglang)$' \
  "$@"
