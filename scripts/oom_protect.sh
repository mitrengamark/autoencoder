#!/usr/bin/env bash
# Shared OOM helpers for heavy jobs (prefer killing these over SSH/shell).
# shellcheck shell=bash

oom_set_victim_score() {
  local score="${1:-${OOM_SCORE_ADJ:-800}}"
  if [[ -w /proc/self/oom_score_adj ]]; then
    echo "${score}" > /proc/self/oom_score_adj 2>/dev/null || return 0
    echo "oom_score_adj=$(cat /proc/self/oom_score_adj) (higher = killed first under memory pressure)"
  fi
}

# Re-exec the caller script under a systemd user scope with RAM + swap caps.
# MemoryMax must be large enough to use free system RAM (Chroma HNSW grows).
# MemorySwapMax stops the job from filling the whole machine swap while RAM
# outside the old tiny cap sat unused.
#
# Usage:
#   export MEMORY_MAX=100G MEMORY_HIGH=90G MEMORY_SWAP_MAX=8G OOM_SCORE_ADJ=700
#   oom_reexec_under_memory_scope EMBED_UNDER_MEMORY_SCOPE "$@"
oom_reexec_under_memory_scope() {
  local flag_name="$1"
  shift
  local flag_val="${!flag_name:-}"

  if [[ "${flag_val}" == "1" ]]; then
    return 0
  fi
  if [[ "${SKIP_MEMORY_SCOPE:-0}" == "1" ]]; then
    echo "SKIP_MEMORY_SCOPE=1 — not wrapping in systemd memory scope"
    return 0
  fi
  if ! command -v systemd-run >/dev/null 2>&1; then
    echo "systemd-run not found — relying on oom_score_adj only"
    return 0
  fi

  local mem_max="${MEMORY_MAX:-100G}"
  local mem_high="${MEMORY_HIGH:-90G}"
  local swap_max="${MEMORY_SWAP_MAX:-8G}"
  local score="${OOM_SCORE_ADJ:-800}"

  echo "Re-exec under systemd user scope: MemoryMax=${mem_max} MemoryHigh=${mem_high} MemorySwapMax=${swap_max} OOMPolicy=kill"
  exec systemd-run --user --same-dir --collect --scope \
    --expand-environment=yes \
    -E "${flag_name}=1" \
    -E "OOM_SCORE_ADJ=${score}" \
    -E "MEMORY_MAX=${mem_max}" \
    -E "MEMORY_HIGH=${mem_high}" \
    -E "MEMORY_SWAP_MAX=${swap_max}" \
    -E "OLLAMA_HOST=${OLLAMA_HOST:-}" \
    -E "OLLAMA_EMBED_MODEL=${OLLAMA_EMBED_MODEL:-}" \
    -E "EMBED_MEMORY_MAX=${EMBED_MEMORY_MAX:-}" \
    -E "EMBED_MEMORY_HIGH=${EMBED_MEMORY_HIGH:-}" \
    -E "EMBED_MEMORY_SWAP_MAX=${EMBED_MEMORY_SWAP_MAX:-}" \
    -p "MemoryMax=${mem_max}" \
    -p "MemoryHigh=${mem_high}" \
    -p "MemorySwapMax=${swap_max}" \
    -p OOMPolicy=kill \
    -- "$0" "$@"
}
