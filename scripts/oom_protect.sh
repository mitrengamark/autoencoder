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

# Re-exec the caller script under a systemd user scope with a hard memory cap.
# Usage from a script:
#   export MEMORY_MAX=100G MEMORY_HIGH=90G OOM_SCORE_ADJ=800
#   oom_reexec_under_memory_scope SGLANG_UNDER_MEMORY_SCOPE "$@"
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
  local score="${OOM_SCORE_ADJ:-800}"

  echo "Re-exec under systemd user scope: MemoryMax=${mem_max} MemoryHigh=${mem_high} OOMPolicy=kill"
  exec systemd-run --user --same-dir --collect --scope \
    --expand-environment=yes \
    -E "${flag_name}=1" \
    -E "OOM_SCORE_ADJ=${score}" \
    -E "MEMORY_MAX=${mem_max}" \
    -E "MEMORY_HIGH=${mem_high}" \
    -E "HF_TOKEN=${HF_TOKEN:-}" \
    -E "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    -E "SGLANG_EMBED_MODEL=${SGLANG_EMBED_MODEL:-}" \
    -E "SGLANG_HOST=${SGLANG_HOST:-}" \
    -E "SGLANG_PORT=${SGLANG_PORT:-}" \
    -E "SGLANG_MEMORY_MAX=${SGLANG_MEMORY_MAX:-}" \
    -E "SGLANG_MEMORY_HIGH=${SGLANG_MEMORY_HIGH:-}" \
    -E "EMBED_MEMORY_MAX=${EMBED_MEMORY_MAX:-}" \
    -E "EMBED_MEMORY_HIGH=${EMBED_MEMORY_HIGH:-}" \
    -p "MemoryMax=${mem_max}" \
    -p "MemoryHigh=${mem_high}" \
    -p OOMPolicy=kill \
    -- "$0" "$@"
}
