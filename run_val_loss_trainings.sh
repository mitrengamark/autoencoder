#!/bin/bash
set -euo pipefail

CONFIG_DIR="Config/configs"
RUN_SCRIPT="python3 run.py"

CONFIGS=(
    # data3 — 7 maneuver groups
    "allando_chirp.ini"
    "allando_savvaltas.ini"
    "allando_sin.ini"
    "valtozo_savvaltas_fek.ini"
    "valtozo_savvaltas_gas.ini"
    "valtozo_sin_fek.ini"
    "valtozo_sin_gas.ini"
    # data3 — all data + thresholds
    "ae_OG_remake.ini"
    "OG_remake_90.ini"
    "OG_remake_95.ini"
    "OG_remake_98.ini"
    # data_bmw_combined — 7 maneuver groups
    "bmw_allando_chirp.ini"
    "bmw_allando_savvaltas.ini"
    "bmw_allando_sin.ini"
    "bmw_valtozo_savvaltas_fek.ini"
    "bmw_valtozo_savvaltas_gas.ini"
    "bmw_valtozo_sin_fek.ini"
    "bmw_valtozo_sin_gas.ini"
    # data_bmw_combined — all data + thresholds
    "bmw_OG_remake.ini"
    "bmw_OG_remake_90.ini"
    "bmw_OG_remake_95.ini"
    "bmw_OG_remake_98.ini"
)

mkdir -p logs Results/val_losses

echo "Starting ${#CONFIGS[@]} training runs with validation-loss CSV export."
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/$CONFIG"
    CONFIG_NAME=$(basename "$CONFIG" .ini)
    LOGNAME="log_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

    export CONFIG_PATH
    echo "Indítás: $CONFIG_PATH -> log: logs/$LOGNAME"
    $RUN_SCRIPT > "logs/$LOGNAME" 2>&1

    CSV="Results/val_losses/${CONFIG_NAME}.csv"
    if [[ -f "$CSV" ]]; then
        echo "  CSV OK: $CSV ($(wc -l < "$CSV") lines)"
    else
        echo "  HIBA: hiányzó CSV: $CSV" >&2
        exit 1
    fi
    echo "Befejeződött: $CONFIG_PATH"
    echo ""
done

echo "Mind a ${#CONFIGS[@]} tanítás lefutott."
echo "Validation loss CSV-k: Results/val_losses/"
