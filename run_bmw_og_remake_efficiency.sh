#!/bin/bash
set -euo pipefail

# Same runner pattern as run_og_remake_efficiency.sh, but only the 4 BMW OG remake configs.
CONFIG_DIR="Config/configs"
CONFIGS=(
    "bmw_OG_remake.ini"
    "bmw_OG_remake_90.ini"
    "bmw_OG_remake_95.ini"
    "bmw_OG_remake_98.ini"
)
RUN_SCRIPT="python3 run.py"

mkdir -p logs Results/efficiency

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/$CONFIG"
    CONFIG_NAME=$(basename "$CONFIG" .ini)
    LOGNAME="log_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

    export CONFIG_PATH
    echo "Indítás: $CONFIG_PATH -> log: logs/$LOGNAME"
    $RUN_SCRIPT > "logs/$LOGNAME" 2>&1

    echo "Befejeződött: $CONFIG_PATH"
    echo ""
done

echo "Minden konfiguráció lefutott!"
echo "Efficiency összefoglaló: Results/efficiency_summary.csv"
