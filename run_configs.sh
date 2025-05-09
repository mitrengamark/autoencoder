#!/bin/bash

CONFIG_DIR="Config/configs"
CONFIGS=(
    "allando_savvaltas_diff.ini"
    "allando_sin_diff.ini"
    "valtozo_savvaltas_fek_diff.ini"
    "valtozo_savvaltas_gas_diff.ini"
    "valtozo_sin_fek_diff.ini"
    "valtozo_sin_gas_diff.ini"
)
RUN_SCRIPT="python run.py"

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/$CONFIG"
    CONFIG_NAME=$(basename "$CONFIG" .ini)
    LOGNAME="log_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

    export CONFIG_PATH
    echo "Indítás: $CONFIG_PATH -> log: $LOGNAME"
    $RUN_SCRIPT > "logs/$LOGNAME" 2>&1

    echo "Befejeződött: $CONFIG_PATH"
    echo ""
done

echo "Minden konfiguráció lefutott!"
