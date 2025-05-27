#!/bin/bash

CONFIG_DIR="Config/configs"
CONFIGS=(
    "bmw_OG_remake.ini"
    "bmw_allando_chirp.ini"
    "bmw_allando_savvaltas.ini"
    "bmw_allando_sin.ini"
    "bmw_valtozo_savvaltas_fek.ini"
    "bmw_valtozo_savvaltas_gas.ini"
    "bmw_valtozo_sin_fek.ini"
    "bmw_valtozo_sin_gas.ini"
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
