#!/bin/bash

CONFIG_DIR="Config/configs"
CONFIGS=(
    "bmw_allando_chirp_98.ini"
    "bmw_allando_chirp_95.ini"
    "bmw_allando_chirp_90.ini"
    "bmw_allando_savvaltas_98.ini"
    "bmw_allando_savvaltas_95.ini"
    "bmw_allando_savvaltas_90.ini"
    "bmw_allando_sin_98.ini"
    "bmw_allando_sin_95.ini"
    "bmw_allando_sin_90.ini"
    "bmw_valtozo_savvaltas_fek_98.ini"
    "bmw_valtozo_savvaltas_fek_95.ini"
    "bmw_valtozo_savvaltas_fek_90.ini"
    "bmw_valtozo_savvaltas_gas_98.ini"
    "bmw_valtozo_savvaltas_gas_95.ini"
    "bmw_valtozo_savvaltas_gas_90.ini"
    "bmw_valtozo_sin_fek_98.ini"
    "bmw_valtozo_sin_fek_95.ini"
    "bmw_valtozo_sin_fek_90.ini"
    "bmw_valtozo_sin_gas_98.ini"
    "bmw_valtozo_sin_gas_95.ini"
    "bmw_valtozo_sin_gas_90.ini"
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
