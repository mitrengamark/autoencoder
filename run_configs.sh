#!/bin/bash

CONFIG_DIR="Config/configs"
CONFIGS=(
    "bmw_OG_remake_98.ini"
    "bmw_OG_remake_95.ini"
    "bmw_OG_remake_90.ini"
    
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
