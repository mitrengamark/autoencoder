#!/bin/bash

CONFIG_DIR="Config/configs"
CONFIGS=(
    "vae1.ini"
    "vae2.ini"
    "vae3.ini"
    "vae4.ini"
    "vae5.ini"
    "vae6.ini"
    "vae7.ini"
    "vae8.ini"
    "vae9.ini"
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
