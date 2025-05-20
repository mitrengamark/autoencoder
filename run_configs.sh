#!/bin/bash

CONFIG_DIR="Config/configs"
CONFIGS=(
    "vae_1.ini"
    "vae_2.ini"
    "vae_3.ini"
    "vae_4.ini"
    "vae_5.ini"
    "vae_6.ini"
    "vae_7.ini"
    "vae_8.ini"
    "vae_9.ini"
    "vae_10.ini"
    "vae_11.ini"
    "vae_12.ini"
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
