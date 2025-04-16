#!/bin/bash

CONFIG_DIR="Config/configs"
CONFIGS=("ae3.ini" "ae4.ini" "ae5.ini" "ae6.ini")
RUN_SCRIPT="python run.py"

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/$CONFIG"
    CONFIG_NAME=$(basename "$CONFIG" .ini)
    LOGNAME="log_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"

    export CONFIG_NAME=$CONFIG_NAME

    echo "Indítás: $CONFIG_PATH -> log: $LOGNAME"
    $RUN_SCRIPT --config "$CONFIG_PATH" > "logs/$LOGNAME" 2>&1

    echo "Befejeződött: $CONFIG_PATH"
    echo ""
done

echo "Minden konfiguráció lefutott!"
