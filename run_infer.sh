#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=${PROJ:-$SCRIPT_DIR}
DATA=${DATA:-$HOME/searchr1_data}
SIF=${SIF:-$PROJ/search_r1.sif}

singularity exec --nv \
    --bind $PROJ:/workspace \
    --bind $DATA:$DATA \
    $SIF \
    python /workspace/infer.py
