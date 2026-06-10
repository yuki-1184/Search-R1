#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=01:00:00
#PJM -L jobenv=singularity
#PJM -g pj24003196
#PJM -j
#PJM -o logs/prepare_nq_search_%j.log

set -euo pipefail

module load singularity-ce

PROJ=${PROJ:-/home/pj24003196/ku60000140/Search-R1}
SIF=${SIF:-$PROJ/search_r1.sif}
DATA_DIR=${DATA_DIR:-data/nq_search}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

mkdir -p "$PROJ/logs" "$HF_HOME"
test -f "$SIF" || { echo "SIF not found: $SIF" >&2; exit 1; }

singularity exec \
    --bind "$PROJ:/workspace" \
    --bind "$HF_HOME:$HF_HOME" \
    --env HF_HOME="$HF_HOME" \
    --env PYTHONPATH=/workspace \
    "$SIF" \
    bash -c "cd /workspace && python scripts/data_process/nq_search.py --local_dir '$DATA_DIR'"

test -s "$PROJ/$DATA_DIR/train.parquet"
test -s "$PROJ/$DATA_DIR/test.parquet"
echo "Prepared NQ Search-R1 data under $PROJ/$DATA_DIR"
