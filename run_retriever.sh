#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ=${PROJ:-$SCRIPT_DIR}
DATA=${DATA:-$HOME/searchr1_data}
SIF=${SIF:-$PROJ/search_r1.sif}
PORT=${RETRIEVAL_PORT:-8000}
USE_FAISS_GPU=${USE_FAISS_GPU:-0}

FAISS_GPU_FLAG=""
if [ "$USE_FAISS_GPU" = "1" ]; then
    FAISS_GPU_FLAG="--faiss_gpu"
fi

singularity exec --nv \
    --bind $PROJ:/workspace \
    --bind $DATA:$DATA \
    --env RETRIEVAL_PORT=$PORT \
    $SIF \
    python /workspace/search_r1/search/retrieval_server.py \
        --index_path $DATA/e5_Flat.index \
        --corpus_path $DATA/wiki-18.jsonl \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        $FAISS_GPU_FLAG
