#!/bin/bash
# rscgrp: 使えるグループ名は `pjstat --rsc` で確認
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -L jobenv=singularity
#PJM -g pj24003196
#PJM -j
#PJM -o logs/infer_%j.log

set -eu

module load singularity-ce

PROJ=${PROJ:-/home/pj24003196/ku60000140/Search-R1}
DATA=${DATA:-$HOME/searchr1_data}
SIF=${SIF:-$PROJ/search_r1.sif}
PORT=8000
USE_FAISS_GPU=${USE_FAISS_GPU:-0}

mkdir -p $PROJ/logs
test -f "$SIF" || { echo "SIF not found: $SIF" >&2; exit 1; }

cleanup() {
    if [ -n "${RETRIEVER_PID:-}" ] && kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        kill "$RETRIEVER_PID"
    fi
}
trap cleanup EXIT

FAISS_GPU_FLAG=()
if [ "$USE_FAISS_GPU" = "1" ]; then
    FAISS_GPU_FLAG=(--faiss_gpu)
fi

# retrieval server をバックグラウンドで起動
RETRIEVAL_PORT=$PORT singularity exec --nv \
    --bind $PROJ:/workspace \
    --bind $DATA:$DATA \
    $SIF \
    python /workspace/search_r1/search/retrieval_server.py \
        --index_path $DATA/e5_Flat.index \
        --corpus_path $DATA/wiki-18.jsonl \
        --topk 3 \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        "${FAISS_GPU_FLAG[@]}" &

RETRIEVER_PID=$!

# サーバーが起動するまで待機
echo "Waiting for retrieval server to start..."
for i in $(seq 1 60); do
    if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        echo "Retrieval server exited before becoming ready" >&2
        wait "$RETRIEVER_PID"
        exit 1
    fi
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Retrieval server is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Retrieval server did not become ready in time" >&2
        exit 1
    fi
    sleep 5
done

# infer.py を実行
RETRIEVAL_PORT=$PORT singularity exec --nv \
    --bind $PROJ:/workspace \
    --bind $DATA:$DATA \
    $SIF \
    python /workspace/infer.py
