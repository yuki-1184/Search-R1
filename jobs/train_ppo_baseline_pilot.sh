#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=04:00:00
#PJM -L jobenv=singularity
#PJM -g pj24003196
#PJM -j
#PJM -o logs/train_ppo_baseline_pilot_%j.log

set -euo pipefail

module load singularity-ce

PROJ=${PROJ:-/home/pj24003196/ku60000140/Search-R1}
DATA=${DATA:-$HOME/searchr1_data}
SIF=${SIF:-$PROJ/search_r1.sif}
DATA_DIR=${DATA_DIR:-data/nq_search}
PORT=${RETRIEVAL_PORT:-8000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
RAY_TMPDIR=${RAY_SHORT_TMPDIR:-/tmp/ray_${PJM_JOBID:-baseline}}
USE_FAISS_GPU=${USE_FAISS_GPU:-0}
SEED=${SEED:-13}

BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-nq-search-r1-ppo-llama3.2-3b-pilot-seed${SEED}}
# The upstream loop starts at step 1 and exits after incrementing, so 21 yields
# 20 optimizer updates and allows the step-20 checkpoint to be written.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-21}
TRAIN_DATA_NUM=${TRAIN_DATA_NUM:-1024}
VAL_DATA_NUM=${VAL_DATA_NUM:-256}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
MAX_TURNS=${MAX_TURNS:-2}
RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}

RUN_DIR=${RUN_DIR:-$PROJ/experiments/$EXPERIMENT_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_DIR/checkpoints}
RUN_LOG=${RUN_LOG:-$RUN_DIR/train.log}
MANIFEST=${MANIFEST:-$RUN_DIR/manifest.txt}
VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:-$RUN_DIR/validation}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

mkdir -p "$PROJ/logs" "$RUN_DIR" "$OUTPUT_DIR" "$VALIDATION_OUTPUT_DIR" "$HF_HOME"

test -f "$SIF" || { echo "SIF not found: $SIF" >&2; exit 1; }
test -s "$PROJ/$DATA_DIR/train.parquet" || {
    echo "Missing $PROJ/$DATA_DIR/train.parquet" >&2
    echo "Run: pjsub jobs/prepare_nq_search.sh" >&2
    exit 1
}
test -s "$PROJ/$DATA_DIR/test.parquet" || {
    echo "Missing $PROJ/$DATA_DIR/test.parquet" >&2
    echo "Run: pjsub jobs/prepare_nq_search.sh" >&2
    exit 1
}
test -s "$DATA/e5_Flat.index" || { echo "Missing $DATA/e5_Flat.index" >&2; exit 1; }
test -s "$DATA/wiki-18.jsonl" || { echo "Missing $DATA/wiki-18.jsonl" >&2; exit 1; }

{
    echo "experiment_name=$EXPERIMENT_NAME"
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "job_id=${PJM_JOBID:-unknown}"
    echo "git_commit=$(git -C "$PROJ" rev-parse HEAD)"
    echo "git_dirty=$(test -n "$(git -C "$PROJ" status --porcelain)" && echo true || echo false)"
    echo "base_model=$BASE_MODEL"
    echo "data_dir=$DATA_DIR"
    echo "retriever_model=intfloat/e5-base-v2"
    echo "retriever_topk=$RETRIEVER_TOPK"
    echo "max_turns=$MAX_TURNS"
    echo "seed=$SEED"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "total_training_steps=$TOTAL_TRAINING_STEPS"
    echo "train_data_num=$TRAIN_DATA_NUM"
    echo "val_data_num=$VAL_DATA_NUM"
    echo "train_batch_size=$TRAIN_BATCH_SIZE"
    echo "val_batch_size=$VAL_BATCH_SIZE"
    stat -c "sif=%n size=%s mtime=%y" "$SIF"
    stat -c "index=%n size=%s mtime=%y" "$DATA/e5_Flat.index"
    stat -c "corpus=%n size=%s mtime=%y" "$DATA/wiki-18.jsonl"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "$MANIFEST"

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

echo "Starting fixed retrieval server on port ${PORT}"
RETRIEVAL_PORT=$PORT singularity exec --nv \
    --bind "$PROJ:/workspace" \
    --bind "$DATA:$DATA" \
    "$SIF" \
    python /workspace/search_r1/search/retrieval_server.py \
        --index_path "$DATA/e5_Flat.index" \
        --corpus_path "$DATA/wiki-18.jsonl" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model intfloat/e5-base-v2 \
        "${FAISS_GPU_FLAG[@]}" &

RETRIEVER_PID=$!

echo "Waiting for retrieval server health check"
for i in $(seq 1 120); do
    if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        echo "Retrieval server exited before becoming ready" >&2
        wait "$RETRIEVER_PID"
        exit 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "Retrieval server did not become ready in time" >&2
        exit 1
    fi
    sleep 5
done

echo "Launching Search-R1 PPO baseline pilot"
singularity exec --nv \
    --bind "$PROJ:/workspace" \
    --bind "$DATA:$DATA" \
    --bind "$HF_HOME:$HF_HOME" \
    --env RAY_TMPDIR="$RAY_TMPDIR" \
    --env TMPDIR="$RAY_TMPDIR" \
    --env HF_HOME="$HF_HOME" \
    "$SIF" \
    bash -c "cd /workspace && \
        export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
        RETRIEVAL_PORT='$PORT' \
        BASE_MODEL='$BASE_MODEL' \
        EXPERIMENT_NAME='$EXPERIMENT_NAME' \
        DATA_DIR='$DATA_DIR' \
        TOTAL_TRAINING_STEPS='$TOTAL_TRAINING_STEPS' \
        TRAIN_DATA_NUM='$TRAIN_DATA_NUM' \
        VAL_DATA_NUM='$VAL_DATA_NUM' \
        TRAIN_BATCH_SIZE='$TRAIN_BATCH_SIZE' \
        VAL_BATCH_SIZE='$VAL_BATCH_SIZE' \
        SAVE_FREQ='$SAVE_FREQ' \
        TEST_FREQ='$TEST_FREQ' \
        MAX_TURNS='$MAX_TURNS' \
        RETRIEVER_TOPK='$RETRIEVER_TOPK' \
        SEED='$SEED' \
        OUTPUT_DIR='/workspace/experiments/$EXPERIMENT_NAME/checkpoints' \
        VALIDATION_OUTPUT_DIR='/workspace/experiments/$EXPERIMENT_NAME/validation' \
        RUN_LOG='/workspace/experiments/$EXPERIMENT_NAME/train.log' && \
        bash train_ppo.sh"
