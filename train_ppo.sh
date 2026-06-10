#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export DATA_DIR=${DATA_DIR:-data/nq_search}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')}
export RETRIEVAL_PORT=${RETRIEVAL_PORT:-8000}
export RAY_TMPDIR=${RAY_SHORT_TMPDIR:-/tmp/ray_${PJM_JOBID:-$$}}
export TMPDIR=$RAY_TMPDIR
mkdir -p "$RAY_TMPDIR"

WAND_PROJECT=${WAND_PROJECT:-Search-R1}

export BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nq-search-r1-ppo-llama3.2-3b-em}
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-it-em

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

TRAIN_DATA_NUM=${TRAIN_DATA_NUM:-null}
VAL_DATA_NUM=${VAL_DATA_NUM:-null}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1005}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-50}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
VAL_ONLY=${VAL_ONLY:-false}
MAX_TURNS=${MAX_TURNS:-2}
RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}
SEED=${SEED:-13}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console']"}
OUTPUT_DIR=${OUTPUT_DIR:-verl_checkpoints/$EXPERIMENT_NAME}
RUN_LOG=${RUN_LOG:-$EXPERIMENT_NAME.log}
VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:-validation_outputs/$EXPERIMENT_NAME}

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=$TRAIN_DATA_NUM \
    data.val_data_num=$VAL_DATA_NUM \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger="$TRAINER_LOGGER" \
    +trainer.seed=$SEED \
    +trainer.validation_output_dir=$VALIDATION_OUTPUT_DIR \
    +trainer.val_only=$VAL_ONLY \
    +trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$OUTPUT_DIR \
    max_turns=$MAX_TURNS \
    retriever.url="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" \
    retriever.topk=$RETRIEVER_TOPK \
    2>&1 | tee "$RUN_LOG"
