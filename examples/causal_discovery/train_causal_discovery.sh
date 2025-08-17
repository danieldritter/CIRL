#!/bin/bash
set -e

MODEL_PATH=""Qwen/Qwen2-1.5B-Instruct""
NLI_MODEL_NAME="cross-encoder/nli-deberta-v3-base"
DATA_DIR="./datasets"
TRAIN_FILES="['$DATA_DIR/apple_gastronome_synthetic/train.parquet']"
# TEST_FILES="['$DATA_DIR/apple_gastronome_synthetic/val.parquet','$DATA_DIR/neuropathic_pain_causal/train.parquet']"
TEST_FILES="['$DATA_DIR/neuropathic_pain_causal/train.parquet']"
MAX_FACTORS=10
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=512
BATCH_SIZE=16
MINIBATCH_SIZE=8
BATCH_SIZE_PER_GPU=4
N_GPUS=2
EXPERIMENT_NAME="test"

python verl/trainer/main_causal_discovery.py \
    data.train_files=$TRAIN_FILES \
    data.val_files=$TEST_FILES \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.train_batch_size=$BATCH_SIZE \
    data.max_factors="$MAX_FACTORS" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=3 \
    trainer.n_gpus_per_node=$N_GPUS \
    causal_discovery_reward.causal_method=PC-bidirectional \
    nli.model_name=$NLI_MODEL_NAME \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINIBATCH_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$BATCH_SIZE_PER_GPU