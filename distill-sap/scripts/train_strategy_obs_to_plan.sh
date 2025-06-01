#!/bin/bash
set -euo pipefail

# Trap interrupts (e.g. Ctrl+C) to kill all subprocesses
trap "echo 'Interrupted. Killing subprocesses...'; pkill -P $$; exit 1" SIGINT SIGTERM

# -------------------------------------
# Qwen3-4B Training Script (DeepSpeed Zero3 + Swift)
# -------------------------------------
# Usage:
#   bash distill-sap/scripts/train_strategy_obs_to_plan.sh > logs/train_strategy_obs_to_plan.log 2>&1 &

#######################
# CONFIGURATION
#######################
MODEL_PATH="/mnt/public/models/Qwen3-4B"
DATA_VERSION=1  # data size 8753
DATASET_PATH="/mnt/public/xushuai/sap/distill-sap/data/strategy_obs_to_plan_v${DATA_VERSION}.jsonl"
BASE_OUTPUT_DIR="/mnt/public/xushuai/ft_models/SAP-4B-v${DATA_VERSION}"
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=200
NUM_EPOCHS=4

#######################
# GENERATE UNIQUE OUTPUT DIR
#######################
TRAINING_ARGS_VERSION=1
OUTPUT_DIR="$BASE_OUTPUT_DIR"
while [ -d "$OUTPUT_DIR" ]; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_${TRAINING_ARGS_VERSION}"
    ((TRAINING_ARGS_VERSION++))
done

echo "[INFO] Using output directory: $OUTPUT_DIR"

#######################
# TRAINING
#######################
echo "[INFO] Starting training..."

CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
OMP_NUM_THREADS=16 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_length 3400 \
    --warmup_ratio 0.1 \
    --learning_rate 1e-5 \
    --eval_strategy no \
    --deepspeed zero3 \
    --save_only_model true \
    --gradient_checkpointing \
    --ddp_backend nccl \
    --save_strategy epoch \
    --save_total_limit 1 \
    --train_type full \
    --torch_dtype bfloat16 \
    --add_version false \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --logging_steps 1 \
    --report_to swanlab \
    --attn_impl flash_attn \
    --use_liger_kernel true

#######################
# EVALUATION
#######################
echo "[INFO] Starting evaluation..."

CUDA_VISIBLE_DEVICES=3,4 \
python src/eval.py --model "$OUTPUT_DIR" --qwen3

echo "[INFO] Done. Training and evaluation complete."

# End of script