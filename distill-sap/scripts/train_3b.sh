#!/bin/bash
set -euo pipefail

# Trap interrupts (e.g. Ctrl+C) to kill all subprocesses
trap "echo 'Interrupted. Killing subprocesses...'; pkill -P $$; exit 1" SIGINT SIGTERM
unset http_proxy https_proxy
# -------------------------------------
# Qwen3-4B Training Script (DeepSpeed Zero3 + Swift)
# -------------------------------------
# Usage:
#   nohup bash distill-sap/scripts/train_4b.sh > train.log 2>&1 &

# =====================
#    CONFIGURATION
# =====================
MODEL_PATH="/mnt/opsstorage/models/Qwen2.5-3B-Instruct"
DATA_VERSION=1  # data size 15260
DATASET_PATH="/mnt/opsstorage/xushuai/SAP/distill-sap/data/data-${DATA_VERSION}/train.jsonl"
OUTPUT_DIR="/mnt/opsstorage/xushuai/SAP/distill-sap/output/SAP-2.5_3B-v${DATA_VERSION}"
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8
NUM_EPOCHS=4

echo "[INFO] Using output directory: $OUTPUT_DIR"

# =====================
#       TRAINING
# =====================
echo "[INFO] Starting training..."

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --system None \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_length 4000 \
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
    --torch_dtype float16 \
    --add_version false \
    --output_dir "$OUTPUT_DIR" \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --logging_steps 1 \
    --report_to swanlab \
    --use_liger_kernel true \
    --swanlab_project sap

echo "[INFO] Done. Training complete."

# End of script
