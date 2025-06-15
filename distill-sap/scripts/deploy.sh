#!/bin/bash

source /mnt/opsstorage/xushuai/SAP/.venv/bin/activate

CUDA_VISIBLE_DEVICES=2,3 \
nohup vllm serve /mnt/opsstorage/xushuai/SAP/distill-sap/output/SAP-2.5_3B-v1/checkpoint-240 \
    --host 0.0.0.0 \
    --tensor-parallel-size 2 \
    --port 20020 \
    --gpu-memory-utilization 0.98 \
    --served-model-name sap_distill sap_agent \
    --max-model-len 8192 \
    --dtype half > /mnt/opsstorage/xushuai/SAP/vllm.log 2>&1 &


# curl -X POST http://172.18.36.65:20020/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "sap_distill",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What is the capital of France?"
#             }
#         ],
#         "max_tokens": 100,
#         "temperature": 0.7
#     }'