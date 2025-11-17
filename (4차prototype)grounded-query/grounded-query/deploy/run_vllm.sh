#!/bin/bash
set -euo pipefail

python3 -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_PATH}" \
    --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --api-key "${VLLM_API_KEY}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
