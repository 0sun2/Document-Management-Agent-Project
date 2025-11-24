#!/bin/bash
# vLLM 서버 시작 스크립트
# CUDA 메모리 부족 문제를 해결하기 위한 설정 포함

# 환경 변수 설정
export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/home/lys/Desktop/qwen3_4b_ft/final_merged_model}"
export VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen3-4b-ft}"
export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
export VLLM_PORT="${VLLM_PORT:-9000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

# 메모리 최적화 설정
# GPU 메모리 사용률을 낮춤 (기본값: 0.9, OOM 발생 시 0.75로 낮춤)
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"

# 최대 시퀀스 수를 낮춤 (기본값은 더 높을 수 있음, OOM 발생 시 128로 제한)
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"

# PyTorch CUDA 메모리 할당 최적화 (메모리 단편화 방지)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting vLLM server..."
echo "Model: $VLLM_MODEL_PATH"
echo "Port: $VLLM_PORT"
echo "GPU Memory Utilization: $VLLM_GPU_MEMORY_UTILIZATION"

# vLLM 서버 시작
# 원래 명령어에 메모리 최적화 옵션 및 tool calling 옵션 추가
python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL_PATH" \
    --served-model-name "$VLLM_SERVED_MODEL_NAME" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --api-key "$VLLM_API_KEY" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
