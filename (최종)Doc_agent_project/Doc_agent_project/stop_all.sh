#!/bin/bash

# 통합 서비스 종료 스크립트
# 실행 중인 모든 서비스를 종료합니다.

cd "$(dirname "$0")"

LOGS_DIR="logs"
PID_FILE="$LOGS_DIR/pids.txt"

echo "🛑 서비스를 종료합니다..."

# PID 파일에서 프로세스 종료
if [ -f "$PID_FILE" ]; then
    while read pid service; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  - $service 종료 중 (PID: $pid)"
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
fi

# 추가 프로세스 확인 및 종료
echo "  - 남은 프로세스 확인 중..."

# vLLM 관련 프로세스
VLLM_PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server" 2>/dev/null)
if [ ! -z "$VLLM_PIDS" ]; then
    echo "  - vLLM 프로세스 종료 중..."
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
fi

# 백엔드 관련 프로세스
BACKEND_PIDS=$(pgrep -f "uvicorn main:app" 2>/dev/null)
if [ ! -z "$BACKEND_PIDS" ]; then
    echo "  - 백엔드 프로세스 종료 중..."
    pkill -f "uvicorn main:app" 2>/dev/null
fi

# 프론트엔드 관련 프로세스 (vite)
FRONTEND_PIDS=$(pgrep -f "vite" 2>/dev/null)
if [ ! -z "$FRONTEND_PIDS" ]; then
    echo "  - 프론트엔드 프로세스 종료 중..."
    pkill -f "vite" 2>/dev/null
fi

# 스크립트 프로세스
pkill -f "start_vllm.sh" 2>/dev/null
pkill -f "start_backend.sh" 2>/dev/null
pkill -f "start_frontend.sh" 2>/dev/null

sleep 1

echo "✅ 모든 서비스가 종료되었습니다."
