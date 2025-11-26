#!/bin/bash

# 통합 서비스 시작 스크립트
# vLLM, 백엔드, 프론트엔드를 한 번에 실행합니다.

# 스크립트가 위치한 디렉토리로 이동
cd "$(dirname "$0")"

# PID 파일 및 로그 디렉토리 설정
LOGS_DIR="logs"
PID_FILE="$LOGS_DIR/pids.txt"

# 로그 디렉토리 생성
mkdir -p "$LOGS_DIR"

# 기존 프로세스 확인 및 종료 함수
cleanup() {
    echo ""
    echo "🛑 서비스를 종료합니다..."

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

    # 추가 프로세스 확인 (스크립트 이름으로)
    pkill -f "start_vllm.sh" 2>/dev/null
    pkill -f "start_backend.sh" 2>/dev/null
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
    pkill -f "uvicorn main:app" 2>/dev/null

    echo "✅ 모든 서비스가 종료되었습니다."
    exit 0
}

# 종료 시그널 처리
trap cleanup SIGINT SIGTERM

# 실행 권한 확인 및 부여
chmod +x backend/start_vllm.sh backend/start_backend.sh frontend/start_frontend.sh 2>/dev/null

# PID 파일 초기화
> "$PID_FILE"

echo "🚀 Document Management RAG System 시작"
echo ""

# 1. vLLM 서버 시작
echo "📦 vLLM 서버 시작 중..."
(
    cd backend
    ./start_vllm.sh > "../$LOGS_DIR/vllm.log" 2>&1
) &
VLLM_PID=$!
echo "$VLLM_PID vLLM" >> "$PID_FILE"
echo "  ✅ vLLM 서버 시작됨 (PID: $VLLM_PID, 로그: $LOGS_DIR/vllm.log)"
echo "     포트: 9000"
sleep 3

# 2. 백엔드 서버 시작
echo "🔧 백엔드 서버 시작 중..."
(
    cd backend
    ./start_backend.sh > "../$LOGS_DIR/backend.log" 2>&1
) &
BACKEND_PID=$!
echo "$BACKEND_PID Backend" >> "$PID_FILE"
echo "  ✅ 백엔드 서버 시작됨 (PID: $BACKEND_PID, 로그: $LOGS_DIR/backend.log)"
echo "     포트: 8000"
sleep 3

# 3. 프론트엔드 서버 시작 (포그라운드)
echo "🎨 프론트엔드 서버 시작 중..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 모든 서비스가 시작되었습니다!"
echo ""
echo "📍 서비스 주소:"
echo "  - 프론트엔드:    http://127.0.0.1:8080"
echo "  - 백엔드 API:    http://127.0.0.1:8000"
echo "  - API 문서:      http://127.0.0.1:8000/docs"
echo "  - vLLM 서버:     http://127.0.0.1:9000"
echo ""
echo "📝 로그 파일:"
echo "  - vLLM:          $LOGS_DIR/vllm.log"
echo "  - 백엔드:        $LOGS_DIR/backend.log"
echo "  - 프론트엔드:    콘솔 출력"
echo ""
echo "⏹️  종료하려면 Ctrl+C를 누르세요"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 프론트엔드는 포그라운드로 실행 (Ctrl+C로 종료 가능)
cd frontend
./start_frontend.sh

# 프론트엔드가 종료되면 다른 서비스도 종료
cleanup
