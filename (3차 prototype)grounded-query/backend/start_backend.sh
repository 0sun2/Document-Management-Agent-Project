#!/bin/bash

# FastAPI 백엔드 시작 스크립트

cd "$(dirname "$0")"

# 가상 환경 활성화 (~/venv/agent 사용)
if [ -d ~/venv/agent ]; then
    echo "가상 환경 활성화: ~/venv/agent"
    source ~/venv/agent/bin/activate
else
    echo "⚠️  가상 환경을 찾을 수 없습니다: ~/venv/agent"
    echo "프로젝트 로컬 가상환경을 생성합니다..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# 의존성 업데이트 (Supabase 등 새 패키지)
echo "📦 의존성 확인 및 설치 중..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# .env 파일이 없으면 .env.example을 복사
if [ ! -f ".env" ]; then
    echo ".env 파일이 없습니다. .env.example을 복사합니다..."
    cp .env.example .env
    echo "✅ .env 파일이 생성되었습니다. 필요시 수정하세요."
fi

# 환경 변수 로드 (python-dotenv가 있으면 자동으로 로드되지만, 명시적으로 export)
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# FastAPI 서버 시작
echo "FastAPI 백엔드 서버를 시작합니다..."
echo "📍 서버 주소: http://${FASTAPI_HOST:-127.0.0.1}:${FASTAPI_PORT:-8000}"
echo "📚 API 문서: http://${FASTAPI_HOST:-127.0.0.1}:${FASTAPI_PORT:-8000}/docs"
echo ""
uvicorn main:app \
    --host "${FASTAPI_HOST:-127.0.0.1}" \
    --port "${FASTAPI_PORT:-8000}" \
    --reload

