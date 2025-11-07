#!/bin/bash

# 프런트엔드 시작 스크립트

cd "$(dirname "$0")"

# 현재 디렉토리 확인 (프런트엔드 디렉토리여야 함)
if [ ! -f "package.json" ]; then
    echo "❌ 오류: package.json을 찾을 수 없습니다. 프런트엔드 디렉토리에서 실행하세요."
    exit 1
fi

# .env.local 파일이 없으면 생성
if [ ! -f ".env.local" ]; then
    echo "📝 .env.local 파일이 없습니다. 생성 중... (프런트엔드 디렉토리: $(pwd))"
    echo "VITE_FASTAPI_URL=http://127.0.0.1:8000" > .env.local
    echo "✅ .env.local 파일이 생성되었습니다."
fi

# node_modules가 없으면 의존성 설치
if [ ! -d "node_modules" ]; then
    echo "의존성 설치 중..."
    npm install
fi

# 프런트엔드 서버 시작
echo "프런트엔드 개발 서버를 시작합니다..."
echo "📍 서버 주소: http://127.0.0.1:8080"
echo ""
npm run dev

