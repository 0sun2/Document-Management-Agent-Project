#!/bin/bash

echo "🌐 Starting ngrok tunnel..."
echo ""

# ngrok이 이미 실행 중인지 확인
if pgrep -f "ngrok http" > /dev/null; then
    echo "⚠️  ngrok is already running!"
    echo ""
    echo "Current tunnel URL:"
    curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url'
    echo ""
    echo "To restart, run: ./stop-ngrok.sh first"
    exit 1
fi

# ngrok 백그라운드 실행
ngrok http 8080 > /tmp/ngrok.log 2>&1 &

echo "Waiting for ngrok to start..."
sleep 3

# URL 가져오기
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | jq -r '.tunnels[0].public_url')

if [ -z "$PUBLIC_URL" ] || [ "$PUBLIC_URL" = "null" ]; then
    echo "❌ Failed to start ngrok!"
    echo "Check: /tmp/ngrok.log"
    exit 1
fi

echo ""
echo "✅ ngrok tunnel started!"
echo ""
echo "🌐 Public URL: $PUBLIC_URL"
echo "📊 Dashboard:  http://localhost:4040"
echo ""
echo "⚠️  Note: Free plan URLs change when restarted"
echo ""
echo "Share this URL with others to access your chatbot!"
echo ""
