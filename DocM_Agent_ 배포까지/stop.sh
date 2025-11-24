#!/bin/bash

echo "🛑 Stopping RAG Chatbot System..."
echo ""

# ngrok 중지
echo "📡 Stopping ngrok tunnel..."
if pkill ngrok 2>/dev/null; then
    echo "   ✅ ngrok stopped"
else
    echo "   ℹ️  ngrok was not running"
fi

echo ""

# Docker 중지
echo "🐳 Stopping Docker containers..."
docker compose down
echo "   ✅ Docker containers stopped"

echo ""

# vLLM 중지
echo "📡 Stopping vLLM server..."
if pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; then
    echo "   ✅ vLLM server stopped"
else
    echo "   ℹ️  vLLM was not running"
fi

echo ""
echo "✅ System stopped!"
echo ""
