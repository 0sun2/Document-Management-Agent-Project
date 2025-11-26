#!/bin/bash

echo "🚀 Starting RAG Chatbot System..."
echo ""

# vLLM 시작
echo "📡 Starting vLLM server on port 9000..."
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /home/lys/Desktop/qwen3_4b_ft/final_merged_model \
  --served-model-name qwen3-4b-ft \
  --host 0.0.0.0 \
  --port 9000 \
  --dtype auto \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 128 \
  --api-key EMPTY \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > vllm.log 2>&1 &

echo "   Waiting for vLLM to start..."
sleep 10

# vLLM 상태 확인
if curl -s http://localhost:9000/v1/models -H "Authorization: Bearer EMPTY" > /dev/null 2>&1; then
    echo "   ✅ vLLM server started successfully"
else
    echo "   ⚠️  vLLM server may not be ready yet (check vllm.log)"
fi

echo ""

# Docker 시작
echo "🐳 Starting Docker containers..."
docker compose up -d

echo "   Waiting for containers to be healthy..."
sleep 5

# 상태 확인
echo ""
echo "📊 System Status:"
docker compose ps

echo ""
echo "✅ System started!"
echo ""
echo "🌐 Access URLs:"
echo "   Local:  http://localhost:8080"
echo "   API:    http://localhost:8080/api"
echo ""
echo "📋 Useful commands:"
echo "   View logs:   docker logs -f rag-backend"
echo "   Stop:        ./stop.sh"
echo "   Start ngrok: ngrok http 8080"
echo ""
