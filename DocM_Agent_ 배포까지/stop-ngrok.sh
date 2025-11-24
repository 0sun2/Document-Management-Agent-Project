#!/bin/bash

echo "🛑 Stopping ngrok tunnel..."
echo ""

if pkill -f "ngrok http" 2>/dev/null; then
    echo "✅ ngrok tunnel stopped"
else
    echo "ℹ️  ngrok was not running"
fi

echo ""
