#!/bin/bash
# Start React frontend development server

cd "$(dirname "$0")"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "Warning: .env.local not found, using defaults"
fi

# Start development server
echo "Starting frontend development server..."
npm run dev
