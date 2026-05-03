#!/usr/bin/env fish

# SOOG Local Development Launcher (Fixed Version)
# Runs both Backend (Flask) and Frontend (Nuxt) concurrently

# Use absolute path for the root directory
set ROOT_DIR (realpath (dirname (status --current-filename)))
echo "🚀 Starting SOOG Development Environment in $ROOT_DIR..."

# 1. Setup Backend
set BACKEND_DIR "$ROOT_DIR/backend"
set PYTHON_BIN "$BACKEND_DIR/venv/bin/python"
set BACKEND_PORT 10000

if not test -f "$PYTHON_BIN"
    echo "❌ Error: Backend virtual environment not found at $PYTHON_BIN"
    exit 1
end

# Check and clear port
if lsof -i :$BACKEND_PORT > /dev/null
    echo "⚠️  Port $BACKEND_PORT is busy. Cleaning up..."
    lsof -ti :$BACKEND_PORT | xargs kill -9
end

# Start Backend
echo "📡 Starting Backend (Flask)..."
cd $BACKEND_DIR
$PYTHON_BIN app.py &
set BACKEND_PID (jobs -p | tail -n 1)
echo "✅ Backend started (PID: $BACKEND_PID)"

# 2. Setup Frontend
set FRONTEND_DIR "$ROOT_DIR/frontend"
if not test -d "$FRONTEND_DIR"
    echo "❌ Error: Frontend directory not found at $FRONTEND_DIR"
    exit 1
end

echo "🎨 Starting Frontend (Nuxt)..."
cd $FRONTEND_DIR

# Try bun if available, fallback to npm
if type -q bun
    bun run dev &
else
    npm run dev &
end
set FRONTEND_PID (jobs -p | tail -n 1)
echo "✅ Frontend started (PID: $FRONTEND_PID)"

# 3. Handle cleanup on exit (Fixed for Fish)
function cleanup_soog
    echo -e "\n🛑 Shutting down SOOG..."
    # Kill the PIDs if they exist
    if test -n "$BACKEND_PID"
        kill $BACKEND_PID 2>/dev/null
    end
    if test -n "$FRONTEND_PID"
        kill $FRONTEND_PID 2>/dev/null
    end
    echo "👋 Bye!"
    exit 0
end

# Trap signals
function __on_sigint --on-signal SIGINT
    cleanup_soog
end
function __on_sigterm --on-signal SIGTERM
    cleanup_soog
end

# Wait for background processes
echo "✨ Both services are running. Press Ctrl+C to stop."
while true
    sleep 2
    # Check if processes are still alive, if not, exit
    if not kill -0 $BACKEND_PID 2>/dev/null
        echo "⚠️  Backend stopped unexpectedly."
        cleanup_soog
    end
end
