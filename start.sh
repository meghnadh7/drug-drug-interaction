#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# start.sh — Launch the DDI Extraction System
#
# Usage:   bash start.sh
#
# Starts two processes:
#   1. Flask API  → http://localhost:5001   (BioBERT model)
#   2. React UI   → http://localhost:3000   (opens automatically)
# ──────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate Python venv ──────────────────────────────────────
if [ -d "venv" ]; then
  source venv/bin/activate
else
  echo "ERROR: venv not found. Run: python3 -m venv venv && pip install -r requirements.txt"
  exit 1
fi

# ── Start Flask API in background ────────────────────────────
echo "→ Starting Flask API on http://localhost:5001 ..."
python api.py &
API_PID=$!
echo "  Flask PID: $API_PID"

# Give the API a moment to load the model (~30s first run, ~5s after)
sleep 3

# ── Start React dev server in foreground ─────────────────────
echo "→ Starting React UI on http://localhost:3000 ..."
cd ui && npm run dev -- --open

# ── Cleanup on exit ──────────────────────────────────────────
trap "echo '→ Stopping API (PID $API_PID)'; kill $API_PID 2>/dev/null" EXIT
