#!/bin/bash
set -e

REMOTE_HOST="ubuntu@115.159.223.227"
REMOTE_DIR="/opt/lingkong-webui"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Config:"
echo "  Remote: $REMOTE_HOST:$REMOTE_DIR"
echo "  Local:  $LOCAL_DIR"
echo ""

# 1. Upload server_lite.py as server.py (matching CI behavior)
echo "[1/3] Uploading server_lite.py as server.py..."
scp "$LOCAL_DIR/server_lite.py" "$REMOTE_HOST:$REMOTE_DIR/server.py"

# 2. Upload static files
echo "[2/3] Uploading static files..."
scp -r "$LOCAL_DIR/static/"* "$REMOTE_HOST:$REMOTE_DIR/static/"

# 3. Restart service
echo "[3/3] Restarting service..."
ssh $REMOTE_HOST "sudo systemctl restart lingkong-webui"

echo "Deployment Done!"
