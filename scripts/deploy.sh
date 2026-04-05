#!/usr/bin/env bash
# Deploy project to active vast.ai instance.
#
# Usage:
#   bash scripts/deploy.sh              # Deploy (git pull + restart)
#   bash scripts/deploy.sh --check      # Verify instance is reachable
#   bash scripts/deploy.sh --setup      # First-time setup on fresh instance
#   bash scripts/deploy.sh --sync-models # Sync .onnx.data weights
#
# Reads vast-instances.json for active instance connection details.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTANCES_FILE="$PROJECT_DIR/vast-instances.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

# --- Read instance config ---
if [ ! -f "$INSTANCES_FILE" ]; then
    err "No $INSTANCES_FILE found. Rent an instance first: /vast-gpu provision"
fi

INSTANCE=$(python3 -c "
import json, sys
with open('$INSTANCES_FILE') as f:
    instances = json.load(f)
for inst in instances:
        if inst.get('status') == 'running':
            print(json.dumps(inst))
            sys.exit(0)
print('', file=sys.stderr)
" 2>/dev/null)

if [ -z "$INSTANCE" ]; then
    err "No running instance found in vast-instances.json"
fi

SSH_HOST=$(echo "$INSTANCE" | python3 -c "import json,sys; print(json.load(sys.stdin)['ssh_host'])")
SSH_PORT=$(echo "$INSTANCE" | python3 -c "import json,sys; print(json.load(sys.stdin)['ssh_port'])")
SSH_KEY=$(echo "$INSTANCE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh_key','~/.ssh/id_rsa_remote_nopass'))")
PROJECT_DIR_REMOTE=$(echo "$INSTANCE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('project_dir','/workspace/project'))")
VENV_REMOTE=$(echo "$INSTANCE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('venv','/root/project-env'))")
GRADIO_PORT=$(echo "$INSTANCE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('gradio_port','7860'))")

# Expand ~ in SSH key
SSH_KEY="${SSH_KEY/#\~/$HOME}"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=15 -i $SSH_KEY -p $SSH_PORT"
SSH_CMD="ssh $SSH_OPTS root@$SSH_HOST"

# --- Helper: run command on remote ---
remote() {
    $SSH_CMD bash -c "$1" 2>&1
}

# --- Flags ---
CHECK_ONLY=false
DO_SETUP=false
SYNC_MODELS=false

for arg in "$@"; do
    case "$arg" in
        --check)      CHECK_ONLY=true ;;
        --setup)      DO_SETUP=true ;;
        --sync-models) SYNC_MODELS=true ;;
        *) warn "Unknown flag: $arg";;
    esac
done

# --- Check instance reachability ---
log "Instance: root@$SSH_HOST:$SSH_PORT"
if ! $SSH_CMD echo ok 2>/dev/null; then
    err "Cannot reach instance. SSH failed."
fi
log "SSH connection OK"

if [ "$CHECK_ONLY" = true ]; then
    GRADIO_STATUS=$(remote "PATH=/root/.local/bin:\$PATH && source $VENV_REMOTE/bin/activate 2>/dev/null && python3 -c \"
import urllib.request
try:
    r = urllib.request.urlopen('http://localhost:7860/', timeout=5)
    print(f'HTTP {r.status}')
except Exception as e:
    print(f'DOWN ({e})')
\" 2>/dev/null")
    echo "$GRADIO_STATUS" | grep -q "HTTP 200" && log "Gradio running on port 7860 (external: $GRADIO_PORT)" || warn "Gradio not responding: $GRADIO_STATUS"
    remote "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader" || true
    exit 0
fi

# --- Sync model weights ---
if [ "$SYNC_MODELS" = true ]; then
    log "Syncing ONNX model weights (.onnx.data)..."
    # Find and sync all .onnx.data files
    for data_file in "$PROJECT_DIR"/data/models/*.onnx.data; do
        [ -f "$data_file" ] || continue
        fname=$(basename "$data_file")
        size=$(du -h "$data_file" | cut -f1)
        log "  Uploading $fname ($size)..."

        # Split large files (>50MB) into 10MB chunks for reliability
        FILE_SIZE=$(stat -c%s "$data_file" 2>/dev/null || echo 0)
        if [ "$FILE_SIZE" -gt 52428800 ]; then
            TMPDIR=$(mktemp -d)
            split -b 10M "$data_file" "$TMPDIR/chunk_"
            for chunk in "$TMPDIR"/chunk_*; do
                scp -o StrictHostKeyChecking=no -o ServerAliveInterval=10 -i "$SSH_KEY" -P "$SSH_PORT" \
                    "$chunk" "root@$SSH_HOST:/tmp/" 2>/dev/null
            done
            remote "cat /tmp/chunk_* > $PROJECT_DIR_REMOTE/data/models/$fname && rm /tmp/chunk_*"
            rm -rf "$TMPDIR"
        else
            scp -o StrictHostKeyChecking=no -o ServerAliveInterval=10 -i "$SSH_KEY" -P "$SSH_PORT" \
                "$data_file" "root@$SSH_HOST:$PROJECT_DIR_REMOTE/data/models/" 2>/dev/null
        fi
        log "  $fname done"
    done
    log "Model sync complete"
fi

# --- First-time setup ---
if [ "$DO_SETUP" = true ]; then
    log "Running first-time setup..."

    remote bash << 'SETUP_EOF'
export DEBIAN_FRONTEND=noninteractive

# Update mirrors (Poland)
sed -i 's|http://archive.ubuntu.com|http://pl.archive.ubuntu.com|g' /etc/apt/sources.list 2>/dev/null || true
for f in /etc/apt/sources.list.d/*.list; do
    sed -i 's|http://archive.ubuntu.com|http://pl.archive.ubuntu.com|g' $f 2>/dev/null || true
done

# System packages
apt-get update -qq
apt-get install -y -qq software-properties-common > /dev/null 2>&1
add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev python3-pip git curl rsync libxcb1 libgl1 libglib2.0-0 ffmpeg > /dev/null 2>&1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 > /dev/null 2>&1
echo "System packages OK"

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
echo "uv installed"

# Create venv
PATH=/root/.local/bin:$PATH
cd /workspace/project
uv venv --python 3.12 /root/project-env
source /root/project-env/bin/activate
echo "Venv created"

# PyTorch cu124 + all deps (with CUDA libs)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy typing_extensions sympy filelock networkx jinja2 fsspec packaging
uv pip install onnxruntime-gpu --force-reinstall --no-deps
echo "PyTorch + ONNX Runtime OK"

# Project deps
uv pip install opencv-python-headless scipy scikit-learn pandas matplotlib pillow gradio dtw-python rtmlib ultralytics filterpy lapx einops timm mmengine
uv pip install onnxruntime-gpu --force-reinstall --no-deps
uv pip install av --force-reinstall --no-deps
uv pip install -e . --no-deps
echo "Project deps installed"

# Verify stack
python3 -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0)}')"
python3 -c "import onnxruntime as ort; print(f'onnxruntime={ort.__version__}, CUDA={\"CUDAExecutionProvider\" in ort.get_available_providers()}')"
python3 -c "import cv2; print(f'cv2={cv2.__version__}')"
python3 -c "import gradio; print(f'gradio={gradio.__version__}')"
echo "SETUP COMPLETE"
SETUP_EOF

    log "First-time setup done"
fi

# --- Deploy (git pull + restart) ---
log "Deploying to instance..."

remote bash << 'DEPLOY_EOF'
cd /workspace/project
PATH=/root/.local/bin:$PATH
source /root/project-env/bin/activate

# Pull latest code
git pull origin master

# Reinstall project (editable)
uv pip install -e . --no-deps 2>/dev/null

# Fix deps that uv may have removed
uv pip install av opencv-python-headless onnxruntime-gpu --no-deps --force-reinstall 2>/dev/null

# Kill old Gradio
ps aux | grep gradio | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null
sleep 1

# Start new Gradio
nohup python3 scripts/gradio_app.py > /tmp/gradio.log 2>&1 &
echo "Gradio PID: $!"

# Wait and verify
sleep 6
PATH=/root/.local/bin:$PATH source $VENV_REMOTE/bin/activate 2>/dev/null
python3 -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://localhost:7860/', timeout=5)
    print(f'HTTP {r.status}')
except Exception as e:
    print(f'DOWN ({e})')
"
DEPLOY_EOF

log "Deploy complete"
