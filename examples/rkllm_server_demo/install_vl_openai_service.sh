#!/bin/bash
set -euo pipefail

# Run this script on the board.
# It compiles multimodal binaries locally, installs Python deps,
# and registers a systemd service for OpenAI-compatible VL API.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="$ROOT_DIR/examples/multimodal_model_demo/deploy"
SERVER_DEMO_DIR="$ROOT_DIR/examples/rkllm_server_demo"
SERVER_SRC_DIR="$SERVER_DEMO_DIR/rkllm_server"

WORKSHOP="/opt/rkllm-vl-openai"
TARGET_PLATFORM="rk3588"
RKLLM_MODEL_PATH=""
VISION_MODEL_PATH=""
MODEL_NAME="qwen3-vl-4b-rk3588"
PORT="8080"
RKNN_CORE_NUM="3"
MAX_CONTEXT_LEN="4096"
MAX_NEW_TOKENS="1024"
IMG_START="<|vision_start|>"
IMG_END="<|vision_end|>"
IMG_CONTENT="<|image_pad|>"

usage() {
  cat <<EOF
Usage:
  sudo bash install_vl_openai_service.sh \\
    --rkllm_model /abs/path/model.rkllm \\
    --vision_model /abs/path/model.rknn \\
    [--workshop /opt/rkllm-vl-openai] \\
    [--platform rk3588] \\
    [--model_name qwen3-vl-4b-rk3588] \\
    [--port 8080] \\
    [--rknn_core_num 3] \\
    [--max_context_len 4096] \\
    [--max_new_tokens 1024] \\
    [--img_start "<|vision_start|>"] \\
    [--img_end "<|vision_end|>"] \\
    [--img_content "<|image_pad|>"]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workshop)
      WORKSHOP="$2"; shift 2 ;;
    --platform)
      TARGET_PLATFORM="$2"; shift 2 ;;
    --rkllm_model)
      RKLLM_MODEL_PATH="$2"; shift 2 ;;
    --vision_model)
      VISION_MODEL_PATH="$2"; shift 2 ;;
    --model_name)
      MODEL_NAME="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --rknn_core_num)
      RKNN_CORE_NUM="$2"; shift 2 ;;
    --max_context_len)
      MAX_CONTEXT_LEN="$2"; shift 2 ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"; shift 2 ;;
    --img_start)
      IMG_START="$2"; shift 2 ;;
    --img_end)
      IMG_END="$2"; shift 2 ;;
    --img_content)
      IMG_CONTENT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$RKLLM_MODEL_PATH" || -z "$VISION_MODEL_PATH" ]]; then
  echo "--rkllm_model and --vision_model are required" >&2
  usage
  exit 1
fi

if [[ ! -f "$RKLLM_MODEL_PATH" ]]; then
  echo "RKLLM model not found: $RKLLM_MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$VISION_MODEL_PATH" ]]; then
  echo "Vision model not found: $VISION_MODEL_PATH" >&2
  exit 1
fi

echo "[1/6] Installing runtime dependencies"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv \
  cmake make g++ pkg-config \
  libopencv-dev

python3 -m pip install --upgrade pip --break-system-packages
python3 -m pip install flask==2.2.2 werkzeug==2.2.2 requests numpy --break-system-packages

echo "[2/6] Building multimodal deploy binaries on board"
rm -rf "$DEPLOY_DIR/build"
mkdir -p "$DEPLOY_DIR/build"
pushd "$DEPLOY_DIR/build" >/dev/null
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64
make -j"$(nproc)"
make install
popd >/dev/null

echo "[3/6] Preparing service workspace: $WORKSHOP"
mkdir -p "$WORKSHOP/lib"
cp "$SERVER_SRC_DIR/flask_vl_openai_server.py" "$WORKSHOP/"
cp "$ROOT_DIR/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so" "$WORKSHOP/lib/"
cp "$ROOT_DIR/scripts/fix_freq_rk3588.sh" "$WORKSHOP/" || true
cp "$ROOT_DIR/scripts/fix_freq_rk3576.sh" "$WORKSHOP/" || true
cp "$ROOT_DIR/scripts/fix_freq_rk3562.sh" "$WORKSHOP/" || true
cp "$ROOT_DIR/scripts/fix_freq_rv1126b.sh" "$WORKSHOP/" || true

IMG_ENC_BIN="$DEPLOY_DIR/install/demo_Linux_aarch64/imgenc"
if [[ ! -x "$IMG_ENC_BIN" ]]; then
  echo "imgenc not found after build: $IMG_ENC_BIN" >&2
  exit 1
fi
cp "$IMG_ENC_BIN" "$WORKSHOP/"

# Ensure imgenc can find its dependent runtime libs.
cp "$DEPLOY_DIR/install/demo_Linux_aarch64/lib"/*.so "$WORKSHOP/lib/" || true

echo "[4/6] Writing environment file"
cat >/etc/default/rkllm-vl-openai <<EOF
WORKSHOP=$WORKSHOP
RKLLM_MODEL_PATH=$RKLLM_MODEL_PATH
VISION_MODEL_PATH=$VISION_MODEL_PATH
TARGET_PLATFORM=$TARGET_PLATFORM
MODEL_NAME=$MODEL_NAME
PORT=$PORT
RKNN_CORE_NUM=$RKNN_CORE_NUM
MAX_CONTEXT_LEN=$MAX_CONTEXT_LEN
MAX_NEW_TOKENS=$MAX_NEW_TOKENS
IMG_START=$IMG_START
IMG_END=$IMG_END
IMG_CONTENT=$IMG_CONTENT
EOF

echo "[5/6] Installing systemd unit"
sed "s#__WORKSHOP__#$WORKSHOP#g" "$SERVER_SRC_DIR/rkllm-vl-openai.service" > /etc/systemd/system/rkllm-vl-openai.service

systemctl daemon-reload
systemctl enable rkllm-vl-openai.service


echo "[6/6] Starting service"
systemctl restart rkllm-vl-openai.service
systemctl --no-pager --full status rkllm-vl-openai.service

echo "Done. API endpoint: http://<board_ip>:$PORT/v1/chat/completions"
