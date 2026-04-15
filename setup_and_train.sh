#!/usr/bin/env bash
# =============================================================================
# Fine-tune Gemma on a legal JSONL dataset on a fresh H100 VM.
#
# Usage:
#   chmod +x setup_and_train.sh
#   ./setup_and_train.sh
#
# What this does:
#   1. Installs system packages (build tools, git, python)
#   2. Verifies NVIDIA driver + CUDA is visible
#   3. Creates a Python venv and installs PyTorch (cu121) + HF stack
#   4. (Optional) installs flash-attn for speed on H100
#   5. Logs in to Hugging Face (Gemma is gated - you need a token)
#   6. Runs the training script on legislation.jsonl
#
# Expected layout in the working dir:
#   setup_and_train.sh          (this file)
#   train_gemma.py              (created below by this script if missing)
#   legislation.jsonl           (your dataset)
# =============================================================================

set -euo pipefail

# ---- user-configurable ------------------------------------------------------
# NOTE: Unsloth's notebook calls it "gemma-4-E4B" but on HF the real model id
# for the E4B checkpoint is google/gemma-3n-E4B (Gemma 3n family). Swap to
# whichever id you have access to.
MODEL_ID="${MODEL_ID:-google/gemma-3n-E4B-it}"
DATA_PATH="${DATA_PATH:-$(pwd)/legislation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/gemma-legal-lora}"
VENV_DIR="${VENV_DIR:-$(pwd)/.venv}"
HF_TOKEN="${HF_TOKEN:-}"   # export HF_TOKEN=hf_xxx before running, or paste below
# -----------------------------------------------------------------------------

echo "==> 1/6  System packages"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    python3 python3-venv python3-pip python3-dev \
    ninja-build pkg-config

echo "==> 2/6  GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "!! nvidia-smi not found. This script assumes the VM image already"
    echo "   ships with NVIDIA drivers (most H100 cloud images do)."
    echo "   If not, install drivers first (e.g. 'sudo ubuntu-drivers install')"
    echo "   and reboot before rerunning."
    exit 1
fi
nvidia-smi | head -20

echo "==> 3/6  Python venv + PyTorch (CUDA 12.1)"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Torch built against CUDA 12.1 — works on H100 (compute cap 9.0)
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1

pip install \
    "transformers>=4.45,<4.50" \
    "accelerate>=0.34" \
    "peft>=0.13" \
    "trl>=0.11" \
    "datasets>=3.0" \
    "bitsandbytes>=0.44" \
    "sentencepiece" "protobuf" "einops" "safetensors" \
    "huggingface_hub[cli]>=0.25"

echo "==> 4/6  (optional) flash-attn 2 for H100 speedup"
# Skipping flash-attn if it fails to build is fine; training still works.
pip install --no-build-isolation "flash-attn==2.6.3" || \
    echo "   flash-attn build skipped — training will fall back to SDPA"

echo "==> 5/6  Hugging Face login (Gemma is a gated model)"
if [[ -n "$HF_TOKEN" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "   HF_TOKEN not set. Launching interactive login — paste your token:"
    huggingface-cli login
fi

echo "==> 6/6  Sanity checks"
[[ -f "$DATA_PATH" ]] || { echo "!! dataset not found at $DATA_PATH"; exit 1; }
[[ -f "$(pwd)/train_gemma.py" ]] || { echo "!! train_gemma.py missing in cwd"; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo
echo "==> Starting fine-tune"
echo "    model : $MODEL_ID"
echo "    data  : $DATA_PATH"
echo "    out   : $OUTPUT_DIR"
echo

# Single-GPU H100. For multi-GPU, prefix with:
#   accelerate launch --num_processes <N>
python train_gemma.py \
    --model_id "$MODEL_ID" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR"

echo
echo "==> Done. LoRA adapters saved to $OUTPUT_DIR"
