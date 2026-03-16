#!/bin/bash
# Download GGUF model files for llama.cpp SYCL benchmarks.
#
# These are quantized models from HuggingFace in GGUF format.
# Q4_0 has optimized SYCL kernels (~30% faster than Q4_K_M on Intel Arc),
# but most models only publish Q4_K_M.
#
# Adjust GGUF_DIR to match your setup.

set -e

GGUF_DIR="${GGUF_DIR:-/media/models/gguf}"
cd "$GGUF_DIR"

echo "Downloading GGUF models to $GGUF_DIR"
echo ""

echo "=== 1/6 Qwen3-14B-Q4_0 ==="
curl -L --progress-bar -o Qwen3-14B-Q4_0.gguf \
  'https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_0.gguf'
ls -lh Qwen3-14B-Q4_0.gguf

echo "=== 2/6 Qwen3-8B-Q4_K_M ==="
curl -L --progress-bar -o Qwen3-8B-Q4_K_M.gguf \
  'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf'
ls -lh Qwen3-8B-Q4_K_M.gguf

echo "=== 3/6 gemma-2-9b-it-Q4_K_M ==="
curl -L --progress-bar -o gemma-2-9b-it-Q4_K_M.gguf \
  'https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M-fp16.gguf'
ls -lh gemma-2-9b-it-Q4_K_M.gguf

echo "=== 4/6 Phi-4-reasoning-Q4_K_M ==="
curl -L --progress-bar -o Phi-4-reasoning-Q4_K_M.gguf \
  'https://huggingface.co/unsloth/Phi-4-reasoning-GGUF/resolve/main/phi-4-reasoning-Q4_K_M.gguf'
ls -lh Phi-4-reasoning-Q4_K_M.gguf

echo "=== 5/6 Mistral-7B-Instruct-v0.3-Q4_K_M ==="
curl -L --progress-bar -o Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  'https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'
ls -lh Mistral-7B-Instruct-v0.3-Q4_K_M.gguf

echo "=== 6/6 DeepSeek-R1-Distill-Qwen-14B-Q4_0 ==="
curl -L --progress-bar -o DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf \
  'https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf'
ls -lh DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf

echo ""
echo "=== ALL DONE ==="
ls -lh *.gguf
