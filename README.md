# Intel Arc A770 LLM Inference Benchmark

**llama.cpp SYCL vs OpenVINO OVMS** — head-to-head comparison on the same hardware, same prompts, same models.

There's very little practical benchmark data for running LLMs on Intel Arc GPUs. This repo provides reproducible benchmarks comparing the two main inference backends available for Intel discrete GPUs.

## Hardware

| Component | Spec |
|-----------|------|
| GPU | Intel Arc A770 16GB |
| CPU | Intel Core i5-13400 |
| RAM | 64GB DDR5 |
| OS | Ubuntu 24.04 |
| Driver | i915 (Linux 6.8+) |

## Results

<!-- RESULTS_TABLE_START -->
*Benchmark in progress — results will be posted here.*
<!-- RESULTS_TABLE_END -->

### Key Findings

<!-- FINDINGS_START -->
*Pending benchmark completion.*
<!-- FINDINGS_END -->

## Backends

### llama.cpp SYCL

[llama.cpp](https://github.com/ggerganov/llama.cpp) with Intel SYCL backend. Uses GGUF quantized models. Docker image: `ghcr.io/ggml-org/llama.cpp:server-intel`.

**Optimizations applied:**
- Flash attention enabled (`--flash-attn 1` + `GGML_SYCL_ENABLE_FLASH_ATTN=1`)
- Relaxed VRAM allocation (`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`) — allows using the full 16GB
- Level Zero backend (`ONEAPI_DEVICE_SELECTOR=level_zero:0`)
- All layers offloaded to GPU (`-ngl 999`)
- Q4_0 quantization where available (has optimized SYCL kernels, ~30% faster than Q4_K_M)

### OpenVINO OVMS

[OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) with GPU target. Downloads INT4 models from HuggingFace automatically. Docker image: `openvino/model_server:latest-gpu`.

**Optimizations applied:**
- u8 KV cache precision (reduces VRAM usage, minimal quality impact)
- Dynamic quantization with group size 32
- Model caching enabled (`--cache_size 4`)
- GPU target device

**Note:** OVMS uses the `/v3` API prefix (not `/v1`). The chat completions endpoint is at `/v3/chat/completions`.

## Models Tested

| Model | Parameters | SYCL Quantization | OpenVINO Format |
|-------|-----------|-------------------|-----------------|
| Qwen3-14B | 14B | Q4_0 | INT4 |
| Qwen3-8B | 8B | Q4_K_M | INT4 |
| Gemma 2 9B IT | 9B | Q4_K_M | INT4 |
| Phi-4 Reasoning | 14B | Q4_K_M | INT4 |
| Mistral 7B Instruct v0.3 | 7B | Q4_K_M | INT4 |
| DeepSeek-R1-Distill-Qwen-14B | 14B | Q4_0 | INT4 |

## Methodology

- 4 diverse prompts (reasoning, coding, logic, creative writing)
- 512 max completion tokens per prompt
- Temperature 0.7
- Single-user (n_parallel=1) to measure raw throughput
- Fresh container per model/backend (cold start, but model cached after first run for OpenVINO)
- 10-second GPU cooldown between backend switches
- Both client-side and server-side tok/s measured where available

## Reproduction

### Prerequisites

1. Intel Arc A770 (or A750/A580) with working i915/xe driver
2. Docker with GPU device passthrough
3. ~50GB disk space for GGUF models + OpenVINO cache

### Setup

```bash
git clone https://github.com/diunox/arc-inference-bench.git
cd arc-inference-bench

# Download GGUF models (~37GB total)
export GGUF_DIR=/path/to/your/gguf/models
bash scripts/download-gguf.sh

# Create OpenVINO cache directory
mkdir -p openvino-models && chmod 777 openvino-models
```

### Run

```bash
# Full benchmark (all models, both backends)
python3 scripts/bench.py

# Single backend
python3 scripts/bench.py --backend sycl
python3 scripts/bench.py --backend openvino

# Single model
python3 scripts/bench.py --model qwen3-14b

# Custom output file
python3 scripts/bench.py --output my-results.json
```

### Configuration

Edit the `CONFIGURATION` section at the top of `scripts/bench.py`:

```python
GGUF_DIR = "/media/models/gguf"          # Host path to GGUF files
OPENVINO_CACHE = "./openvino-models"     # OpenVINO model cache
GPU_RENDER = "/dev/dri/renderD128"       # GPU render device
GPU_CARD = "/dev/dri/card1"              # GPU card device
RENDER_GID = "991"                       # render group GID
VIDEO_GID = "44"                         # video group GID
```

### Docker Compose (production)

The included `docker-compose.yml` runs llama.cpp SYCL as a production service with an optional OpenVINO benchmark profile:

```bash
# Production SYCL inference
docker compose up -d llm

# Add OpenVINO for comparison
docker compose --profile benchmark up -d openvino-llm
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No host part in URL` | Check that model source strings don't have shell variable issues |
| `render group not found` | Use numeric GIDs instead: `group_add: ["44", "991"]` |
| `size_in_bytes <= total_mem_size` | Reduce `--cache_size` (try 4 or 2) |
| `Permission denied` on openvino-models | `chmod 777 openvino-models/` |
| OpenVINO 404 on chat endpoint | Model still loading — OVMS health returns 200 before model is ready |
| Qwen 3.5 fails on all backends | Gated DeltaNet architecture not supported on Intel GPU SYCL kernels |

## Known Limitations

- **Qwen 3.5** uses a novel Gated DeltaNet architecture whose kernels are not implemented for SYCL. It fails on all Intel GPU backends (llama.cpp, OpenVINO, Ollama).
- **Q4_0 vs Q4_K_M**: Q4_0 has hand-optimized SYCL dot-product kernels that are ~30% faster on Intel Arc, but most model publishers only provide Q4_K_M GGUFs.
- **Ollama** has no SYCL support and falls back to CPU-only on Intel GPUs (3-7 tok/s).
- **IPEX-LLM** (formerly BigDL-LLM) is archived/deprecated as of 2025.

## License

MIT
