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
| Model | Params | SYCL (tok/s) | OpenVINO (tok/s) | Advantage |
|-------|--------|:------------:|:----------------:|:---------:|
| Qwen3-1.7B | 1.7B | 33.2 | 65.4 | +97% |
| Llama 3.2 3B Instruct | 3B | 33.0 | 78.4 | +138% |
| Qwen3-4B | 4B | 25.6 | 61.5 | +140% |
| Mistral 7B Instruct v0.3 | 7B | 25.6 | 57.3 | +124% |
| Qwen3-8B | 8B | 21.0 | 45.5 | +117% |
| Gemma 2 9B IT | 9B | 15.4 | 31.7 | +106% |
| Qwen3-14B | 14B | 14.7 | 23.3 | +59% |
| Phi-4 Reasoning | 14B | 13.5 | 31.1 | +130% |
| DeepSeek-R1-Distill-Qwen-14B | 14B | 13.4 | 29.1 | +117% |
| Mistral Small 3.1 24B | 24B | OOM | OOM | — |

*Tested 2026-03-15. Sorted by parameter count.*
<!-- RESULTS_TABLE_END -->

### Key Findings

<!-- FINDINGS_START -->
- **OpenVINO is faster across every model tested**, from +59% (Qwen3-14B) to +140% (Qwen3-4B)
- **Average advantage: +113%** — OpenVINO is roughly 2x faster than SYCL on Intel Arc
- **Smaller models see larger gains**: OpenVINO's optimization pipeline extracts more from models that don't saturate the GPU's compute units
- **Qwen3-14B is the outlier** at only +59% — likely because SYCL has hand-optimized Q4_0 kernels for this model, partially closing the gap
- **24B models exceed 16GB VRAM** on both backends — the Arc A770's practical ceiling is ~14B parameters at 4-bit quantization
- **Qwen3 scaling curve** (1.7B → 4B → 8B → 14B): OpenVINO scales more efficiently as models get smaller, while SYCL performance plateaus earlier
- **SYCL Q4_0 vs Q4_K_M matters**: Models with Q4_0 GGUFs (Qwen3-14B, DeepSeek-R1-14B) show smaller OpenVINO advantages because Q4_0 has optimized SYCL kernels
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
| Qwen3-1.7B | 1.7B | Q8_0 | INT4 |
| Llama 3.2 3B Instruct | 3B | Q4_K_M | INT4 |
| Qwen3-4B | 4B | Q4_K_M | INT4 |
| Mistral 7B Instruct v0.3 | 7B | Q4_K_M | INT4 |
| Qwen3-8B | 8B | Q4_K_M | INT4 |
| Gemma 2 9B IT | 9B | Q4_K_M | INT4 |
| Qwen3-14B | 14B | Q4_0 | INT4 |
| Phi-4 Reasoning | 14B | Q4_K_M | INT4 |
| DeepSeek-R1-Distill-Qwen-14B | 14B | Q4_0 | INT4 |
| Mistral Small 3.1 24B | 24B | Q4_K_M | INT4 |

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

- **24B models don't fit**: Mistral Small 3.1 24B (14GB Q4_K_M) OOMs on both backends. The Arc A770 16GB practical ceiling is ~14B parameters at 4-bit quantization.
- **Qwen 3.5** uses a novel Gated DeltaNet architecture whose kernels are not implemented for SYCL. It fails on all Intel GPU backends (llama.cpp, OpenVINO, Ollama).
- **Q4_0 vs Q4_K_M**: Q4_0 has hand-optimized SYCL dot-product kernels that are ~30% faster on Intel Arc, but most model publishers only provide Q4_K_M GGUFs.
- **Ollama** has no SYCL support and falls back to CPU-only on Intel GPUs (3-7 tok/s).
- **IPEX-LLM** (formerly BigDL-LLM) is archived/deprecated as of 2025.

## License

MIT

## Extended Benchmarks (OpenVINO OVMS)

The following benchmarks go beyond raw throughput to characterize the Arc A770's real-world inference behavior. All tests use OpenVINO OVMS with INT4 models.

### Power Efficiency

GPU power measured via sysfs `energy1_input` counter (cumulative microjoules). Idle power baseline: **39.8W**.

| Model | Params | tok/s | Watts | J/token | tok/Watt |
|-------|--------|:-----:|:-----:|:-------:|:--------:|
| Qwen3-1.7B | 1.7B | 64.8 | 94.0 | 1.451 | 0.690 |
| Qwen3-4B | 4B | 60.6 | 142.0 | 2.346 | 0.426 |
| Mistral-7B-Instruct-v0.3 | 7B | 56.7 | 175.0 | 3.088 | 0.324 |
| Qwen3-8B | 8B | 45.0 | 161.8 | 3.599 | 0.278 |
| gemma-2-9b-it | 9B | 32.2 | 152.2 | 4.728 | 0.211 |
| DeepSeek-R1-14B | 14B | 29.2 | 174.7 | 5.991 | 0.167 |
| Phi-4-reasoning | 14B | 31.0 | 184.4 | 5.961 | 0.168 |
| Qwen3-14B | 14B | 23.4 | 144.6 | 6.184 | 0.162 |

**Key findings:**
- Power scales roughly linearly with model size: 94W (1.7B) → 175W (14B)
- Qwen3-1.7B is the efficiency champion at **0.69 tok/Watt** — 4.3x more efficient than Qwen3-14B
- Phi-4-reasoning draws the most power (184W) despite being only marginally faster than DeepSeek-R1-14B
- The Arc A770 TDP is 225W; even the heaviest model only reaches ~82% of that

### Time to First Token (TTFT)

Streaming latency measured from HTTP request to first `delta.content` SSE chunk. Median of 3 runs per prompt.

| Model | Params | Median TTFT (ms) |
|-------|--------|:----------------:|
| Qwen3-4B | 4B | 71.0 |
| Qwen3-1.7B | 1.7B | 72.5 |
| Qwen3-8B | 8B | 79.6 |
| Mistral-7B-Instruct-v0.3 | 7B | 83.7 |
| Qwen3-14B | 14B | 118.8 |
| DeepSeek-R1-14B | 14B | 124.0 |
| gemma-2-9b-it | 9B | 125.6 |
| Phi-4-reasoning | 14B | 149.4 |

**Key findings:**
- All models deliver sub-150ms TTFT — excellent for interactive use
- Models ≤8B cluster in the 71-84ms range with negligible practical difference
- 14B models add ~40-70ms overhead but still feel instant
- TTFT is dominated by prompt processing, not model size per se (Qwen3-4B beats Qwen3-1.7B)

### Context Length Scaling

Generation speed (tok/s) at increasing context lengths. Fixed 128 output tokens to isolate the impact of context processing.

| Model | Params | 1K | 2K | 4K | 8K | 16K |
|-------|--------|:---:|:---:|:---:|:---:|:---:|
| Qwen3-1.7B | 1.7B | 55.7 | 57.0 | 48.6 | 44.5 | 20.9 |
| Qwen3-4B | 4B | 52.2 | 52.4 | 44.6 | 33.9 | 17.6 |
| Mistral-7B-Instruct-v0.3 | 7B | 48.2 | 49.0 | 40.5 | 28.8 | 16.4 |
| Qwen3-8B | 8B | 38.2 | 38.7 | 31.1 | 26.8 | 16.4 |
| gemma-2-9b-it | 9B | 28.7 | 28.9 | 25.5 | 20.4 | OOM |
| DeepSeek-R1-14B | 14B | 25.1 | 25.7 | 22.1 | 16.4 | 9.6 |
| Phi-4-reasoning | 14B | 27.2 | 27.5 | 23.6 | 17.7 | 10.4 |
| Qwen3-14B | 14B | 19.6 | 20.6 | 17.2 | 14.9 | 8.9 |

**Key findings:**
- All models handle 1K-2K context with no speed loss (KV cache fits comfortably in VRAM)
- Speed drops ~15-20% at 4K, ~40% at 8K, ~60% at 16K
- gemma-2-9b-it OOMs at 16K (9B model with less efficient KV cache compression)
- 14B models still generate at 8.9-10.4 tok/s with 16K context — usable for long-document tasks
- The 1K→2K *increase* in some models is within noise/warmup variance

### Reasoning Accuracy

20 math/logic problems with known numeric answers. Models instructed: "Think step by step. Put your final answer after ANSWER:". Auto-scored by extracting the answer and comparing to ground truth.

| Model | Params | Correct | Accuracy |
|-------|--------|:-------:|:--------:|
| Phi-4-reasoning | 14B | 20/20 | **100.0%** |
| Mistral-7B-Instruct-v0.3 | 7B | 13/20 | 65.0% |
| DeepSeek-R1-14B | 14B | 6/20 | 30.0% |
| Qwen3-8B | 8B | 3/20 | 15.0% |
| Qwen3-14B | 14B | 2/20 | 10.0% |
| Qwen3-4B | 4B | 2/20 | 10.0% |
| Qwen3-1.7B | 1.7B | 2/20 | 10.0% |

*Note: gemma-2-9b-it returned 400 errors (prompt format issue) and is excluded.*

**Key findings:**
- **Phi-4-reasoning is the standout** — perfect 20/20 score, purpose-built for mathematical reasoning
- Mistral-7B surprises at 65% — strong reasoning for a 7B generalist model
- **Qwen3 models used `/no_think` mode** (suppressed thinking tokens for fair output comparison). Their low scores likely reflect this constraint — Qwen3 relies heavily on its thinking chain for math. Full-think mode would score significantly higher
- DeepSeek-R1-14B's 30% is also likely output-truncation limited (all responses hit the token ceiling at ~17.5s)
- For math/reasoning workloads, Phi-4-reasoning is the clear choice on this hardware

### Quality Shootout (Response Length)

20 prompts across 5 categories (coding, summarization, reasoning, creative, extraction). Average completion tokens per category. Higher isn't always better — this measures verbosity and completeness.

| Model | Params | Coding | Summary | Reasoning | Creative | Extraction |
|-------|--------|:------:|:-------:|:---------:|:--------:|:----------:|
| DeepSeek-R1-14B | 14B | 992 | 507 | 537 | 636 | 485 |
| Phi-4-reasoning | 14B | 873 | 1023 | 893 | 957 | 886 |
| Qwen3-1.7B | 1.7B | 573 | 106 | 372 | 65 | 129 |
| gemma-2-9b-it | 9B | 550 | 114 | 126 | 86 | 172 |
| Mistral-7B | 7B | 479 | 200 | 200 | 202 | 158 |
| Qwen3-4B | 4B | 438 | 121 | 525 | 157 | 149 |
| Qwen3-8B | 8B | 373 | 105 | 376 | 111 | 179 |
| Qwen3-14B | 14B | 371 | 145 | 577 | 128 | 174 |

*Raw outputs saved to `results/quality/` for manual inspection.*

**Key findings:**
- Phi-4-reasoning and DeepSeek-R1-14B are the most verbose across all categories (CoT reasoning tokens inflate output)
- Qwen3 models with `/no_think` produce concise outputs — Qwen3-14B's 371 coding tokens vs DeepSeek's 992 represents a very different style
- gemma-2-9b-it produces the shortest creative responses (86 tokens avg) — it's very terse
- Response length alone doesn't indicate quality; the raw outputs need human evaluation

*Tested 2026-03-16 on Intel Arc A770 16GB. All models via OpenVINO OVMS with INT4 quantization.*
