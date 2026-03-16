#!/usr/bin/env python3
"""Benchmark: llama.cpp SYCL vs OpenVINO OVMS on Intel Arc GPUs.

Runs the same prompts against both backends for a fair comparison.
Each model is loaded in a fresh Docker container with GPU access.

Usage:
    python3 bench.py                        # run all models, both backends
    python3 bench.py --backend sycl         # SYCL only
    python3 bench.py --backend openvino     # OpenVINO only
    python3 bench.py --model qwen3-14b      # single model

Requirements:
    - Intel Arc GPU with i915/xe driver
    - Docker with GPU passthrough configured
    - GGUF models downloaded to GGUF_DIR
    - Internet access for OpenVINO (downloads models from HuggingFace on first run)
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

# === CONFIGURATION ===
# Adjust these paths for your system
GGUF_DIR = "/media/models/gguf"          # Host path to GGUF model files
OPENVINO_CACHE = "./openvino-models"     # Host path for OpenVINO model cache
GPU_RENDER = "/dev/dri/renderD128"       # GPU render device
GPU_CARD = "/dev/dri/card1"              # GPU card device (may be card0 on some systems)
RENDER_GID = "991"                       # GID of the 'render' group (check: getent group render)
VIDEO_GID = "44"                         # GID of the 'video' group

# === MODELS ===
# Add or remove models here. Each needs a SYCL config (GGUF file) and/or
# an OpenVINO config (HuggingFace model ID).
MODELS = {
    "qwen3-14b": {
        "sycl": {"gguf": "Qwen3-14B-Q4_0.gguf", "alias": "qwen3-14b"},
        "openvino": {"source": "OpenVINO/Qwen3-14B-int4-ov"},
    },
    "qwen3-8b": {
        "sycl": {"gguf": "Qwen3-8B-Q4_K_M.gguf", "alias": "qwen3-8b"},
        "openvino": {"source": "OpenVINO/Qwen3-8B-int4-ov"},
    },
    "gemma2-9b": {
        "sycl": {"gguf": "gemma-2-9b-it-Q4_K_M.gguf", "alias": "gemma2-9b"},
        "openvino": {"source": "OpenVINO/gemma-2-9b-it-int4-ov"},
    },
    "phi4-reasoning": {
        "sycl": {"gguf": "Phi-4-reasoning-Q4_K_M.gguf", "alias": "phi4-reasoning"},
        "openvino": {"source": "OpenVINO/Phi-4-reasoning-int4-ov"},
    },
    "mistral-7b": {
        "sycl": {"gguf": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "alias": "mistral-7b"},
        "openvino": {"source": "OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov"},
    },
    "deepseek-r1-14b": {
        "sycl": {"gguf": "DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf", "alias": "deepseek-r1-14b"},
        "openvino": {"source": "OpenVINO/DeepSeek-R1-Distill-Qwen-14B-int4-ov"},
    },
}

PROMPTS = [
    "Explain the difference between TCP and UDP in detail with examples.",
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes. Include comments.",
    "A farmer has 17 sheep. All but 9 run away. How many does he have left? Think step by step.",
    "Write a short story (200 words) about a robot learning to paint.",
]


def run(cmd, check=True):
    """Run a shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  CMD FAILED: {cmd}")
        print(f"  STDERR: {result.stderr[:500]}")
    return result


def wait_for_health(url, timeout=300):
    """Wait for an endpoint to become healthy."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def wait_for_chat_ready(base_url, model_name, timeout=600):
    """Wait for the chat completions endpoint to actually respond.

    OVMS health check returns 200 before the model is fully loaded.
    This sends a real request to verify readiness.
    """
    t0 = time.time()
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
    }).encode()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def bench_endpoint(base_url, model_name, label):
    """Run benchmark prompts against an OpenAI-compatible endpoint."""
    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"    [{i+1}/{len(PROMPTS)}] {prompt[:55]}...", end=" ", flush=True)
        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        elapsed = time.time() - t0

        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        timings = data.get("timings", {})
        server_tps = timings.get("predicted_per_second")

        results.append({
            "prompt_idx": i,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_s": round(elapsed, 2),
            "tok_per_sec_client": round(tok_per_sec, 1),
            "tok_per_sec_server": round(server_tps, 1) if server_tps else None,
        })

        tps = f"{server_tps:.1f}" if server_tps else f"{tok_per_sec:.1f}"
        print(f"{completion_tokens} tok in {elapsed:.1f}s = {tps} tok/s")

    if results:
        avg = sum(r["tok_per_sec_client"] for r in results) / len(results)
        server_vals = [r["tok_per_sec_server"] for r in results if r["tok_per_sec_server"]]
        avg_server = sum(server_vals) / len(server_vals) if server_vals else None
        best_tps = avg_server if avg_server else avg
        print(f"    => Avg: {best_tps:.1f} tok/s")
    return results


def start_sycl(model_cfg):
    """Start llama.cpp SYCL with a specific model."""
    gguf_file = model_cfg["gguf"]
    alias = model_cfg["alias"]
    run("docker stop llm-bench 2>/dev/null")
    run("docker rm llm-bench 2>/dev/null")

    run(f"""docker run -d --name llm-bench \
        --device {GPU_RENDER}:{GPU_RENDER} \
        --device {GPU_CARD}:{GPU_CARD} \
        -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
        -e ZES_ENABLE_SYSMAN=1 \
        -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
        -e GGML_SYCL_ENABLE_FLASH_ATTN=1 \
        -e LLAMA_ARG_MODEL=/models/{gguf_file} \
        -e LLAMA_ARG_ALIAS={alias} \
        -e LLAMA_ARG_HOST=0.0.0.0 \
        -e LLAMA_ARG_PORT=8000 \
        -e LLAMA_ARG_CTX_SIZE=8192 \
        -e LLAMA_ARG_N_GPU_LAYERS=999 \
        -e LLAMA_ARG_N_PARALLEL=1 \
        -v {GGUF_DIR}:/models:ro \
        --shm-size 16g \
        -p 8000:8000 \
        ghcr.io/ggml-org/llama.cpp:server-intel \
        --metrics --flash-attn 1""")

    print("    Waiting for SYCL to load...", end=" ", flush=True)
    if wait_for_health("http://localhost:8000/health", timeout=300):
        print("ready!")
        return True
    print("TIMEOUT!")
    return False


def stop_sycl():
    run("docker stop llm-bench 2>/dev/null")
    run("docker rm llm-bench 2>/dev/null")


def start_openvino(model_cfg):
    """Start OpenVINO OVMS with a specific model."""
    source = model_cfg["source"]
    run("docker stop openvino-bench 2>/dev/null")
    run("docker rm openvino-bench 2>/dev/null")

    run(f"""docker run -d --name openvino-bench \
        --device {GPU_RENDER}:{GPU_RENDER} \
        --device {GPU_CARD}:{GPU_CARD} \
        --group-add {VIDEO_GID} --group-add {RENDER_GID} \
        -v {OPENVINO_CACHE}:/models:rw \
        --shm-size 16g \
        -p 8001:8000 \
        openvino/model_server:latest-gpu \
        --source_model {source} \
        --model_repository_path /models \
        --task text_generation \
        --rest_port 8000 \
        --target_device GPU \
        --plugin_config '{{"KV_CACHE_PRECISION":"u8","DYNAMIC_QUANTIZATION_GROUP_SIZE":"32"}}' \
        --cache_size 4""")

    print("    Waiting for OpenVINO to load...", end=" ", flush=True)
    if not wait_for_health("http://localhost:8001/v3/models", timeout=600):
        print("TIMEOUT (health)!")
        return False
    print("server up...", end=" ", flush=True)
    if wait_for_chat_ready("http://localhost:8001/v3", source, timeout=300):
        print("ready!")
        return True
    print("TIMEOUT (chat)!")
    return False


def stop_openvino():
    run("docker stop openvino-bench 2>/dev/null")
    run("docker rm openvino-bench 2>/dev/null")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark llama.cpp SYCL vs OpenVINO OVMS on Intel Arc GPUs"
    )
    parser.add_argument("--backend", choices=["sycl", "openvino"], help="Run only one backend")
    parser.add_argument("--model", help="Run only one model (e.g. qwen3-14b)")
    parser.add_argument("--output", default="results.json", help="Output file for results")
    args = parser.parse_args()

    models_to_run = MODELS
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}. Available: {list(MODELS.keys())}")
            sys.exit(1)
        models_to_run = {args.model: MODELS[args.model]}

    backends = ["sycl", "openvino"]
    if args.backend:
        backends = [args.backend]

    all_results = {}

    print(f"\n{'#'*60}")
    print(f"# llama.cpp SYCL vs OpenVINO OVMS — Intel Arc GPU Benchmark")
    print(f"# Models: {len(models_to_run)}, Backends: {backends}")
    print(f"# Prompts: {len(PROMPTS)} x 512 max tokens each")
    print(f"{'#'*60}")

    for model_name, model_cfgs in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        for backend in backends:
            if backend not in model_cfgs:
                print(f"\n  [{backend}] No config for this backend, skipping.")
                continue

            cfg = model_cfgs[backend]
            label = f"{model_name} ({backend})"
            print(f"\n  [{backend.upper()}] {label}")
            print(f"  {'-'*50}")

            try:
                if backend == "sycl":
                    if not start_sycl(cfg):
                        print("    FAILED to start, skipping.")
                        stop_sycl()
                        continue
                    results = bench_endpoint("http://localhost:8000/v1", cfg["alias"], label)
                    stop_sycl()
                else:
                    if not start_openvino(cfg):
                        print("    FAILED to start, skipping.")
                        stop_openvino()
                        continue
                    results = bench_endpoint("http://localhost:8001/v3", cfg["source"], label)
                    stop_openvino()

                all_results[f"{model_name}_{backend}"] = {
                    "model": model_name,
                    "backend": backend,
                    "config": cfg,
                    "results": results,
                }

                # Brief pause between backends to let GPU cool
                time.sleep(10)

            except Exception as e:
                print(f"    EXCEPTION: {e}")
                stop_sycl()
                stop_openvino()

    # Summary table
    print(f"\n\n{'#'*60}")
    print(f"# SUMMARY")
    print(f"{'#'*60}")
    print(f"\n{'Model':<25} {'Backend':<12} {'Avg tok/s':<12} {'Total time':<12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    for key, data in all_results.items():
        results = data["results"]
        if not results:
            continue
        server_vals = [r["tok_per_sec_server"] for r in results if r.get("tok_per_sec_server")]
        client_vals = [r["tok_per_sec_client"] for r in results]
        avg = sum(server_vals) / len(server_vals) if server_vals else sum(client_vals) / len(client_vals)
        total = sum(r["elapsed_s"] for r in results)
        print(f"{data['model']:<25} {data['backend']:<12} {avg:<12.1f} {total:<12.1f}s")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
