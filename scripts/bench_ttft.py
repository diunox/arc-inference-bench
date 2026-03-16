#!/usr/bin/env python3
"""Time-to-First-Token benchmark for Intel Arc A770 with OpenVINO OVMS.

Measures TTFT via streaming SSE responses. Runs 3 iterations per prompt
and takes the median for stability.
"""

import http.client
import json
import statistics
import time
import urllib.request

SWITCHER_URL = "http://localhost:3005"
LLM_HOST = "localhost"
LLM_PORT = 8000

PROMPTS = [
    "Explain the difference between TCP and UDP in detail with examples.",
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes. Include comments.",
    "A farmer has 17 sheep. All but 9 run away. How many does he have left? Think step by step.",
    "Write a short story (200 words) about a robot learning to paint.",
]

MAX_TOKENS = 256
ITERATIONS = 3


def api_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def switch_model(model_id):
    print(f"  Switching to {model_id}...")
    api_post(f"{SWITCHER_URL}/api/switch", {"model": model_id})
    for _ in range(120):
        time.sleep(5)
        status = api_get(f"{SWITCHER_URL}/api/status")
        if not status["switching"] and status["model_ready"]:
            print(f"  Model ready.")
            return
    raise TimeoutError(f"Model {model_id} did not become ready in 600s")


def measure_ttft(model_id, prompt):
    """Send a streaming request and measure time to first content token."""
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    conn = http.client.HTTPConnection(LLM_HOST, LLM_PORT, timeout=120)
    conn.request("POST", "/v3/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})

    t_start = time.monotonic()
    resp = conn.getresponse()
    t_headers = time.monotonic()

    ttft = None
    total_tokens = 0
    full_content = []

    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        if buf.endswith(b"\n\n"):
            line = buf.decode("utf-8", errors="replace").strip()
            buf = b""
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
                delta = event.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.monotonic() - t_start
                    total_tokens += 1
                    full_content.append(content)
            except json.JSONDecodeError:
                pass

    t_end = time.monotonic()
    conn.close()

    total_time = t_end - t_start
    # Token count from SSE chunks is approximate — each chunk may have multiple tokens
    # Use the full content length as a rough proxy (1 chunk ≈ 1-3 tokens)
    gen_time = total_time - (ttft or total_time)
    content_str = "".join(full_content)

    return {
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "total_time_s": round(total_time, 2),
        "chunks": total_tokens,
        "content_length": len(content_str),
    }


def bench_model_ttft(model_id):
    """Benchmark TTFT for a model across all prompts with multiple iterations."""
    results = []
    for i, prompt in enumerate(PROMPTS):
        ttft_samples = []
        for it in range(ITERATIONS):
            time.sleep(1)
            r = measure_ttft(model_id, prompt)
            if r["ttft_ms"] is not None:
                ttft_samples.append(r["ttft_ms"])
            print(f"    Prompt {i+1} iter {it+1}: TTFT={r['ttft_ms']}ms | total={r['total_time_s']}s")

        median_ttft = statistics.median(ttft_samples) if ttft_samples else None
        results.append({
            "prompt_idx": i,
            "ttft_samples_ms": ttft_samples,
            "median_ttft_ms": round(median_ttft, 1) if median_ttft else None,
        })

    return results


def main():
    models = api_get(f"{SWITCHER_URL}/api/models")
    original = api_get(f"{SWITCHER_URL}/api/status")["current_model"]

    print(f"TTFT Benchmark — Intel Arc A770 + OpenVINO OVMS")
    print(f"Models: {len(models)} | Prompts: {len(PROMPTS)} | Iterations: {ITERATIONS}")
    print()

    all_results = {"models": {}}

    for model in models:
        model_id = model["id"]
        short = model["name"].replace("-int4-ov", "")
        print(f"[{short}] ({model['params']})")

        try:
            switch_model(model_id)
            time.sleep(5)
            results = bench_model_ttft(model_id)

            median_ttfts = [r["median_ttft_ms"] for r in results if r["median_ttft_ms"] is not None]
            overall_median = statistics.median(median_ttfts) if median_ttfts else None

            all_results["models"][model_id] = {
                "name": short,
                "params": model["params"],
                "prompts": results,
                "overall_median_ttft_ms": round(overall_median, 1) if overall_median else None,
            }
            print(f"  Overall median TTFT: {overall_median:.1f}ms" if overall_median else "  No valid TTFT")
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results["models"][model_id] = {"error": str(e)}
            print()

    # Restore
    print(f"Restoring original model: {original}")
    switch_model(original)

    out_file = f"results/ttft/ttft_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary table
    print("\n## TTFT Summary")
    print(f"| Model | Params | Median TTFT (ms) |")
    print(f"|-------|--------|:----------------:|")
    for mid, data in all_results["models"].items():
        if "error" in data:
            print(f"| {mid} | — | ERROR |")
        else:
            ttft = data["overall_median_ttft_ms"]
            print(f"| {data['name']} | {data['params']} | {ttft} |")


if __name__ == "__main__":
    main()
