#!/usr/bin/env python3
"""Power efficiency benchmark for Intel Arc A770 with OpenVINO OVMS.

Measures watts, joules/token, and tokens/watt for each cached model
by reading the GPU energy counter before/after inference.

Uses the live OVMS instance via the model switcher API — no isolated
containers needed. Switches model, waits for readiness, benchmarks,
then restores the original model.
"""

import json
import sys
import time
import urllib.request
import urllib.error

from gpu_monitor import PowerSample, read_energy_uj, read_gpu_temp

SWITCHER_URL = "http://localhost:3005"
LLM_URL = "http://localhost:8000"

PROMPTS = [
    "Explain the difference between TCP and UDP in detail with examples.",
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes. Include comments.",
    "A farmer has 17 sheep. All but 9 run away. How many does he have left? Think step by step.",
    "Write a short story (200 words) about a robot learning to paint.",
]

MAX_TOKENS = 512


def api_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def get_status():
    return api_get(f"{SWITCHER_URL}/api/status")


def switch_model(model_id):
    print(f"  Switching to {model_id}...")
    api_post(f"{SWITCHER_URL}/api/switch", {"model": model_id})
    # Wait for switch to complete and model to be ready
    for _ in range(120):
        time.sleep(5)
        status = get_status()
        if not status["switching"] and status["model_ready"]:
            print(f"  Model ready.")
            return
    raise TimeoutError(f"Model {model_id} did not become ready in 600s")


def get_models():
    return api_get(f"{SWITCHER_URL}/api/models")


def bench_model(model_id):
    """Benchmark a single model, return per-prompt results with power data."""
    results = []

    for i, prompt in enumerate(PROMPTS):
        # Small pause between prompts for cleaner energy readings
        time.sleep(2)

        with PowerSample() as pwr:
            resp = api_post(f"{LLM_URL}/v3/chat/completions", {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.7,
            })

        usage = resp.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        tok_per_sec = completion_tokens / pwr.elapsed_s if pwr.elapsed_s > 0 else 0
        joules_per_token = pwr.energy_j / completion_tokens if completion_tokens > 0 else 0
        tokens_per_watt = tok_per_sec / pwr.avg_watts if pwr.avg_watts > 0 else 0

        result = {
            "prompt_idx": i,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_s": round(pwr.elapsed_s, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "avg_watts": round(pwr.avg_watts, 1),
            "joules_per_token": round(joules_per_token, 3),
            "tokens_per_watt": round(tokens_per_watt, 3),
            "temp_start_c": round(pwr.temp_start, 1),
            "temp_end_c": round(pwr.temp_end, 1),
        }
        results.append(result)

        print(f"    Prompt {i+1}: {tok_per_sec:.1f} tok/s | {pwr.avg_watts:.1f}W | "
              f"{joules_per_token:.3f} J/tok | {pwr.temp_end:.0f}°C")

    return results


def main():
    models = get_models()
    original_status = get_status()
    original_model = original_status["current_model"]

    print(f"Power Efficiency Benchmark — Intel Arc A770 + OpenVINO OVMS")
    print(f"Models: {len(models)} | Prompts: {len(PROMPTS)} | Max tokens: {MAX_TOKENS}")
    print(f"Current model: {original_model}")
    print()

    # Read idle power for baseline
    print("Reading idle GPU power (5s)...")
    e1 = read_energy_uj()
    time.sleep(5)
    e2 = read_energy_uj()
    idle_watts = (e2 - e1) / 5_000_000.0
    print(f"Idle power: {idle_watts:.1f}W | Temp: {read_gpu_temp():.0f}°C")
    print()

    all_results = {"idle_watts": round(idle_watts, 1), "models": {}}

    for model in models:
        model_id = model["id"]
        short = model["name"].replace("-int4-ov", "")
        print(f"[{short}] ({model['params']})")

        try:
            switch_model(model_id)
            # 10s GPU cooldown after model load
            time.sleep(10)
            results = bench_model(model_id)

            avg_tps = sum(r["tok_per_sec"] for r in results) / len(results)
            avg_watts = sum(r["avg_watts"] for r in results) / len(results)
            avg_jpt = sum(r["joules_per_token"] for r in results) / len(results)
            avg_tpw = sum(r["tokens_per_watt"] for r in results) / len(results)

            all_results["models"][model_id] = {
                "name": short,
                "params": model["params"],
                "prompts": results,
                "avg_tok_per_sec": round(avg_tps, 1),
                "avg_watts": round(avg_watts, 1),
                "avg_joules_per_token": round(avg_jpt, 3),
                "avg_tokens_per_watt": round(avg_tpw, 3),
            }
            print(f"  AVG: {avg_tps:.1f} tok/s | {avg_watts:.1f}W | {avg_jpt:.3f} J/tok | {avg_tpw:.3f} tok/W")
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results["models"][model_id] = {"error": str(e)}
            print()

    # Restore original model
    print(f"Restoring original model: {original_model}")
    switch_model(original_model)

    # Save results
    out_file = f"results/power/power_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print summary table
    print("\n## Power Efficiency Summary")
    print(f"| Model | Params | tok/s | Watts | J/token | tok/Watt |")
    print(f"|-------|--------|:-----:|:-----:|:-------:|:--------:|")
    for mid, data in all_results["models"].items():
        if "error" in data:
            print(f"| {data.get('name', mid)} | — | — | — | — | — |")
        else:
            print(f"| {data['name']} | {data['params']} | {data['avg_tok_per_sec']} | "
                  f"{data['avg_watts']} | {data['avg_joules_per_token']} | {data['avg_tokens_per_watt']} |")


if __name__ == "__main__":
    main()
