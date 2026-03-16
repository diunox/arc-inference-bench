#!/usr/bin/env python3
"""Context length scaling benchmark for Intel Arc A770 with OpenVINO OVMS.

Measures generation tok/s at different input context lengths (1K-16K tokens).
Tests all 8 cached models to find the throughput degradation curve and
VRAM ceiling per model size.
"""

import json
import time
import urllib.request
import urllib.error

SWITCHER_URL = "http://localhost:3005"
LLM_URL = "http://localhost:8000"

# Target context lengths in approximate tokens (1 token ≈ 4 chars in English)
CONTEXT_TARGETS = [1000, 2000, 4000, 8000, 16000]

# Fixed small output to isolate context processing effect
OUTPUT_TOKENS = 128

# Filler text — real English paragraph repeated to reach target length
FILLER = (
    "The development of modern computing has been shaped by decades of innovation "
    "in both hardware and software. From the earliest vacuum tube machines to today's "
    "multi-core processors and specialized accelerators, each generation has brought "
    "exponential increases in capability while reducing cost and power consumption. "
    "Operating systems evolved from simple batch processors to sophisticated "
    "multitasking environments, enabling users to run complex applications "
    "simultaneously. Networking protocols like TCP/IP transformed isolated machines "
    "into a global interconnected system, giving rise to the internet and cloud "
    "computing. Database management systems progressed from flat files to relational "
    "models and eventually to distributed NoSQL solutions capable of handling "
    "petabytes of data. Programming languages diversified to address different "
    "domains, from systems programming with C to web development with JavaScript "
    "and data science with Python. "
)


def api_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_post(url, data, timeout=300):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def generate_prompt(target_tokens):
    """Generate a prompt of approximately target_tokens length."""
    target_chars = target_tokens * 4
    filler_len = len(FILLER)
    repeats = (target_chars // filler_len) + 1
    text = (FILLER * repeats)[:target_chars]
    return f"Summarize the following text in 3 bullet points:\n\n{text}\n\nProvide a concise summary."


def bench_context_length(model_id, target_tokens):
    """Benchmark a single context length, return results or None on failure."""
    prompt = generate_prompt(target_tokens)

    try:
        t_start = time.monotonic()
        resp = api_post(f"{LLM_URL}/v3/chat/completions", {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.7,
        }, timeout=600)
        t_end = time.monotonic()

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        elapsed = t_end - t_start
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

        return {
            "target_tokens": target_tokens,
            "actual_prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_s": round(elapsed, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "status": "ok",
        }

    except urllib.error.URLError as e:
        return {"target_tokens": target_tokens, "status": "error", "error": str(e)}
    except TimeoutError:
        return {"target_tokens": target_tokens, "status": "timeout"}
    except Exception as e:
        return {"target_tokens": target_tokens, "status": "error", "error": str(e)}


def main():
    models = api_get(f"{SWITCHER_URL}/api/models")
    original = api_get(f"{SWITCHER_URL}/api/status")["current_model"]

    print(f"Context Length Scaling Benchmark — Intel Arc A770 + OpenVINO OVMS")
    print(f"Models: {len(models)} | Context targets: {CONTEXT_TARGETS} | Output: {OUTPUT_TOKENS} tokens")
    print()

    all_results = {"output_tokens": OUTPUT_TOKENS, "models": {}}

    for model in models:
        model_id = model["id"]
        short = model["name"].replace("-int4-ov", "")
        print(f"[{short}] ({model['params']})")

        try:
            switch_model(model_id)
            time.sleep(10)

            model_results = []
            for target in CONTEXT_TARGETS:
                time.sleep(3)
                print(f"    {target:>5} tokens: ", end="", flush=True)
                r = bench_context_length(model_id, target)
                model_results.append(r)

                if r["status"] == "ok":
                    print(f"{r['tok_per_sec']:.1f} tok/s (actual input: {r['actual_prompt_tokens']} tokens)")
                else:
                    print(f"{r['status'].upper()}: {r.get('error', 'container crashed or timed out')}")
                    # If this length failed, skip larger ones
                    for remaining in CONTEXT_TARGETS[CONTEXT_TARGETS.index(target)+1:]:
                        model_results.append({
                            "target_tokens": remaining,
                            "status": "skipped",
                        })
                        print(f"    {remaining:>5} tokens: SKIPPED (previous length failed)")
                    break

            all_results["models"][model_id] = {
                "name": short,
                "params": model["params"],
                "results": model_results,
            }
            print()

        except Exception as e:
            print(f"  ERROR: {e}\n")
            all_results["models"][model_id] = {"name": short, "error": str(e)}

    # Restore
    print(f"Restoring original model: {original}")
    switch_model(original)

    out_file = f"results/context/context_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary matrix
    print("\n## Context Scaling Summary (tok/s)")
    header = "| Model | Params | " + " | ".join(f"{t//1000}K" for t in CONTEXT_TARGETS) + " |"
    sep = "|-------|--------|" + "|".join(":---:" for _ in CONTEXT_TARGETS) + "|"
    print(header)
    print(sep)
    for mid, data in all_results["models"].items():
        if "error" in data:
            row = f"| {data.get('name', mid)} | — |" + " — |" * len(CONTEXT_TARGETS)
        else:
            cells = []
            for r in data["results"]:
                if r["status"] == "ok":
                    cells.append(str(r["tok_per_sec"]))
                else:
                    cells.append(r["status"].upper())
            row = f"| {data['name']} | {data['params']} | " + " | ".join(cells) + " |"
        print(row)


if __name__ == "__main__":
    main()
