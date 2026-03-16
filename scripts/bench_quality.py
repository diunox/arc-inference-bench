#!/usr/bin/env python3
"""Task-specific quality shootout across all 8 OpenVINO models.

Runs 20 prompts (4 per category: coding, summarization, reasoning,
creative, extraction) against each model via the model switcher.
Saves raw outputs for review.
"""

import json
import os
import time
import urllib.request

SWITCHER_URL = "http://localhost:3005"
LLM_URL = "http://localhost:8000"
PROMPTS_FILE = "prompts/quality_prompts.json"
RESULTS_DIR = "results/quality"
MAX_TOKENS = 1024


def api_get(url):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
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
    raise TimeoutError(f"Model {model_id} did not become ready")


def run_prompt(model_id, prompt):
    t_start = time.monotonic()
    resp = api_post(f"{LLM_URL}/v3/chat/completions", {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt + "\n/no_think"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    })
    elapsed = time.monotonic() - t_start
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    return {
        "content": content,
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed_s": round(elapsed, 2),
    }


def main():
    with open(PROMPTS_FILE) as f:
        prompts_by_category = json.load(f)

    models = api_get(f"{SWITCHER_URL}/api/models")
    original = api_get(f"{SWITCHER_URL}/api/status")["current_model"]

    total_prompts = sum(len(v) for v in prompts_by_category.values())
    print(f"Quality Shootout — {len(models)} models × {total_prompts} prompts")
    print()

    all_results = {}

    for model in models:
        model_id = model["id"]
        short = model["name"].replace("-int4-ov", "")
        print(f"[{short}] ({model['params']})")

        try:
            switch_model(model_id)
            time.sleep(5)

            model_dir = os.path.join(RESULTS_DIR, short)
            os.makedirs(model_dir, exist_ok=True)

            model_data = {}
            for category, category_prompts in prompts_by_category.items():
                category_results = []
                for idx, prompt in enumerate(category_prompts):
                    time.sleep(1)
                    print(f"    {category}[{idx}]: ", end="", flush=True)
                    result = run_prompt(model_id, prompt)
                    category_results.append(result)

                    # Save raw output
                    out_path = os.path.join(model_dir, f"{category}_{idx}.txt")
                    with open(out_path, "w") as f:
                        f.write(result["content"])

                    preview = result["content"][:60].replace("\n", " ")
                    print(f"{result['completion_tokens']} tok, {result['elapsed_s']}s — {preview}...")

                model_data[category] = category_results

            all_results[model_id] = {"name": short, "params": model["params"], "categories": model_data}
            print()

        except Exception as e:
            print(f"  ERROR: {e}\n")
            all_results[model_id] = {"error": str(e)}

    # Restore
    print(f"Restoring original model: {original}")
    switch_model(original)

    out_file = f"results/quality/quality_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary: average tokens per category per model
    print("\n## Quality Shootout — Response Lengths (avg tokens)")
    categories = list(prompts_by_category.keys())
    header = "| Model | Params | " + " | ".join(categories) + " |"
    sep = "|-------|--------|" + "|".join(":---:" for _ in categories) + "|"
    print(header)
    print(sep)
    for mid, data in all_results.items():
        if "error" in data:
            continue
        cells = []
        for cat in categories:
            avg = sum(r["completion_tokens"] for r in data["categories"][cat]) / len(data["categories"][cat])
            cells.append(str(int(avg)))
        print(f"| {data['name']} | {data['params']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
