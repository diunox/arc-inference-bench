#!/usr/bin/env python3
"""Reasoning benchmark — 20 math/logic problems with auto-scoring.

Tests all 8 models on GSM8K-style problems. Extracts answers from
model output and compares to known correct answers.
"""

import json
import re
import time
import urllib.request

SWITCHER_URL = "http://localhost:3005"
LLM_URL = "http://localhost:8000"
PROBLEMS_FILE = "prompts/reasoning_problems.json"
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "Solve this math problem step by step. After your solution, "
    "write your final numerical answer on its own line in the format: ANSWER: <number>"
)


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
            return
    raise TimeoutError(f"Model {model_id} did not become ready")


def strip_think_tags(text):
    """Remove <think>...</think> blocks from reasoning model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_answer(text):
    """Extract the answer after 'ANSWER:' in the model output."""
    text = strip_think_tags(text)
    # Look for ANSWER: pattern
    match = re.search(r"ANSWER:\s*\$?([\d./]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: look for the last number in the text
    numbers = re.findall(r"[\d.]+", text)
    return numbers[-1] if numbers else None


def normalize_answer(ans):
    """Normalize answer for comparison (handle fractions, decimals, etc.)."""
    if ans is None:
        return None
    ans = ans.strip().rstrip(".")
    # Handle fractions
    if "/" in ans:
        try:
            num, den = ans.split("/")
            return round(float(num) / float(den), 6)
        except (ValueError, ZeroDivisionError):
            return ans
    try:
        return round(float(ans), 6)
    except ValueError:
        return ans


def score(extracted, correct):
    """Compare extracted answer to correct answer."""
    e = normalize_answer(extracted)
    c = normalize_answer(correct)
    if e is None:
        return False
    if isinstance(e, float) and isinstance(c, float):
        return abs(e - c) < 0.01
    return str(e) == str(c)


def main():
    with open(PROBLEMS_FILE) as f:
        problems = json.load(f)

    models = api_get(f"{SWITCHER_URL}/api/models")
    original = api_get(f"{SWITCHER_URL}/api/status")["current_model"]

    print(f"Reasoning Benchmark — {len(models)} models × {len(problems)} problems")
    print()

    all_results = {}

    for model in models:
        model_id = model["id"]
        short = model["name"].replace("-int4-ov", "")
        print(f"[{short}] ({model['params']})")

        try:
            switch_model(model_id)
            time.sleep(5)

            correct_count = 0
            problem_results = []

            for i, prob in enumerate(problems):
                time.sleep(1)
                t_start = time.monotonic()
                resp = api_post(f"{LLM_URL}/v3/chat/completions", {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prob["question"]},
                    ],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.1,  # Low temp for deterministic math
                })
                elapsed = time.monotonic() - t_start

                content = resp["choices"][0]["message"]["content"]
                extracted = extract_answer(content)
                is_correct = score(extracted, prob["answer"])
                if is_correct:
                    correct_count += 1

                status = "✓" if is_correct else "✗"
                print(f"    Q{i+1:2d}: {status}  extracted={extracted}  expected={prob['answer']}  ({elapsed:.1f}s)")

                problem_results.append({
                    "question_idx": i,
                    "correct_answer": prob["answer"],
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "elapsed_s": round(elapsed, 2),
                    "full_response": content,
                })

            accuracy = correct_count / len(problems) * 100
            all_results[model_id] = {
                "name": short,
                "params": model["params"],
                "correct": correct_count,
                "total": len(problems),
                "accuracy_pct": round(accuracy, 1),
                "problems": problem_results,
            }
            print(f"  Score: {correct_count}/{len(problems)} ({accuracy:.1f}%)\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            all_results[model_id] = {"error": str(e)}

    # Restore
    print(f"Restoring original model: {original}")
    switch_model(original)

    out_file = f"results/reasoning/reasoning_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary
    print("\n## Reasoning Benchmark Summary")
    print("| Model | Params | Correct | Accuracy |")
    print("|-------|--------|:-------:|:--------:|")
    for mid, data in all_results.items():
        if "error" in data:
            continue
        print(f"| {data['name']} | {data['params']} | {data['correct']}/{data['total']} | {data['accuracy_pct']}% |")


if __name__ == "__main__":
    main()
