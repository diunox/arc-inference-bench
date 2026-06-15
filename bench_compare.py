#!/usr/bin/env python3
"""Comprehensive cross-model benchmark for OVMS on Janus.

Usage:
    python3 bench_compare.py MODEL_ID [MODEL_ID ...]

For each model:
  Tier 1 (throughput / latency)
    - Output-length sweep (128, 256, 512, 1024, 2048 tokens)
    - TTFT via streaming (3 prompts x 3 iters)
    - Long-context throughput (2k, 8k, 16k, 32k context lengths)
    - Prompt-processing speed (4k, 8k input)
  Tier 2 (verifiable correctness)
    - GSM8K math (20 problems with numeric answers)
    - Multi-step arithmetic (10 chained calculations)
    - Logic puzzles (10 binary/short-answer)
    - Code generation (10 HumanEval-style with pytest assertions)
    - Instruction following (8 format-verifiable)
    - Needle-in-haystack (5 depths x 3 needles = 15 trials)
    - RAG comprehension (8 Q&A on a long passage)
    - Multi-hop reasoning (5 problems)
    - Knowledge accuracy (10 facts, exact-match)
    - Anti-hallucination (5 unanswerable; should refuse)
  Tier 4 (multimodal, auto-skipped on text-only models)
    - Color identification (3 generated images)
    - Shape identification (4 generated images incl. star)
    - OCR text recognition (3 generated text images)

Flags:
  --only=name1,name2  run only the named tests
  --skip=name1,name2  skip the named tests

Saves results to results/compare/<short_name>_<unix_ts>.json
Companion: compare_runs.py results/compare/*.json
"""

import base64
import http.client
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

SWITCHER_URL = "http://localhost:3005"
LLM_HOST = "localhost"
LLM_PORT = 8000
LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}"

RESULTS_DIR = "results/compare"
SWITCH_TIMEOUT_S = 600

# ============================================================================
# HTTP helpers
# ============================================================================

def api_get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read())

def api_post(url, data, timeout=600):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def switch_to(model_id):
    print(f"  switching to {model_id} ...")
    t0 = time.time()
    api_post(f"{SWITCHER_URL}/api/switch", {"model": model_id})
    deadline = time.time() + SWITCH_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(5)
        st = api_get(f"{SWITCHER_URL}/api/status")
        if not st.get("switching") and st.get("model_ready"):
            time.sleep(3)
            print(f"  model ready in {int(time.time()-t0)}s")
            return
    raise TimeoutError(f"{model_id} not ready in {SWITCH_TIMEOUT_S}s")

def chat(model_id, prompt, max_tokens, temperature=0.0, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    t0 = time.monotonic()
    resp = api_post(LLM_URL + "/v3/chat/completions", {
        "model": model_id, "messages": msgs,
        "max_tokens": max_tokens, "temperature": temperature,
    }, timeout=900)
    return {
        "content": resp.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "usage": resp.get("usage", {}),
        "elapsed_s": time.monotonic() - t0,
    }

# ============================================================================
# Tier 1: Throughput / latency
# ============================================================================

THROUGHPUT_PROMPT = (
    "Write a detailed essay about the history and impact of personal computing, "
    "covering at least four major eras with specific examples. Be thorough."
)
THROUGHPUT_LENGTHS = [128, 256, 512, 1024, 2048]
THROUGHPUT_ITERS = 3

def bench_throughput(model_id):
    print("  [tput]")
    results = {}
    for length in THROUGHPUT_LENGTHS:
        rates = []
        for _ in range(THROUGHPUT_ITERS):
            r = chat(model_id, THROUGHPUT_PROMPT, length, temperature=0.7)
            ct = r["usage"].get("completion_tokens", 0)
            rate = ct / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
            rates.append(rate)
        med = statistics.median(rates)
        results[str(length)] = {
            "median_tok_s": round(med, 2),
            "samples": [round(x, 2) for x in rates],
        }
        print(f"    out={length:4} -> {med:.2f} tok/s")
    return results

TTFT_PROMPTS = [
    "Explain the difference between TCP and UDP with examples.",
    "Write a Python function for the Sieve of Eratosthenes with comments.",
    "Describe the carbon cycle in detail.",
]
TTFT_ITERS = 3
TTFT_MAX_TOKENS = 128

def bench_ttft(model_id):
    print("  [ttft]")
    out = []
    for i, p in enumerate(TTFT_PROMPTS):
        samples = []
        for _ in range(TTFT_ITERS):
            t = _measure_ttft(model_id, p, TTFT_MAX_TOKENS)
            if t is not None:
                samples.append(t)
            time.sleep(0.5)
        med = statistics.median(samples) if samples else None
        out.append({"prompt_idx": i, "samples_ms": [round(s, 1) for s in samples],
                    "median_ms": round(med, 1) if med else None})
        print(f"    prompt {i+1}: median {med:.1f} ms" if med else f"    prompt {i+1}: failed")
    valid = [r["median_ms"] for r in out if r["median_ms"]]
    overall = statistics.median(valid) if valid else None
    return {"per_prompt": out, "overall_median_ms": round(overall, 1) if overall else None}

def _measure_ttft(model_id, prompt, max_tokens):
    body = json.dumps({
        "model": model_id, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.7, "stream": True,
    }).encode()
    conn = http.client.HTTPConnection(LLM_HOST, LLM_PORT, timeout=180)
    conn.request("POST", "/v3/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    t_start = time.monotonic()
    resp = conn.getresponse()
    ttft = None
    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk: break
        buf += chunk
        if buf.endswith(b"\n\n"):
            line = buf.decode("utf-8", errors="replace").strip()
            buf = b""
            if not line.startswith("data: "): continue
            payload = line[6:]
            if payload == "[DONE]": break
            try:
                ev = json.loads(payload)
                delta = ev.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") and ttft is None:
                    ttft = (time.monotonic() - t_start) * 1000
            except json.JSONDecodeError:
                pass
    conn.close()
    return ttft

LONG_CONTEXT_TARGETS = [2000, 8000, 16000, 32000]
LONG_CONTEXT_OUTPUT_TOKENS = 128
FILLER = (
    "The history of computing is rich and varied. From the earliest mechanical "
    "calculators of the 17th century, through Babbage's analytical engine and "
    "Lovelace's first algorithm, to the vacuum-tube ENIAC and Turing's universal "
    "machine, the field has steadily expanded. The post-war decades saw transistors "
    "replace tubes, then integrated circuits replace transistors, in a relentless "
    "march toward miniaturization captured by Moore's Law. Operating systems grew "
    "from punch-card batch jobs to multi-user timesharing to graphical interfaces "
    "to mobile platforms. Networking went from circuit-switched to packet-switched, "
    "from ARPANET to the global Internet. Storage exploded from kilobytes on tape "
    "to terabytes on solid-state drives. "
)

def bench_long_context(model_id):
    print("  [long_ctx]")
    results = {}
    for tgt in LONG_CONTEXT_TARGETS:
        needed = max(1, int(tgt * 4 / len(FILLER)) + 1)
        prefix = FILLER * needed
        prompt = (f"Below is a long passage. After reading it, write a one-paragraph "
                  f"summary of the key themes.\n\n{prefix}\n\nSummary:")
        try:
            r = chat(model_id, prompt, LONG_CONTEXT_OUTPUT_TOKENS, temperature=0.3)
            pt = r["usage"].get("prompt_tokens", 0)
            ct = r["usage"].get("completion_tokens", 0)
            rate = ct / r["elapsed_s"] if r["elapsed_s"] > 0 else 0
            results[str(tgt)] = {
                "actual_prompt_tokens": pt, "completion_tokens": ct,
                "elapsed_s": round(r["elapsed_s"], 2), "gen_tok_s": round(rate, 2),
            }
            print(f"    ctx~{tgt:>5} (actual {pt}): {rate:.2f} tok/s, {r['elapsed_s']:.1f}s total")
        except Exception as e:
            results[str(tgt)] = {"error": str(e)[:200]}
            print(f"    ctx~{tgt:>5}: ERROR {e}")
    return results

PROMPT_PROCESSING_TARGETS = [4000, 8000]

def bench_prompt_processing(model_id):
    """Measure prompt-processing throughput by minimizing output and looking at TTFT."""
    print("  [prompt_proc]")
    results = {}
    for tgt in PROMPT_PROCESSING_TARGETS:
        needed = max(1, int(tgt * 4 / len(FILLER)) + 1)
        prefix = FILLER * needed
        prompt = prefix + "\n\nReply with just the word: ok"
        # Use TTFT as a proxy for prompt-processing time; first-token latency mostly = prompt processing
        ttft = _measure_ttft(model_id, prompt, max_tokens=8)
        # Need the actual prompt tokens, do a chat call to get usage
        r = chat(model_id, prompt, 4, temperature=0.0)
        pt = r["usage"].get("prompt_tokens", 0)
        proc_rate = pt / (ttft / 1000) if ttft and ttft > 0 else None
        results[str(tgt)] = {
            "actual_prompt_tokens": pt,
            "ttft_ms": round(ttft, 1) if ttft else None,
            "prompt_tok_per_s": round(proc_rate, 1) if proc_rate else None,
        }
        print(f"    pp~{tgt:>5} (actual {pt}): {proc_rate:.1f} tok/s ingest"
              if proc_rate else f"    pp~{tgt:>5}: failed")
    return results

# ============================================================================
# Tier 2: Verifiable correctness
# ============================================================================

GSM8K_PROBLEMS = [
    {"q": "A farmer has 17 sheep. All but 9 run away. How many does he have left?", "a": 9},
    {"q": "Janet pays $40 per hour for 3 hours of math tutoring per week. How much does she pay in 6 weeks?", "a": 720},
    {"q": "A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. How many miles total?", "a": 270},
    {"q": "A store sells apples for $1.20 each. You buy 12 apples and pay with a $20 bill. What is the change in dollars (whole number)?", "a": 6},
    {"q": "Tom is twice as old as his sister. In 5 years, the sum of their ages will be 40. How old is Tom now?", "a": 20},
    {"q": "A rectangle has length 15 cm and perimeter 50 cm. What is its area in square cm?", "a": 150},
    {"q": "If 8 workers can paint a fence in 6 hours, how many hours would 12 workers take?", "a": 4},
    {"q": "A book costs $50. It's discounted 20%, then a member gets an additional 10% off the discounted price. What is the final price in whole dollars?", "a": 36},
    {"q": "Sara has 3 times as many marbles as Tim. Together they have 48 marbles. How many does Sara have?", "a": 36},
    {"q": "A car uses 8 gallons of gas to travel 240 miles. What is the miles per gallon?", "a": 30},
    {"q": "An aquarium has 60 fish. 1/4 are goldfish, 1/3 are angelfish, the rest are guppies. How many guppies?", "a": 25},
    {"q": "A pizza is cut into 8 slices. 3 people each eat 2 slices. What percentage of the pizza is left (whole number)?", "a": 25},
    {"q": "A bus seats 45 people. If 7 buses are needed to transport a school group, what is the maximum number of people?", "a": 315},
    {"q": "Mike runs 5 km on Monday, twice that on Tuesday, and 3 km less than Tuesday on Wednesday. Total km?", "a": 22},
    {"q": "A recipe calls for 2 cups of flour per 3 servings. How many cups for 9 servings?", "a": 6},
    {"q": "A tree was 8 feet tall in 2020 and grows 1.5 feet per year. How tall in 2026 (whole feet)?", "a": 17},
    {"q": "Lisa earned $300 in tips over 4 days. Her average was the same each day. How much per day?", "a": 75},
    {"q": "A bag has 30 marbles: 12 red, 8 blue, the rest green. What percentage are green?", "a": 33},
    {"q": "Three friends split a $90 dinner bill equally. Each leaves a 20% tip on their share. How much total does each pay?", "a": 36},
    {"q": "A factory makes 200 widgets per hour. How many widgets in a 7.5 hour shift?", "a": 1500},
]
NUMERIC_PROMPT = ("Solve the problem step by step. After your reasoning, "
                  "write your final numerical answer on its own line as: ANSWER: <number>")

def _strip_think(text):
    """Strip <think>...</think> reasoning blocks (Qwen3 etc). Handles:
       - Standard closed blocks
       - Unclosed <think> at start (model ran out of tokens mid-think)
       - Multiple think blocks
    """
    if not text:
        return ""
    # Closed blocks first
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Any remaining unclosed <think> consumes the rest (model truncated)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()

def _extract_number(text):
    text = _strip_think(text)
    # Prefer the LAST ANSWER: occurrence (some models say ANSWER: X mid-reasoning)
    matches = list(re.finditer(r"ANSWER:\s*\$?\s*([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE))
    if matches:
        return float(matches[-1].group(1))
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return float(nums[-1]) if nums else None

def bench_gsm8k(model_id):
    print("  [gsm8k]")
    correct = 0; details = []
    for i, p in enumerate(GSM8K_PROBLEMS):
        r = chat(model_id, p["q"], 768, temperature=0.0, system=NUMERIC_PROMPT)
        parsed = _extract_number(r["content"])
        ok = parsed is not None and abs(parsed - p["a"]) < 0.5
        if ok: correct += 1
        details.append({"idx": i, "expected": p["a"], "parsed": parsed, "correct": ok,
                        "tokens": r["usage"].get("completion_tokens", 0)})
    pct = correct / len(GSM8K_PROBLEMS) * 100
    print(f"    -> {correct}/{len(GSM8K_PROBLEMS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(GSM8K_PROBLEMS),
            "accuracy_pct": round(pct, 1), "problems": details}

ARITHMETIC_PROBLEMS = [
    {"q": "Compute: (15 * 23) + (47 * 11) - (8 * 12).", "a": 766},
    {"q": "Compute: 1248 / 8 + 256 - 17 * 4.", "a": 344},
    {"q": "What is 17% of 4500?", "a": 765},
    {"q": "Compute: 2^10 - 3^5.", "a": 781},
    {"q": "Compute: (144 + 169) * 3 - 600.", "a": 339},
    {"q": "What is the sum of integers from 1 to 50?", "a": 1275},
    {"q": "Compute: 7! / 5!.", "a": 42},
    {"q": "What is 0.375 expressed as a fraction in simplest form? Give the numerator only assuming denominator 8.", "a": 3},
    {"q": "Compute: gcd(84, 126).", "a": 42},
    {"q": "Compute: sqrt(625) + sqrt(196).", "a": 39},
]

def bench_arithmetic(model_id):
    print("  [arith]")
    correct = 0; details = []
    for i, p in enumerate(ARITHMETIC_PROBLEMS):
        # Generous budget so models with reasoning still have room for the final answer
        r = chat(model_id, p["q"], 1024, temperature=0.0, system=NUMERIC_PROMPT)
        parsed = _extract_number(r["content"])
        ok = parsed is not None and abs(parsed - p["a"]) < 0.5
        if ok: correct += 1
        details.append({"idx": i, "expected": p["a"], "parsed": parsed, "correct": ok})
    pct = correct / len(ARITHMETIC_PROBLEMS) * 100
    print(f"    -> {correct}/{len(ARITHMETIC_PROBLEMS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(ARITHMETIC_PROBLEMS),
            "accuracy_pct": round(pct, 1), "problems": details}

# Logic puzzles with verifiable short-answer
LOGIC_PROBLEMS = [
    {"q": "Three boxes labeled apples, oranges, mixed. Each label is wrong. You pick one fruit from the box labeled 'mixed' and get an apple. What is actually in the box labeled 'oranges'? One word.",
     "expected": ["mixed"]},
    {"q": "All cats are mammals. All mammals are animals. Are all cats animals? Answer yes or no only.",
     "expected": ["yes"]},
    {"q": "If today is Wednesday, what day of the week is it 100 days from now? Answer with just the day name.",
     "expected": ["friday"]},
    {"q": "A is taller than B. B is taller than C. C is taller than D. Who is the shortest? Answer just the letter.",
     "expected": ["d"]},
    {"q": "On an island, knights always tell the truth and knaves always lie. You meet two people, X and Y. X says 'Y is a knight.' Y says 'X and I are both knaves.' What is Y? One word: knight or knave.",
     "expected": ["knave"]},
    {"q": "If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets? Answer in minutes (number only).",
     "expected": ["5"]},
    {"q": "A man looks at a portrait and says: 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the portrait? Answer: my son, my father, my brother, or my nephew.",
     "expected": ["my son", "son"]},
    {"q": "There are 4 socks in a drawer: 2 red and 2 blue. In the dark, what is the minimum number you must pick to guarantee a matching pair? Number only.",
     "expected": ["3"]},
    {"q": "A snail climbs a 10-meter wall. Each day it climbs 3 meters but slides back 2 meters at night. How many days to reach the top? Number only.",
     "expected": ["8"]},
    {"q": "If A implies B, and B implies C, and we know NOT C, what can we conclude about A? Answer: must be true, must be false, or undetermined. Two or three words.",
     "expected": ["must be false", "false"]},
]

def _normalize_short(s):
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def bench_logic(model_id):
    print("  [logic]")
    sys_prompt = "Reason briefly, then give your final answer on a new line prefixed with 'ANSWER:'."
    correct = 0; details = []
    for i, p in enumerate(LOGIC_PROBLEMS):
        r = chat(model_id, p["q"], 512, temperature=0.0, system=sys_prompt)
        cleaned = _strip_think(r["content"])
        # Extract the last "ANSWER:" line or the last non-empty line
        ms = list(re.finditer(r"ANSWER:\s*(.+)", cleaned, re.IGNORECASE))
        if ms:
            ans = ms[-1].group(1).strip()
        else:
            lines = [l for l in cleaned.strip().split("\n") if l.strip()]
            ans = lines[-1].strip() if lines else ""
        norm = _normalize_short(ans)
        ok = any(_normalize_short(e) in norm for e in p["expected"])
        if ok: correct += 1
        details.append({"idx": i, "expected": p["expected"], "got": ans[:80], "correct": ok})
    pct = correct / len(LOGIC_PROBLEMS) * 100
    print(f"    -> {correct}/{len(LOGIC_PROBLEMS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(LOGIC_PROBLEMS),
            "accuracy_pct": round(pct, 1), "problems": details}

# Code-generation problems with pytest-style assertions
CODE_PROBLEMS = [
    {"name": "fibonacci",
     "q": "Write a Python function `fib(n)` that returns the nth Fibonacci number (0-indexed, fib(0)=0, fib(1)=1). Return only the code.",
     "tests": ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(10) == 55", "assert fib(15) == 610"]},
    {"name": "is_prime",
     "q": "Write a Python function `is_prime(n)` that returns True if n is prime, False otherwise. Return only the code.",
     "tests": ["assert is_prime(2) == True", "assert is_prime(17) == True", "assert is_prime(1) == False", "assert is_prime(100) == False", "assert is_prime(97) == True"]},
    {"name": "reverse_words",
     "q": "Write a Python function `reverse_words(s)` that returns the input string with the order of words reversed. Words are separated by single spaces. Return only the code.",
     "tests": ["assert reverse_words('hello world') == 'world hello'", "assert reverse_words('a b c') == 'c b a'", "assert reverse_words('one') == 'one'"]},
    {"name": "max_subarray",
     "q": "Write a Python function `max_subarray(nums)` returning the maximum sum of a contiguous subarray of nums (list of ints). Return only the code.",
     "tests": ["assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6", "assert max_subarray([1]) == 1", "assert max_subarray([5,4,-1,7,8]) == 23"]},
    {"name": "factorial",
     "q": "Write a Python function `factorial(n)` returning n! Return only the code.",
     "tests": ["assert factorial(0) == 1", "assert factorial(1) == 1", "assert factorial(5) == 120", "assert factorial(10) == 3628800"]},
    {"name": "anagram",
     "q": "Write a Python function `is_anagram(s, t)` returning True if s and t are anagrams (case-insensitive, ignore non-letters). Return only the code.",
     "tests": ["assert is_anagram('listen', 'silent') == True", "assert is_anagram('hello', 'world') == False", "assert is_anagram('Astronomer', 'Moon starer') == True"]},
    {"name": "rotate_list",
     "q": "Write a Python function `rotate(lst, k)` that returns a new list with the elements rotated right by k positions (k can be 0 or larger than length). Return only the code.",
     "tests": ["assert rotate([1,2,3,4,5], 2) == [4,5,1,2,3]", "assert rotate([1,2,3], 0) == [1,2,3]", "assert rotate([1,2,3], 4) == [3,1,2]"]},
    {"name": "gcd",
     "q": "Write a Python function `gcd(a, b)` returning the greatest common divisor. Return only the code.",
     "tests": ["assert gcd(12, 18) == 6", "assert gcd(48, 36) == 12", "assert gcd(7, 13) == 1", "assert gcd(100, 75) == 25"]},
    {"name": "count_vowels",
     "q": "Write a Python function `count_vowels(s)` returning the number of vowels (aeiou, case-insensitive) in s. Return only the code.",
     "tests": ["assert count_vowels('hello') == 2", "assert count_vowels('aeiou') == 5", "assert count_vowels('xyz') == 0", "assert count_vowels('AaEe') == 4"]},
    {"name": "balanced_parens",
     "q": "Write a Python function `balanced(s)` returning True iff parentheses in s are balanced. Only ()[]{} matter. Return only the code.",
     "tests": ["assert balanced('()') == True", "assert balanced('([{}])') == True", "assert balanced('([)]') == False", "assert balanced('') == True", "assert balanced('(((') == False"]},
]

def _extract_code(text):
    text = _strip_think(text)
    # Look for ```python ... ``` or ``` ... ```
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    # Look for def line and capture from there
    m = re.search(r"^(def\s+\w+.*)$", text, re.MULTILINE | re.DOTALL)
    if m: return text[m.start():].strip()
    return text.strip()

def bench_code(model_id):
    print("  [code]")
    correct = 0; details = []
    for prob in CODE_PROBLEMS:
        # Generous budget for reasoning-mode models (thinking + code together)
        r = chat(model_id, prob["q"], 1024, temperature=0.0)
        code = _extract_code(r["content"])
        # Combine code + tests in a temp file and exec
        full = code + "\n\n" + "\n".join(prob["tests"]) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full); tmppath = f.name
        try:
            res = subprocess.run([sys.executable, tmppath],
                                 capture_output=True, timeout=10, text=True)
            ok = (res.returncode == 0)
            err = (res.stderr or "")[:200] if not ok else None
        except Exception as e:
            ok = False; err = str(e)[:200]
        finally:
            os.unlink(tmppath)
        if ok: correct += 1
        details.append({"name": prob["name"], "passed": ok,
                        "error": err if not ok else None,
                        "code_preview": code[:200]})
        print(f"    {prob['name']:18}: {'PASS' if ok else 'FAIL'}")
    pct = correct / len(CODE_PROBLEMS) * 100
    print(f"    -> {correct}/{len(CODE_PROBLEMS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(CODE_PROBLEMS),
            "accuracy_pct": round(pct, 1), "problems": details}

# Instruction-following with verifiable constraints
INSTRUCTION_TESTS = [
    {"q": "Reply with exactly the word 'pong' and nothing else.",
     "check": lambda s: s.strip().lower() == "pong"},
    {"q": "Output a valid JSON object with keys 'name' (string) and 'count' (integer). No code fences, no other text.",
     "check": lambda s: _is_valid_json_with_keys(s, ["name", "count"])},
    {"q": "Write exactly 3 sentences about coffee. Each sentence must end with a period.",
     "check": lambda s: _has_n_sentences(s, 3)},
    {"q": "List exactly 5 colors, one per line, no numbering, no other text.",
     "check": lambda s: _is_n_nonempty_lines(s, 5)},
    {"q": "Output the result in the format: KEY=VALUE\\nKEY=VALUE for keys 'a' and 'b' with values 1 and 2.",
     "check": lambda s: "a=1" in s.lower() and "b=2" in s.lower()},
    {"q": "Reply with a single integer between 50 and 60 inclusive, with no other text.",
     "check": lambda s: _is_int_in_range(s, 50, 60)},
    {"q": "Write a sentence about dogs that contains exactly the word 'fetch' exactly once.",
     "check": lambda s: s.lower().split().count("fetch") == 1},
    {"q": "Output the following exactly: 'OK_TO_PROCEED'. No other text, no quotes.",
     "check": lambda s: s.strip() == "OK_TO_PROCEED"},
]

def _is_valid_json_with_keys(s, keys):
    s = s.strip()
    # strip optional code fence
    s = re.sub(r"^```(?:json)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    try:
        d = json.loads(s)
        return isinstance(d, dict) and all(k in d for k in keys)
    except Exception:
        return False

def _has_n_sentences(s, n):
    # Count periods at sentence boundaries
    parts = [p for p in re.split(r"[.!?]+\s*", s.strip()) if p.strip()]
    return len(parts) == n

def _is_n_nonempty_lines(s, n):
    lines = [l for l in s.strip().split("\n") if l.strip()]
    if len(lines) != n: return False
    # No numbering
    return not any(re.match(r"^\s*\d+[.)]\s", l) for l in lines)

def _is_int_in_range(s, lo, hi):
    s = s.strip()
    if not re.fullmatch(r"\d+", s): return False
    try:
        v = int(s); return lo <= v <= hi
    except Exception:
        return False

def bench_instruction(model_id):
    print("  [instruct]")
    correct = 0; details = []
    for i, t in enumerate(INSTRUCTION_TESTS):
        # Generous max_tokens so models with reasoning still have budget for final answer
        r = chat(model_id, t["q"], 512, temperature=0.0)
        cleaned = _strip_think(r["content"])
        try:
            ok = t["check"](cleaned)
        except Exception:
            ok = False
        if ok: correct += 1
        details.append({"idx": i, "passed": ok,
                        "response_preview": r["content"][:200],
                        "cleaned_preview": cleaned[:200]})
        print(f"    case {i+1}: {'PASS' if ok else 'FAIL'}")
    pct = correct / len(INSTRUCTION_TESTS) * 100
    print(f"    -> {correct}/{len(INSTRUCTION_TESTS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(INSTRUCTION_TESTS),
            "accuracy_pct": round(pct, 1), "cases": details}

# Needle-in-haystack: place a key fact at different depths, ask about it
NEEDLE_FACT_TEMPLATES = [
    ("The secret access code for the data vault is {code}.",
     "What is the secret access code for the data vault?",
     "{code}"),
    ("Remember that the meeting between Aria and Theo took place in {city} on Tuesday.",
     "In which city did the meeting between Aria and Theo take place?",
     "{city}"),
    ("The blue widget contains {n} hidden chambers, each storing a fragment of the map.",
     "How many hidden chambers does the blue widget contain?",
     "{n}"),
]
NEEDLE_DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]  # fraction of context where needle is placed
NEEDLE_CONTEXT_TOKENS = 4000

def bench_needle(model_id):
    print("  [needle]")
    fillers_needed = max(1, int(NEEDLE_CONTEXT_TOKENS * 4 / len(FILLER)) + 1)
    base = FILLER * fillers_needed
    # We'll generate 5 depths x 3 needles = 15 trials
    needles_data = [
        {"tmpl": NEEDLE_FACT_TEMPLATES[0], "vars": {"code": "X42-MERIDIAN-7"}, "expected_in": "X42-MERIDIAN-7"},
        {"tmpl": NEEDLE_FACT_TEMPLATES[1], "vars": {"city": "Brisbane"}, "expected_in": "Brisbane"},
        {"tmpl": NEEDLE_FACT_TEMPLATES[2], "vars": {"n": "seventeen"}, "expected_in": "seventeen"},
    ]
    correct = 0; total = 0; details = []
    for depth in NEEDLE_DEPTHS:
        for nd in needles_data:
            fact_sentence = nd["tmpl"][0].format(**nd["vars"])
            question = nd["tmpl"][1]
            expected = nd["expected_in"].lower()
            # Place needle at the given depth in base text
            cut = int(len(base) * depth)
            ctx = base[:cut] + " " + fact_sentence + " " + base[cut:]
            prompt = f"{ctx}\n\nBased on the passage above, answer concisely:\n{question}"
            r = chat(model_id, prompt, 128, temperature=0.0)
            cleaned = _strip_think(r["content"])
            ok = expected in cleaned.lower()
            if ok: correct += 1
            total += 1
            details.append({"depth": depth, "expected": expected,
                            "response_preview": r["content"][:150],
                            "cleaned_preview": cleaned[:150], "correct": ok})
    pct = correct / total * 100 if total else 0
    print(f"    -> {correct}/{total} = {pct:.1f}%")
    return {"correct": correct, "total": total, "accuracy_pct": round(pct, 1),
            "trials": details}

# RAG: long passage + questions with exact-match keywords
RAG_PASSAGE = """
The Caspian Sea is the world's largest inland body of water, with a surface area of
about 371,000 square kilometers. It is bordered by five countries: Kazakhstan,
Russia, Azerbaijan, Iran, and Turkmenistan. Despite being called a sea, it is
technically the world's largest lake. Its salinity is approximately one-third that
of typical ocean water. The deepest point reaches 1,025 meters in the southern
basin. The Volga River, originating in Russia, supplies about 80% of the freshwater
inflow into the Caspian. The Caspian is home to the beluga sturgeon, which is the
source of beluga caviar. The region is rich in oil and natural gas reserves,
estimated at 48 billion barrels of oil and 292 trillion cubic feet of natural gas
as of recent surveys. The Apsheron Peninsula, near Baku in Azerbaijan, has been
producing oil since the 19th century. Climate change has caused the sea level to
drop by approximately 1.5 meters since the mid-1990s, with projections suggesting
a further decline of up to 9 meters by 2100 under high-emissions scenarios.
"""
RAG_QUESTIONS = [
    {"q": "What is the surface area of the Caspian Sea in square kilometers (number only)?", "expected": "371000"},
    {"q": "How many countries border the Caspian Sea (number only)?", "expected": "5"},
    {"q": "What is the deepest point in meters (number only)?", "expected": "1025"},
    {"q": "Which river supplies about 80% of the freshwater inflow? One word.", "expected": "volga"},
    {"q": "What kind of fish is the source of beluga caviar? Two words including 'sturgeon'.", "expected": "sturgeon"},
    {"q": "How many billion barrels of oil are estimated in the region (number only)?", "expected": "48"},
    {"q": "Near which city has oil been produced since the 19th century?", "expected": "baku"},
    {"q": "By approximately how many meters has the sea level dropped since the mid-1990s (number with decimals allowed)?", "expected": "1.5"},
]

def bench_rag(model_id):
    print("  [rag]")
    correct = 0; details = []
    for i, qd in enumerate(RAG_QUESTIONS):
        prompt = f"Read the passage and answer concisely with just the requested information.\n\nPASSAGE:\n{RAG_PASSAGE.strip()}\n\nQUESTION: {qd['q']}"
        r = chat(model_id, prompt, 256, temperature=0.0)
        cleaned = _strip_think(r["content"])
        got_norm = re.sub(r"[^\w]", "", cleaned.lower())
        expected_norm = re.sub(r"[^\w]", "", qd["expected"].lower())
        ok = expected_norm in got_norm
        if ok: correct += 1
        details.append({"idx": i, "expected": qd["expected"],
                        "response_preview": r["content"][:120], "correct": ok})
    pct = correct / len(RAG_QUESTIONS) * 100
    print(f"    -> {correct}/{len(RAG_QUESTIONS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(RAG_QUESTIONS),
            "accuracy_pct": round(pct, 1), "questions": details}

# Multi-hop reasoning
MULTIHOP_PROBLEMS = [
    {"q": "Alice is older than Bob. Bob is older than Carol. Carol is older than Dan. Dan is 10. If Carol is twice Dan's age and Bob is 5 years older than Carol, how old is Bob? Number only.",
     "a": 25},
    {"q": "There are 3 boxes. Box A has twice as many apples as Box B. Box C has 3 times as many as Box B. Together they have 60 apples. How many in Box A?",
     "a": 20},
    {"q": "A library has science and history books in ratio 3:5. If there are 240 books total, how many are science books?",
     "a": 90},
    {"q": "Jane saved $200 in January, then doubled her savings each month for 3 more months. What were her total savings at the end of April?",
     "a": 3000},
    {"q": "A company has 4 departments. HR has 12 employees. Sales has 3 times HR. Engineering has 2 times Sales. Operations has half of Engineering. Total employees?",
     "a": 156},
]

def bench_multihop(model_id):
    print("  [multihop]")
    correct = 0; details = []
    for i, p in enumerate(MULTIHOP_PROBLEMS):
        r = chat(model_id, p["q"], 768, temperature=0.0, system=NUMERIC_PROMPT)
        parsed = _extract_number(r["content"])
        ok = parsed is not None and abs(parsed - p["a"]) < 0.5
        if ok: correct += 1
        details.append({"idx": i, "expected": p["a"], "parsed": parsed, "correct": ok})
    pct = correct / len(MULTIHOP_PROBLEMS) * 100
    print(f"    -> {correct}/{len(MULTIHOP_PROBLEMS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(MULTIHOP_PROBLEMS),
            "accuracy_pct": round(pct, 1), "problems": details}

# Knowledge accuracy (verifiable hard facts)
KNOWLEDGE_QUESTIONS = [
    {"q": "What is the chemical symbol for gold? One or two letters only.", "expected": ["au"]},
    {"q": "In what year did the first human land on the moon? Year only.", "expected": ["1969"]},
    {"q": "What is the SI unit of electric current? One word.", "expected": ["ampere", "amp"]},
    {"q": "Who painted the Mona Lisa? Two-word name.", "expected": ["leonardo da vinci", "da vinci", "leonardo"]},
    {"q": "What is the smallest prime number? Number only.", "expected": ["2"]},
    {"q": "What gas makes up about 78% of Earth's atmosphere by volume? One word.", "expected": ["nitrogen"]},
    {"q": "What is the speed of light in vacuum in km/s, rounded to nearest thousand? Number only.", "expected": ["300000", "299792"]},
    {"q": "Which planet has the most moons as of recent count? One word.", "expected": ["saturn"]},
    {"q": "What is the largest country in the world by land area? One word.", "expected": ["russia"]},
    {"q": "In what year did World War II end? Year only.", "expected": ["1945"]},
]

def bench_knowledge(model_id):
    print("  [knowledge]")
    sys_prompt = "Reply with just the requested information, no extra prose."
    correct = 0; details = []
    for i, qd in enumerate(KNOWLEDGE_QUESTIONS):
        r = chat(model_id, qd["q"], 256, temperature=0.0, system=sys_prompt)
        cleaned = _strip_think(r["content"])
        got_norm = _normalize_short(cleaned)
        ok = any(_normalize_short(e) in got_norm for e in qd["expected"])
        if ok: correct += 1
        details.append({"idx": i, "expected": qd["expected"],
                        "response_preview": r["content"][:80], "correct": ok})
    pct = correct / len(KNOWLEDGE_QUESTIONS) * 100
    print(f"    -> {correct}/{len(KNOWLEDGE_QUESTIONS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(KNOWLEDGE_QUESTIONS),
            "accuracy_pct": round(pct, 1), "questions": details}

# Anti-hallucination: unanswerable / impossible questions
HALLUCINATION_QUESTIONS = [
    "What was the exact text of the email I sent on March 14, 2023?",
    "List the names of all 47 children of the fictional character John Smith from my unpublished novel.",
    "What is the population of the country Zorblax as of 2024?",
    "Quote the exact contents of file /tmp/secret-document.txt on my computer.",
    "What did I have for breakfast yesterday?",
]
REFUSAL_INDICATORS = [
    "i don't know", "i do not know", "cannot", "can't", "no information",
    "not aware", "no way to", "no access", "no data", "not possible",
    "i am unable", "i'm unable", "no record", "not provided", "as an ai",
    "i don't have", "i do not have", "without more", "made up", "fictional",
    "no such country", "no public", "not a real", "doesn't exist",
]

def bench_hallucination(model_id):
    print("  [no_halluc]")
    sys_prompt = "Answer honestly. If you cannot answer for any reason, say so explicitly."
    refused = 0; details = []
    for i, q in enumerate(HALLUCINATION_QUESTIONS):
        r = chat(model_id, q, 384, temperature=0.0, system=sys_prompt)
        cleaned = _strip_think(r["content"]).lower()
        is_refusal = any(ind in cleaned for ind in REFUSAL_INDICATORS)
        if is_refusal: refused += 1
        details.append({"idx": i, "refused": is_refusal,
                        "response_preview": r["content"][:200]})
        print(f"    Q{i+1}: {'REFUSED' if is_refusal else 'HALLUCINATED'}")
    pct = refused / len(HALLUCINATION_QUESTIONS) * 100
    print(f"    -> {refused}/{len(HALLUCINATION_QUESTIONS)} = {pct:.1f}%")
    return {"refused": refused, "total": len(HALLUCINATION_QUESTIONS),
            "refusal_pct": round(pct, 1), "questions": details}

# ============================================================================
# Tier 4: Multimodal (vision) — auto-skip for text-only models
# ============================================================================

MULTIMODAL_HINTS = ("gemma-3", "gemma-4", "vl-", "-vl", "vision", "llava",
                    "phi-3-vision", "qwen2-vl", "qwen2.5-vl", "qwen3.5", "qwen3_5",
                    "internvl", "molmo")
MODELS_DIR_HOST = "/home/dewie/ai-stack-intel/openvino-models"


def _is_multimodal(model_id):
    """True if model has vision capability.
    Definitive: openvino_vision_embeddings_model.bin on disk.
    Fallback: model_id substring matches MULTIMODAL_HINTS.
    """
    name = model_id.split("/", 1)[-1]
    marker = os.path.join(MODELS_DIR_HOST, name, "openvino_vision_embeddings_model.bin")
    if os.path.exists(marker):
        return True
    return any(h in model_id.lower() for h in MULTIMODAL_HINTS)
IMAGES_DIR = "images"


def _ensure_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        return True
    except ImportError:
        return False


def _data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _gen_color_image(color_rgb, size=(256, 256)):
    from PIL import Image
    img = Image.new("RGB", size, color_rgb)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def _gen_shape_image(shape, size=(256, 256), color=(0, 0, 0), bg=(255, 255, 255)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, bg); d = ImageDraw.Draw(img)
    w, h = size; pad = 40
    if shape == "circle":
        d.ellipse([pad, pad, w - pad, h - pad], fill=color)
    elif shape == "square":
        d.rectangle([pad, pad, w - pad, h - pad], fill=color)
    elif shape == "triangle":
        d.polygon([(w // 2, pad), (pad, h - pad), (w - pad, h - pad)], fill=color)
    elif shape == "star":
        import math
        cx, cy = w // 2, h // 2
        outer = (w - 2 * pad) / 2; inner = outer * 0.4
        pts = []
        for i in range(10):
            r = outer if i % 2 == 0 else inner
            a = math.pi / 2 - i * math.pi / 5
            pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
        d.polygon(pts, fill=color)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def _gen_text_image(text, size=(400, 200), font_size=64):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, (255, 255, 255)); d = ImageDraw.Draw(img)
    font = None
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            try: font = ImageFont.truetype(f, font_size); break
            except Exception: pass
    if font is None:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size[0] - tw) / 2 - bbox[0], (size[1] - th) / 2 - bbox[1]),
           text, fill=(0, 0, 0), font=font)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def _chat_with_image(model_id, prompt, png_bytes, max_tokens=64):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _data_uri(png_bytes)}},
    ]}]
    t0 = time.monotonic()
    resp = api_post(LLM_URL + "/v3/chat/completions", {
        "model": model_id, "messages": msgs,
        "max_tokens": max_tokens, "temperature": 0.0,
    }, timeout=600)
    return {
        "content": resp.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "usage": resp.get("usage", {}),
        "elapsed_s": time.monotonic() - t0,
    }


def bench_multimodal(model_id):
    print("  [multimodal]")
    if not _is_multimodal(model_id):
        print("    skipped (no vision artifacts and no multimodal name hint)")
        return {"skipped": True, "reason": "no vision capability detected"}
    if not _ensure_pillow():
        print("    skipped (Pillow not installed — apt install python3-pil)")
        return {"skipped": True, "reason": "Pillow not available"}
    os.makedirs(IMAGES_DIR, exist_ok=True)

    tests = []
    for label, rgb in [("red", (220, 30, 30)), ("blue", (30, 60, 220)), ("green", (30, 180, 60))]:
        tests.append({"category": "color", "label": f"color:{label}",
                      "question": "What is the dominant color in this image? Respond with one word.",
                      "expected_in": [label], "image": _gen_color_image(rgb)})
    for shape in ["circle", "square", "triangle", "star"]:
        tests.append({"category": "shape", "label": f"shape:{shape}",
                      "question": "What shape is shown in this image? Respond with one word.",
                      "expected_in": [shape], "image": _gen_shape_image(shape)})
    for text in ["OWLBEAR42", "1729", "HELLO"]:
        tests.append({"category": "ocr", "label": f"ocr:{text}",
                      "question": "Read the text in this image. Respond with only the text, exactly as written.",
                      "expected_in": [text.lower()], "image": _gen_text_image(text)})

    correct = 0; details = []
    for t in tests:
        try:
            r = _chat_with_image(model_id, t["question"], t["image"], max_tokens=128)
            got = _strip_think(r["content"] or "").lower()
            got_norm = "".join(c for c in got if c.isalnum()) if t["category"] == "ocr" else got
            ok = any(exp.lower() in got_norm for exp in t["expected_in"])
            if ok: correct += 1
            details.append({"label": t["label"], "category": t["category"],
                            "expected_in": t["expected_in"],
                            "got": r["content"][:120], "correct": ok})
            print(f"    {t['label']:18}: {'OK' if ok else 'FAIL'} -> {r['content'][:60]!r}")
        except Exception as e:
            details.append({"label": t["label"], "error": str(e)[:200], "correct": False})
            print(f"    {t['label']:18}: ERROR {e}")
    pct = correct / len(tests) * 100 if tests else 0
    print(f"    -> {correct}/{len(tests)} = {pct:.1f}%")
    return {"correct": correct, "total": len(tests),
            "accuracy_pct": round(pct, 1), "trials": details}

# ============================================================================
# Main orchestration
# ============================================================================

def short_name(model_id):
    return model_id.split("/")[-1].replace("-int4-ov", "")

TESTS_ORDER = [
    ("throughput", bench_throughput),
    ("ttft", bench_ttft),
    ("long_context", bench_long_context),
    ("prompt_processing", bench_prompt_processing),
    ("gsm8k", bench_gsm8k),
    ("arithmetic", bench_arithmetic),
    ("logic", bench_logic),
    ("code", bench_code),
    ("instruction", bench_instruction),
    ("needle", bench_needle),
    ("rag", bench_rag),
    ("multihop", bench_multihop),
    ("knowledge", bench_knowledge),
    ("hallucination", bench_hallucination),
    ("multimodal", bench_multimodal),
]

def run_one(model_id, skip=None, only=None):
    skip = skip or set()
    only = only or set()
    print(f"\n=== {model_id} ===")
    switch_to(model_id)
    out = {
        "model_id": model_id,
        "short_name": short_name(model_id),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "tests": {},
    }
    for name, fn in TESTS_ORDER:
        if only and name not in only:
            continue
        if name in skip:
            print(f"  [{name}] skipped"); continue
        try:
            t0 = time.time()
            out["tests"][name] = fn(model_id)
            out["tests"][name]["_elapsed_s"] = round(time.time() - t0, 1)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            out["tests"][name] = {"error": str(e)[:300]}
    out["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = int(time.time())
    path = f"{RESULTS_DIR}/{out['short_name']}_{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved -> {path}")
    return out

def main():
    if len(sys.argv) < 2:
        print("usage: bench_compare.py MODEL_ID [MODEL_ID ...]")
        print("  --skip=name1,name2  skip specific tests")
        print("  --only=name1,name2  run only these tests")
        print(f"  available tests: {','.join(n for n,_ in TESTS_ORDER)}")
        sys.exit(2)
    args = sys.argv[1:]
    skip = set(); only = set()
    models = []
    for a in args:
        if a.startswith("--skip="):
            skip = set(a.split("=", 1)[1].split(","))
        elif a.startswith("--only="):
            only = set(a.split("=", 1)[1].split(","))
        else:
            models.append(a)
    original = api_get(f"{SWITCHER_URL}/api/status").get("current_model")
    print(f"Models to test: {len(models)}")
    print(f"Skipping: {skip if skip else '(none)'}")
    print(f"Only: {only if only else '(all)'}")
    print(f"Will restore to: {original}\n")
    for m in models:
        try:
            run_one(m, skip=skip, only=only)
        except Exception as e:
            print(f"\nFATAL on {m}: {e}")
    if original and original not in models:
        print(f"\nRestoring {original}")
        try:
            switch_to(original)
        except Exception as e:
            print(f"  WARN: could not restore: {e}")
    print("\nDone. Next: python3 compare_runs.py results/compare/*.json")

if __name__ == "__main__":
    main()
