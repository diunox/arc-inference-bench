#!/usr/bin/env python3
"""Tool-use / structured-output benchmark for OVMS on Janus.

Companion to bench_compare.py and bench_vlm.py. Tests prompt-based JSON
emission — does the model produce parseable structured output when asked?
This is the capability that smart-router, AnythingLLM agents, and any
auto-tagging pipeline actually depend on.

Usage:
    python3 bench_tools.py [--only=names] [--skip=names] MODEL_ID [MODEL_ID ...]

Suites:
  call         : single tool call from a one-tool schema (4 tests)
  select       : pick the correct tool from a list of N (3 tests)
  refuse       : return a no-tool sentinel when no tool fits (3 tests)
  json_extract : extract structured data from a passage as JSON (4 tests)

Grading is structural: parse the model's JSON, check the right function
name was chosen, check required argument names + values match expectations.
Substring fallback if JSON parsing fails so models that emit valid-looking
tool calls without proper JSON structure still get partial credit.

Results land in results/tools/<short_name>_<unix_ts>.json. Restores prior
model on exit. Strips <think>...</think> reasoning blocks (including the
Qwen3.5 missing-open-tag pattern) before extraction.
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_compare import (
    api_get, chat, switch_to, _strip_think,
    SWITCHER_URL,
)

RESULTS_DIR = "results/tools"

# Each model gets the same system instruction telling it how to emit calls.
SYSTEM_TOOL = (
    "You are a tool-using assistant. When the user's request matches a "
    "provided tool, respond with a JSON object on a single line with this "
    "shape:\n"
    '  {"name": "<tool_name>", "arguments": {<arg_name>: <arg_value>, ...}}\n'
    "If NONE of the provided tools fits the request, respond with exactly:\n"
    '  {"name": null}\n'
    "Do not wrap the JSON in code fences. Do not add commentary before or "
    "after the JSON. Output the JSON and nothing else."
)

SYSTEM_EXTRACT = (
    "You are a data extraction assistant. The user will give you a passage "
    "and a JSON schema. Respond with ONLY a JSON object matching the schema. "
    "No code fences, no commentary — just the JSON object on a single line."
)


# ---------- JSON helpers ----------

_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _extract_json(text):
    """Pull a JSON object out of a model response. Returns dict | list | None.

    Handles:
      - <think>...</think> blocks (using bench_compare._strip_think)
      - Code fences ```json ... ```
      - Free-text framing — finds the FIRST balanced {...} or [...] block
    """
    text = _strip_think(text or "")
    if not text:
        return None
    # Code-fenced JSON wins if present
    m = _CODE_FENCE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # Find a balanced {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start >= 0:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                c = text[i]
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    return None


# ---------- Suite: call ----------

CALL_TESTS = [
    {
        "tool": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "arguments": {"city": "string"},
        },
        "user": "What's the weather like in Tokyo right now?",
        "expected_name": "get_weather",
        "expected_args": {"city": ["tokyo"]},
    },
    {
        "tool": {
            "name": "search_files",
            "description": "Search files by name pattern in a directory.",
            "arguments": {"directory": "string", "pattern": "string"},
        },
        "user": "Find all Python files under /home/dewie/bench.",
        "expected_name": "search_files",
        "expected_args": {
            "directory": ["/home/dewie/bench"],
            "pattern": ["*.py", ".py"],
        },
    },
    {
        "tool": {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "arguments": {"to": "string", "subject": "string", "body": "string"},
        },
        "user": "Email alice@example.com with subject 'Meeting Tomorrow' saying we'll meet at 3pm.",
        "expected_name": "send_email",
        "expected_args": {
            "to": ["alice@example.com"],
            "subject": ["meeting tomorrow"],
            "body": ["3pm", "meet"],
        },
    },
    {
        "tool": {
            "name": "create_calendar_event",
            "description": "Create a calendar event.",
            "arguments": {"title": "string", "date": "ISO date YYYY-MM-DD", "duration_minutes": "integer"},
        },
        "user": "Schedule a 30-minute team standup on 2026-07-15 titled 'Sprint Review'.",
        "expected_name": "create_calendar_event",
        "expected_args": {
            "title": ["sprint review"],
            "date": ["2026-07-15"],
            "duration_minutes": [30],
        },
    },
]


def _format_tools_prompt(tools, user_msg):
    desc = "Available tools:\n"
    for t in tools:
        args = ", ".join(f"{k}: {v}" for k, v in t["arguments"].items())
        desc += f"  - {t['name']}({args}) — {t['description']}\n"
    return desc + "\nUser request: " + user_msg


def _match_arg(actual, expected_options):
    """expected_options is a list of acceptable substrings/exact-values.
    Numbers match exactly; strings match case-insensitive substring on either side.
    """
    if isinstance(actual, (int, float)):
        for opt in expected_options:
            if isinstance(opt, (int, float)) and abs(actual - opt) < 0.001:
                return True
        return False
    actual_s = str(actual).lower().strip()
    for opt in expected_options:
        opt_s = str(opt).lower().strip()
        if opt_s in actual_s or actual_s in opt_s:
            return True
    return False


def _grade_tool_call(obj, expected_name, expected_args):
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    if obj.get("name") != expected_name:
        return False, f"wrong tool name: got {obj.get('name')!r}, want {expected_name!r}"
    args = obj.get("arguments") or obj.get("args") or {}
    if not isinstance(args, dict):
        return False, "arguments not a dict"
    for k, opts in expected_args.items():
        if k not in args:
            return False, f"missing argument {k!r}"
        if not _match_arg(args[k], opts):
            return False, f"argument {k}={args[k]!r} does not match expected {opts}"
    return True, "ok"


def bench_call(model_id):
    print("  [call]")
    correct = 0
    details = []
    for i, t in enumerate(CALL_TESTS):
        prompt = _format_tools_prompt([t["tool"]], t["user"])
        r = chat(model_id, prompt, max_tokens=300, temperature=0.0, system=SYSTEM_TOOL)
        obj = _extract_json(r["content"])
        ok, why = _grade_tool_call(obj or {}, t["expected_name"], t["expected_args"])
        correct += int(ok)
        details.append({"i": i, "ok": ok, "why": why,
                        "parsed": obj,
                        "response": (r["content"] or "")[:500]})
        print(f"    test {i} ({t['expected_name']}): {'PASS' if ok else 'FAIL'}  {why}")
    pct = 100 * correct / len(CALL_TESTS)
    print(f"    -> {correct}/{len(CALL_TESTS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(CALL_TESTS), "pct": round(pct, 1), "details": details}


# ---------- Suite: select ----------

TOOLBOX_A = [
    {"name": "get_weather", "description": "Get current weather for a city.",
     "arguments": {"city": "string"}},
    {"name": "send_email", "description": "Send an email.",
     "arguments": {"to": "string", "subject": "string", "body": "string"}},
    {"name": "create_calendar_event", "description": "Create a calendar event.",
     "arguments": {"title": "string", "date": "YYYY-MM-DD"}},
]

SELECT_TESTS = [
    {"toolbox": TOOLBOX_A, "user": "Is it raining in Paris right now?",
     "expected_name": "get_weather",
     "expected_args": {"city": ["paris"]}},
    {"toolbox": TOOLBOX_A, "user": "Put a haircut on my calendar for 2026-08-01.",
     "expected_name": "create_calendar_event",
     "expected_args": {"title": ["haircut"], "date": ["2026-08-01"]}},
    {"toolbox": TOOLBOX_A, "user": "Reply to bob@x.com with 'thanks' as the subject.",
     "expected_name": "send_email",
     "expected_args": {"to": ["bob@x.com"], "subject": ["thanks"]}},
]


def bench_select(model_id):
    print("  [select]")
    correct = 0
    details = []
    for i, t in enumerate(SELECT_TESTS):
        prompt = _format_tools_prompt(t["toolbox"], t["user"])
        r = chat(model_id, prompt, max_tokens=300, temperature=0.0, system=SYSTEM_TOOL)
        obj = _extract_json(r["content"])
        ok, why = _grade_tool_call(obj or {}, t["expected_name"], t["expected_args"])
        correct += int(ok)
        details.append({"i": i, "ok": ok, "why": why,
                        "parsed": obj,
                        "response": (r["content"] or "")[:500]})
        print(f"    test {i} (-> {t['expected_name']}): {'PASS' if ok else 'FAIL'}  {why}")
    pct = 100 * correct / len(SELECT_TESTS)
    print(f"    -> {correct}/{len(SELECT_TESTS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(SELECT_TESTS), "pct": round(pct, 1), "details": details}


# ---------- Suite: refuse ----------

REFUSE_TESTS = [
    {"toolbox": TOOLBOX_A, "user": "What's the capital of France?"},
    {"toolbox": TOOLBOX_A, "user": "Translate 'good morning' into Japanese."},
    {"toolbox": TOOLBOX_A, "user": "Write me a Python function to sort a list."},
]


def bench_refuse(model_id):
    """Pass if model emits {"name": null} or otherwise indicates no-tool. Models that
    hallucinate a call fail. Models that return non-JSON refusal also fail (must be JSON)."""
    print("  [refuse]")
    correct = 0
    details = []
    for i, t in enumerate(REFUSE_TESTS):
        prompt = _format_tools_prompt(t["toolbox"], t["user"])
        r = chat(model_id, prompt, max_tokens=300, temperature=0.0, system=SYSTEM_TOOL)
        obj = _extract_json(r["content"])
        # PASS conditions: parsed JSON with "name": null OR omitted name
        if isinstance(obj, dict) and (obj.get("name") is None or "name" not in obj):
            ok, why = True, "correctly returned null"
        else:
            ok, why = False, f"hallucinated call: {obj}"
        correct += int(ok)
        details.append({"i": i, "ok": ok, "why": why,
                        "parsed": obj,
                        "response": (r["content"] or "")[:500]})
        print(f"    test {i}: {'PASS' if ok else 'FAIL'}  {why}")
    pct = 100 * correct / len(REFUSE_TESTS)
    print(f"    -> {correct}/{len(REFUSE_TESTS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(REFUSE_TESTS), "pct": round(pct, 1), "details": details}


# ---------- Suite: json_extract ----------

EXTRACT_TESTS = [
    {
        "schema": {"name": "string", "company": "string", "role": "string"},
        "passage": "Alice Chen joined Acme Corp as Director of Engineering in March 2025.",
        "expected": {"name": ["alice chen", "alice"], "company": ["acme"], "role": ["director", "engineering"]},
    },
    {
        "schema": {"total": "number", "currency": "string", "items": "integer"},
        "passage": "Receipt: 3 items purchased, total $47.50 USD.",
        "expected": {"total": [47.50, 47.5], "currency": ["usd"], "items": [3]},
    },
    {
        "schema": {"event": "string", "date": "YYYY-MM-DD", "attendees": "integer"},
        "passage": "Quarterly review meeting on 2026-09-12 with 8 stakeholders attending.",
        "expected": {"event": ["quarterly", "review"], "date": ["2026-09-12"], "attendees": [8]},
    },
    {
        "schema": {"city": "string", "temperature_c": "number", "conditions": "string"},
        "passage": "Today in Oslo: 14 degrees Celsius, overcast with light rain expected later.",
        "expected": {"city": ["oslo"], "temperature_c": [14], "conditions": ["overcast", "rain"]},
    },
]


def _format_extract_prompt(schema, passage):
    return (
        f"Schema (JSON shape required):\n{json.dumps(schema, indent=2)}\n\n"
        f"Passage:\n{passage}\n\n"
        f"Extract the fields from the passage as a JSON object matching the schema."
    )


def _grade_extract(obj, expected):
    if not isinstance(obj, dict):
        return False, "not a dict"
    for k, opts in expected.items():
        if k not in obj:
            return False, f"missing field {k!r}"
        if not _match_arg(obj[k], opts):
            return False, f"field {k}={obj[k]!r} does not match {opts}"
    return True, "ok"


def bench_json_extract(model_id):
    print("  [json_extract]")
    correct = 0
    details = []
    for i, t in enumerate(EXTRACT_TESTS):
        prompt = _format_extract_prompt(t["schema"], t["passage"])
        r = chat(model_id, prompt, max_tokens=400, temperature=0.0, system=SYSTEM_EXTRACT)
        obj = _extract_json(r["content"])
        ok, why = _grade_extract(obj or {}, t["expected"])
        correct += int(ok)
        details.append({"i": i, "ok": ok, "why": why,
                        "parsed": obj,
                        "response": (r["content"] or "")[:500]})
        print(f"    test {i}: {'PASS' if ok else 'FAIL'}  {why}")
    pct = 100 * correct / len(EXTRACT_TESTS)
    print(f"    -> {correct}/{len(EXTRACT_TESTS)} = {pct:.1f}%")
    return {"correct": correct, "total": len(EXTRACT_TESTS), "pct": round(pct, 1), "details": details}


# ---------- driver ----------

SUITES = [
    ("call", bench_call),
    ("select", bench_select),
    ("refuse", bench_refuse),
    ("json_extract", bench_json_extract),
]


def run_one(model_id, only=None, skip=None):
    only = only or set()
    skip = skip or set()
    print(f"\n=== {model_id} ===")
    print(f"  switching to {model_id} ...")
    switch_to(model_id)
    results = {"model": model_id, "ts": int(time.time())}
    for name, fn in SUITES:
        if only and name not in only:
            continue
        if name in skip:
            print(f"  [{name}] skipped")
            continue
        try:
            results[name] = fn(model_id)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            results[name] = {"error": str(e)}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    short = model_id.split("/")[-1].replace("-int4-ov", "")
    out = os.path.join(RESULTS_DIR, f"{short}_{results['ts']}.json")
    open(out, "w").write(json.dumps(results, indent=2))
    print(f"\n  saved -> {out}")
    return results


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    only, skip, models = set(), set(), []
    for a in args:
        if a.startswith("--only="):
            only = set(a.split("=", 1)[1].split(","))
        elif a.startswith("--skip="):
            skip = set(a.split("=", 1)[1].split(","))
        else:
            models.append(a)
    try:
        st = api_get(f"{SWITCHER_URL}/api/status")
        restore_to = st.get("current_model")
    except Exception:
        restore_to = None
    print(f"Models to test: {len(models)}")
    print(f"Only: {only if only else '(all)'}")
    print(f"Skip: {skip if skip else '(none)'}")
    print(f"Will restore to: {restore_to or '(unknown)'}")
    try:
        for m in models:
            run_one(m, only=only, skip=skip)
    finally:
        if restore_to:
            print(f"\nRestoring {restore_to}")
            try:
                print(f"  switching to {restore_to} ...")
                switch_to(restore_to)
            except Exception as e:
                print(f"  WARN: could not restore: {e}", file=sys.stderr)
    print("\nDone.")


if __name__ == "__main__":
    main()
