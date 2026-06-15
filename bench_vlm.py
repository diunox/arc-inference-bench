#!/usr/bin/env python3
"""VLM benchmark for OVMS on Janus.

Companion to bench_compare.py for vision-language models. Auto-detects
multimodal capability via on-disk vision artifacts and skips text-only models.

Usage:
    python3 bench_vlm.py [--only=names] [--skip=names] MODEL_ID [MODEL_ID ...]

Suites:
  count    : object counting (4 scenes: 1, 3, 5, 7 dots)
  spatial  : relative position reasoning (4 two-shape scenes)
  chart    : bar chart comprehension (3 charts)
  layout   : multi-line text + numeric extraction (3 documents)

Each suite produces an accuracy score (correct/total). Results go to
results/vlm/<short_name>_<unix_ts>.json. Restores prior model on exit.
"""

import io
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_compare import (
    api_get, api_post, switch_to,
    _data_uri, _ensure_pillow,
    _is_multimodal,
    SWITCHER_URL, LLM_URL,
)

SYSTEM_PROMPT = (
    "You answer the user's question directly and concisely. "
    "Do not narrate your reasoning, do not say 'the user wants', "
    "do not restate the question. Just give the answer."
)


def _chat_with_image(model_id, prompt, png_bytes, max_tokens=80):
    import time as _t
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _data_uri(png_bytes)}},
        ]},
    ]
    t0 = _t.monotonic()
    resp = api_post(LLM_URL + "/v3/chat/completions", {
        "model": model_id, "messages": msgs,
        "max_tokens": max_tokens, "temperature": 0.0,
    }, timeout=600)
    return {
        "content": resp.get("choices", [{}])[0].get("message", {}).get("content", ""),
        "usage": resp.get("usage", {}),
        "elapsed_s": _t.monotonic() - t0,
    }

RESULTS_DIR = "results/vlm"


# ----- image generators -----

def _gen_count_scene(n, color=(40, 80, 200), size=(384, 384), bg=(245, 245, 245)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    w, h = size
    margin, r = 50, 28
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    cell_w = (w - 2 * margin) / max(1, cols)
    cell_h = (h - 2 * margin) / max(1, rows)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            cx = margin + cell_w * (col + 0.5)
            cy = margin + cell_h * (row + 0.5)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            idx += 1
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _draw_shape(d, cx, cy, shape, r, color):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif shape == "star":
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.4
            a = math.pi / 2 - i * math.pi / 5
            pts.append((cx + rr * math.cos(a), cy - rr * math.sin(a)))
        d.polygon(pts, fill=color)


def _gen_spatial_scene(left_shape, right_shape, size=(512, 256), bg=(245, 245, 245)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    w, h = size
    r = 70
    _draw_shape(d, w // 4, h // 2, left_shape, r, (200, 40, 40))
    _draw_shape(d, 3 * w // 4, h // 2, right_shape, r, (40, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gen_bar_chart(labels, values, size=(512, 384)):
    from PIL import Image, ImageDraw, ImageFont
    bg, axis = (255, 255, 255), (40, 40, 40)
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    w, h = size
    pad_top, pad_bot, pad_x = 50, 60, 50
    font = None
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(f):
            try:
                font = ImageFont.truetype(f, 18)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    d.line([(pad_x, pad_top), (pad_x, h - pad_bot)], fill=axis, width=2)
    d.line([(pad_x, h - pad_bot), (w - pad_x, h - pad_bot)], fill=axis, width=2)
    n = len(labels)
    bar_w = (w - 2 * pad_x) / (n * 1.5)
    gap = bar_w / 2
    max_v = max(values)
    chart_h = h - pad_top - pad_bot
    for i, (lbl, v) in enumerate(zip(labels, values)):
        x = pad_x + gap + i * (bar_w + gap)
        bar_h = chart_h * v / max_v
        y = h - pad_bot - bar_h
        d.rectangle([x, y, x + bar_w, h - pad_bot], fill=(80, 120, 200))
        text = str(v)
        bbox = d.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        d.text((x + bar_w / 2 - tw / 2, y - 24), text, fill=axis, font=font)
        bbox = d.textbbox((0, 0), lbl, font=font)
        tw = bbox[2] - bbox[0]
        d.text((x + bar_w / 2 - tw / 2, h - pad_bot + 6), lbl, fill=axis, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gen_multi_line_text(lines, size=(640, 360)):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    w, h = size
    font = None
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(f):
            try:
                font = ImageFont.truetype(f, 32)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    pad = 40
    line_h = (h - 2 * pad) / max(1, len(lines))
    for i, line in enumerate(lines):
        y = pad + i * line_h + line_h * 0.25
        d.text((pad, y), line, fill=(20, 20, 20), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----- grader -----

def _grade_keyword(content, expected_any):
    c = (content or "").lower().strip().rstrip(".!?,;:")
    expected_lower = [e.lower().strip() for e in expected_any]
    if c in expected_lower:
        return True
    return any(e in c for e in expected_lower)


# ----- suites -----

def bench_count(model_id):
    print("  [count]")
    tests = [
        {"n": 1, "expected": ["1", " one", "one "]},
        {"n": 3, "expected": ["3", "three"]},
        {"n": 5, "expected": ["5", "five"]},
        {"n": 7, "expected": ["7", "seven"]},
    ]
    correct = 0
    details = []
    for t in tests:
        png = _gen_count_scene(t["n"])
        r = _chat_with_image(model_id,
            "How many dots are in this image? Answer with just the number.",
            png, max_tokens=80)
        ok = _grade_keyword(r["content"], t["expected"])
        correct += int(ok)
        snippet = (r["content"] or "").strip().replace("\n", " ")[:50]
        details.append({"n": t["n"], "ok": ok, "response": snippet})
        print(f"    n={t['n']}: {'PASS' if ok else 'FAIL'}  \"{snippet}\"")
    pct = 100 * correct / len(tests)
    print(f"    -> {correct}/{len(tests)} = {pct:.1f}%")
    return {"correct": correct, "total": len(tests), "pct": round(pct, 1), "details": details}


def bench_spatial(model_id):
    print("  [spatial]")
    tests = [
        ("circle", "square", "left", ["circle"]),
        ("triangle", "star", "right", ["star"]),
        ("square", "triangle", "left", ["square"]),
        ("star", "circle", "right", ["circle"]),
    ]
    correct = 0
    details = []
    for left, right, ask, exp in tests:
        png = _gen_spatial_scene(left, right)
        r = _chat_with_image(model_id,
            f"This image has two shapes. What shape is on the {ask}? Answer with just the shape name.",
            png, max_tokens=80)
        ok = _grade_keyword(r["content"], exp)
        correct += int(ok)
        snippet = (r["content"] or "").strip().replace("\n", " ")[:50]
        details.append({"left": left, "right": right, "ask": ask, "ok": ok, "response": snippet})
        print(f"    {left}|{right} ask_{ask}: {'PASS' if ok else 'FAIL'}  \"{snippet}\"")
    pct = 100 * correct / len(tests)
    print(f"    -> {correct}/{len(tests)} = {pct:.1f}%")
    return {"correct": correct, "total": len(tests), "pct": round(pct, 1), "details": details}


def bench_chart(model_id):
    print("  [chart]")
    tests = [
        {"labels": ["A", "B", "C", "D"], "values": [3, 7, 2, 5],
         "q": "Which bar has the highest value? Answer with just the letter.",
         "expected": ["b"]},
        {"labels": ["Mon", "Tue", "Wed", "Thu", "Fri"], "values": [10, 15, 8, 20, 12],
         "q": "Which day has the highest value? Answer with just the day name.",
         "expected": ["thu"]},
        {"labels": ["Red", "Blue", "Green"], "values": [25, 60, 40],
         "q": "What is the value of the Blue bar? Answer with just the number.",
         "expected": ["60"]},
    ]
    correct = 0
    details = []
    for t in tests:
        png = _gen_bar_chart(t["labels"], t["values"])
        r = _chat_with_image(model_id, t["q"], png, max_tokens=80)
        ok = _grade_keyword(r["content"], t["expected"])
        correct += int(ok)
        snippet = (r["content"] or "").strip().replace("\n", " ")[:50]
        details.append({"q": t["q"][:40], "ok": ok, "response": snippet})
        print(f"    {t['q'][:48]}: {'PASS' if ok else 'FAIL'}  \"{snippet}\"")
    pct = 100 * correct / len(tests)
    print(f"    -> {correct}/{len(tests)} = {pct:.1f}%")
    return {"correct": correct, "total": len(tests), "pct": round(pct, 1), "details": details}


def bench_layout(model_id):
    print("  [layout]")
    tests = [
        {"lines": ["Apple", "Banana", "Cherry"],
         "q": "What is the second word in the image?",
         "expected": ["banana"]},
        {"lines": ["Total: $42", "Tax: $3", "Tip: $7"],
         "q": "What is the total amount? Answer with the number.",
         "expected": ["42"]},
        {"lines": ["Status: Ready", "Code: 200", "User: admin"],
         "q": "What is the code? Answer with just the number.",
         "expected": ["200"]},
    ]
    correct = 0
    details = []
    for t in tests:
        png = _gen_multi_line_text(t["lines"])
        r = _chat_with_image(model_id, t["q"], png, max_tokens=80)
        ok = _grade_keyword(r["content"], t["expected"])
        correct += int(ok)
        snippet = (r["content"] or "").strip().replace("\n", " ")[:50]
        details.append({"lines": t["lines"], "q": t["q"], "ok": ok, "response": snippet})
        print(f"    {t['q'][:48]}: {'PASS' if ok else 'FAIL'}  \"{snippet}\"")
    pct = 100 * correct / len(tests)
    print(f"    -> {correct}/{len(tests)} = {pct:.1f}%")
    return {"correct": correct, "total": len(tests), "pct": round(pct, 1), "details": details}


# ----- driver -----

SUITES = [
    ("count", bench_count),
    ("spatial", bench_spatial),
    ("chart", bench_chart),
    ("layout", bench_layout),
]


def run_one(model_id, only=None, skip=None):
    only = only or set()
    skip = skip or set()
    print(f"\n=== {model_id} ===")
    if not _is_multimodal(model_id):
        print("  skipped: not a multimodal model (no vision artifacts and no name hint)")
        return None
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
    if not _ensure_pillow():
        print("ERROR: Pillow not installed; apt install python3-pil", file=sys.stderr)
        sys.exit(2)
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
