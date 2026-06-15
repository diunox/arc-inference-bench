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
  diff     : multi-image difference detection (3 paired scenes) [v1.1]
  frames   : temporal reasoning across a frame sequence (5 scenarios) [v1.3]

v1.1 (2026-06-14):
  - Grader now extracts the final answer from CoT-prefixed responses
    (recognises \\boxed{X}, **X**, "Answer: X", or the trailing sentence)
    before substring-matching. Reduces false positives where a CoT model
    mentions the expected token inside its reasoning but never delivers
    the answer cleanly.
  - Multi-image input via _chat_with_images for the new diff suite.

v1.2 (2026-06-14):
  - Grader now strips the <think>...</think> reasoning block before
    extraction — Qwen3.5 and similar models wrap CoT in those tags and
    emit the actual answer only AFTER </think>. The v1.1 regex was
    matching "answer:" mentions inside the CoT and capturing markup
    instead of the real answer.
  - Default max_tokens raised to 256 (single-image) / 320 (multi-image)
    so CoT-heavy models have room to finish reasoning before OVMS hits
    the length limit. Was 80/120 — way too tight for <think> models.
  - Stored response snippet in details bumped from 50/80 to 500 chars
    so post-mortem debugging actually shows the model's output.

v1.3 (2026-06-14):
  - New `frames` suite: temporal/sequence reasoning across N frames sent
    as a multi-image sequence (motion direction, color transition, frame
    counter, shape appearance). Originally scoped as a video suite, but
    OVMS does not accept `type: video` / `type: video_url` content via
    /v3/chat/completions — it returns "Unsupported content type" from the
    LLMExecutor calculator. Frame-sequence is the practical equivalent
    (most "video" VLM benchmarks sample N frames anyway).

Each suite produces an accuracy score (correct/total). Results go to
results/vlm/<short_name>_<unix_ts>.json. Restores prior model on exit.
"""

import base64
import io
import json
import math
import os
import re
import sys
import time
import urllib.request

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


def _chat_with_images(model_id, prompt, pngs, max_tokens=80):
    """Send one prompt + N images to the chat endpoint. pngs is a list of bytes."""
    import time as _t
    content = [{"type": "text", "text": prompt}]
    for png in pngs:
        content.append({"type": "image_url", "image_url": {"url": _data_uri(png)}})
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
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


def _chat_with_image(model_id, prompt, png_bytes, max_tokens=80):
    """Back-compat single-image wrapper. Honors BENCH_VLM_MAX_TOKENS env override."""
    override = int(os.environ.get("BENCH_VLM_MAX_TOKENS", "0") or "0")
    if override:
        max_tokens = override
    return _chat_with_images(model_id, prompt, [png_bytes], max_tokens=max_tokens)

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


def _gen_scene(shapes, size=(384, 384), bg=(245, 245, 245)):
    """shapes is a list of (shape_name, color_rgb, (cx, cy), r). Renders to PNG bytes."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    for shape, color, (cx, cy), r in shapes:
        _draw_shape(d, cx, cy, shape, r, color)
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

_BOLDED_FINAL = re.compile(r'\*\*([^*\n]{1,80})\*\*\s*\.?\s*$')
_ANSWER_PREFIX = re.compile(
    r'(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:\-]?\s*(.+?)(?:\.|$|\n)',
    re.IGNORECASE,
)
_BOXED = re.compile(r'\\boxed\{([^}]+)\}')


def _extract_final_answer(content):
    """Best-effort extraction of the final answer from a (possibly CoT-prefixed) response.

    For models that wrap reasoning in <think>...</think> blocks (e.g. Qwen3.5),
    everything BEFORE the last </think> is reasoning — only what follows is the
    real answer. Strip that first so later heuristics don't false-positive on
    in-CoT phrases like "the answer is".

    Then try, in order: \\boxed{X}, **X** trailing, "Answer: X" / "the answer is X",
    last non-empty sentence. Returns the cleaned candidate (still lowercased
    by the caller). Empty string on no signal.
    """
    c = (content or "").strip()
    if not c:
        return ""
    # Drop the reasoning block: keep only what comes after the last </think>.
    think_end = c.rfind("</think>")
    if think_end >= 0:
        c = c[think_end + len("</think>"):].strip()
        # If the model emitted </think> with nothing useful after, give up — let
        # the caller fall back to whole-content scan rather than returning "".
        if not c:
            return ""
    m = _BOXED.search(c)
    if m:
        return m.group(1).strip()
    m = _BOLDED_FINAL.search(c)
    if m:
        return m.group(1).strip()
    m = _ANSWER_PREFIX.search(c)
    if m:
        cand = m.group(1).strip().rstrip(".!?,;:")
        if cand:
            return cand
    lines = [l.strip() for l in c.split("\n") if l.strip()]
    if not lines:
        return c
    last_line = lines[-1]
    sentences = re.split(r'(?<=[.!?])\s+', last_line)
    return (sentences[-1] if sentences else last_line).strip().rstrip(".!?,;:")


def _grade_keyword(content, expected_any):
    """v1.1 grader: prefers the extracted final answer over whole-content substring.

    Pass conditions (any one is enough):
      - Final-answer extract exactly equals an expected token (after lower/strip).
      - Final-answer extract contains an expected token as substring.
    If the final-answer extract is empty (model returned nothing useful), fall
    back to whole-content substring as a last resort.
    """
    expected_lower = [e.lower().strip() for e in expected_any]
    final = _extract_final_answer(content).lower().strip().rstrip(".!?,;:")
    if final:
        if final in expected_lower:
            return True
        if any(e in final for e in expected_lower):
            return True
        return False  # strict: extraction found something but it didn't match
    # Last resort: substring on whole response (e.g. very terse models that
    # returned just one token without punctuation/structure to extract)
    raw = (content or "").lower().strip().rstrip(".!?,;:")
    if raw in expected_lower:
        return True
    return any(e in raw for e in expected_lower)


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
            png, max_tokens=256)
        ok = _grade_keyword(r["content"], t["expected"])
        correct += int(ok)
        snippet = (r["content"] or "").strip().replace("\n", " ")[:500]
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
            png, max_tokens=256)
        ok = _grade_keyword(r["content"], exp)
        correct += int(ok)
        snippet = (r["content"] or "").strip().replace("\n", " ")[:500]
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
        r = _chat_with_image(model_id, t["q"], png, max_tokens=256)
        ok = _grade_keyword(r["content"], t["expected"])
        correct += int(ok)
        snippet = (r["content"] or "").strip().replace("\n", " ")[:500]
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
        r = _chat_with_image(model_id, t["q"], png, max_tokens=256)
        ok = _grade_keyword(r["content"], t["expected"])
        correct += int(ok)
        snippet = (r["content"] or "").strip().replace("\n", " ")[:500]
        details.append({"lines": t["lines"], "q": t["q"], "ok": ok, "response": snippet})
        print(f"    {t['q'][:48]}: {'PASS' if ok else 'FAIL'}  \"{snippet}\"")
    pct = 100 * correct / len(tests)
    print(f"    -> {correct}/{len(tests)} = {pct:.1f}%")
    return {"correct": correct, "total": len(tests), "pct": round(pct, 1), "details": details}


def _gen_motion_frames(n_frames=4, direction="right", size=(256, 256),
                       bg=(240, 240, 240), color=(40, 80, 200), r=30):
    """N frames showing a dot moving in `direction`. Returns list of PNG bytes."""
    from PIL import Image, ImageDraw
    W, H = size
    margin = 40
    out = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        if direction == "right":
            cx, cy = margin + t * (W - 2 * margin), H // 2
        elif direction == "left":
            cx, cy = (W - margin) - t * (W - 2 * margin), H // 2
        elif direction == "down":
            cx, cy = W // 2, margin + t * (H - 2 * margin)
        elif direction == "up":
            cx, cy = W // 2, (H - margin) - t * (H - 2 * margin)
        img = Image.new("RGB", size, bg)
        d = ImageDraw.Draw(img)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _gen_color_transition_frames(colors, size=(256, 256), bg=(240, 240, 240), r=80):
    """N frames showing a circle changing color through the given color sequence."""
    from PIL import Image, ImageDraw
    W, H = size
    out = []
    for c in colors:
        img = Image.new("RGB", size, bg)
        d = ImageDraw.Draw(img)
        d.ellipse([W // 2 - r, H // 2 - r, W // 2 + r, H // 2 + r], fill=c)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _gen_counter_frames(numbers, size=(256, 256), bg=(255, 255, 255)):
    """N frames each showing a single large number centered."""
    from PIL import Image, ImageDraw, ImageFont
    out = []
    font = None
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(f):
            try:
                font = ImageFont.truetype(f, 128)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    for n in numbers:
        img = Image.new("RGB", size, bg)
        d = ImageDraw.Draw(img)
        text = str(n)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((size[0] - tw) / 2 - bbox[0], (size[1] - th) / 2 - bbox[1]),
               text, fill=(20, 20, 20), font=font)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _gen_appearance_frames(appear_shape, appear_at=2, n_frames=4,
                           size=(256, 256), bg=(245, 245, 245)):
    """N frames where `appear_shape` is absent in frame 0 then visible from `appear_at` onward."""
    from PIL import Image, ImageDraw
    W, H = size
    out = []
    static = [("circle", (200, 40, 40), (80, 80), 30)]
    for i in range(n_frames):
        img = Image.new("RGB", size, bg)
        d = ImageDraw.Draw(img)
        for shape, color, (cx, cy), r in static:
            _draw_shape(d, cx, cy, shape, r, color)
        if i >= appear_at:
            _draw_shape(d, W - 80, H - 80, appear_shape, 30, (40, 100, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def bench_frames(model_id):
    """Temporal reasoning across a sequence of frames (multi-image proxy for video,
    since OVMS does not accept type=video content as of 2026-06-14)."""
    print("  [frames]")
    tests = [
        {"frames": _gen_motion_frames(4, "right"),
         "q": "These are 4 frames of a short video, in order. Which direction does the dot move?",
         "expected": ["right"]},
        {"frames": _gen_motion_frames(4, "down"),
         "q": "These are 4 frames of a short video, in order. Which direction does the dot move?",
         "expected": ["down"]},
        {"frames": _gen_color_transition_frames([(220, 30, 30), (220, 110, 30), (220, 180, 30), (220, 220, 30)]),
         "q": "These are 4 frames of a short video, in order. What color does the circle end up as in the last frame?",
         "expected": ["yellow"]},
        {"frames": _gen_counter_frames([1, 2, 5, 9]),
         "q": "These are 4 frames of a short video, in order. What is the highest number shown across all frames?",
         "expected": ["9", "nine"]},
        {"frames": _gen_appearance_frames("star", appear_at=2),
         "q": "These are 4 frames of a short video, in order. A new shape appears partway through. What shape?",
         "expected": ["star"]},
    ]
    correct = 0
    details = []
    override = int(os.environ.get("BENCH_VLM_MAX_TOKENS", "0") or "0")
    for i, t in enumerate(tests):
        try:
            r_resp = _chat_with_images(model_id, t["q"], t["frames"],
                                       max_tokens=(override or 400))
        except Exception as e:
            details.append({"i": i, "ok": False, "error": str(e)})
            print(f"    test {i}: ERROR {e}")
            continue
        ok = _grade_keyword(r_resp["content"], t["expected"])
        correct += int(ok)
        final = _extract_final_answer(r_resp["content"]).replace("\n", " ")[:80]
        details.append({"i": i, "expected": t["expected"], "ok": ok,
                        "final_extract": final,
                        "response": (r_resp["content"] or "").replace("\n", " ")[:500]})
        print(f"    test {i} ({len(t['frames'])} frames): {'PASS' if ok else 'FAIL'}  final=\"{final}\"")
    pct = 100 * correct / len(tests)
    print(f"    -> {correct}/{len(tests)} = {pct:.1f}%")
    return {"correct": correct, "total": len(tests), "pct": round(pct, 1), "details": details}


def bench_diff(model_id):
    """Multi-image suite: present two near-identical scenes and ask what changed."""
    print("  [diff]")
    R, G, B, Y = (200, 40, 40), (40, 160, 70), (40, 100, 200), (220, 200, 40)
    r = 45
    # 1. Color change: blue dot in img1 becomes yellow in img2
    img1a = _gen_scene([("circle", R, (100, 120), r),
                        ("circle", G, (260, 120), r),
                        ("circle", B, (180, 260), r)])
    img2a = _gen_scene([("circle", R, (100, 120), r),
                        ("circle", G, (260, 120), r),
                        ("circle", Y, (180, 260), r)])
    # 2. Added shape: triangle appears in img2
    img1b = _gen_scene([("square", R, (120, 180), r),
                        ("circle", B, (260, 180), r)])
    img2b = _gen_scene([("square", R, (120, 180), r),
                        ("circle", B, (260, 180), r),
                        ("triangle", G, (190, 320), r)])
    # 3. Removed shape: star is in img1 but not img2
    img1c = _gen_scene([("circle", R, (90, 100), r),
                        ("square", B, (290, 100), r),
                        ("star", Y, (90, 280), r),
                        ("triangle", G, (290, 280), r)])
    img2c = _gen_scene([("circle", R, (90, 100), r),
                        ("square", B, (290, 100), r),
                        ("triangle", G, (290, 280), r)])
    tests = [
        {"imgs": [img1a, img2a],
         "q": "These are two images. What is different about the second image compared to the first? Be brief.",
         "expected": ["yellow", "color", "blue"]},
        {"imgs": [img1b, img2b],
         "q": "These are two images. What is different about the second image compared to the first? Be brief.",
         "expected": ["triangle", "added", "extra", "new shape"]},
        {"imgs": [img1c, img2c],
         "q": "These are two images. What is different about the second image compared to the first? Be brief.",
         "expected": ["star", "missing", "removed"]},
    ]
    correct = 0
    details = []
    for i, t in enumerate(tests):
        try:
            override = int(os.environ.get("BENCH_VLM_MAX_TOKENS", "0") or "0")
            r_resp = _chat_with_images(model_id, t["q"], t["imgs"],
                                       max_tokens=(override or 320))
        except Exception as e:
            details.append({"i": i, "ok": False, "error": str(e)})
            print(f"    test {i}: ERROR {e}")
            continue
        ok = _grade_keyword(r_resp["content"], t["expected"])
        correct += int(ok)
        final = _extract_final_answer(r_resp["content"]).replace("\n", " ")[:60]
        details.append({"i": i, "expected": t["expected"], "ok": ok,
                        "final_extract": final,
                        "response": (r_resp["content"] or "").replace("\n", " ")[:500]})
        print(f"    test {i}: {'PASS' if ok else 'FAIL'}  final=\"{final}\"")
    pct = 100 * correct / len(tests)
    print(f"    -> {correct}/{len(tests)} = {pct:.1f}%")
    return {"correct": correct, "total": len(tests), "pct": round(pct, 1), "details": details}


# ----- driver -----

SUITES = [
    ("count", bench_count),
    ("spatial", bench_spatial),
    ("chart", bench_chart),
    ("layout", bench_layout),
    ("diff", bench_diff),
    ("frames", bench_frames),
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
