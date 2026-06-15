#!/usr/bin/env python3
"""Side-by-side comparison reporter for bench_compare.py outputs.

Usage:
    python3 compare_runs.py results/compare/*.json

For each model, picks the *latest* result file by mtime. Prints a markdown
comparison report covering all test categories.

Higher-is-better metrics get a ✓ next to the winner.
Lower-is-better (TTFT, latency) get ✓ next to the winner.
"""

import glob
import json
import os
import sys
from collections import defaultdict


def short_label(d):
    # Prefer the short_name we saved; fall back to model_id leaf
    return d.get("short_name") or d.get("model_id", "?").split("/")[-1]


def load_latest_per_model(paths):
    """Group by short_name; merge tests across files per model.

    For each test category, the LATEST non-skipped result wins.
    This lets partial re-runs (e.g. --only=arithmetic,code) slot in
    on top of an earlier comprehensive run without losing other test
    data. Model metadata (model_id, started_at) comes from the most
    recent file.
    """
    by_model = defaultdict(list)
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"WARN: skipping {p}: {e}", file=sys.stderr)
            continue
        by_model[short_label(d)].append((os.path.getmtime(p), d, p))

    merged = {}
    for name, runs in by_model.items():
        runs.sort()  # oldest first
        # Start with oldest as base; overlay newer test results on top.
        # A "real" test result has either accuracy_pct/correct (correctness) or
        # numeric values (perf). A skipped result has {"skipped": True}.
        base = runs[0][1].copy()
        base.setdefault("tests", {})
        base["tests"] = dict(base["tests"])
        merge_sources = [(os.path.basename(p), set()) for _, _, p in runs]
        for mtime, d, p in runs[1:]:
            for test_name, test_data in (d.get("tests") or {}).items():
                # Overlay if the newer one has real data (not a skip).
                if isinstance(test_data, dict) and not test_data.get("skipped"):
                    base["tests"][test_name] = test_data
                    # Track origin
                    for fname, ts in merge_sources:
                        if fname == os.path.basename(p):
                            ts.add(test_name)
            # Update the timestamp / metadata to the most recent
            base["model_id"] = d.get("model_id", base.get("model_id"))
            base["started_at"] = d.get("started_at", base.get("started_at"))
        base["_merged_from"] = [os.path.basename(p) for _, _, p in runs]
        merged[name] = base
    return merged


def fmt(v, w=8):
    if v is None: return "—".rjust(w)
    if isinstance(v, float): return f"{v:.2f}".rjust(w)
    return str(v).rjust(w)


def winner_mark(values, higher_better=True):
    """Return list of strings with ✓ appended to the best, '' to others."""
    valid = [v for v in values if isinstance(v, (int, float))]
    if not valid: return ["" for _ in values]
    best = max(valid) if higher_better else min(valid)
    out = []
    for v in values:
        if isinstance(v, (int, float)) and v == best:
            out.append(" ✓")
        else:
            out.append("")
    return out


def section_header(s):
    return f"\n### {s}\n"


def render_table(headers, rows):
    """rows is list of [str, ...]; first column is the row label."""
    widths = [max(len(headers[i]),
                  max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]
    sep = "|".join("-" * (w + 2) for w in widths)
    def line(cells): return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(headers), "|" + sep + "|"] + [line(r) for r in rows])


def report(latest):
    models = sorted(latest.keys())
    if len(models) < 1:
        print("No results to compare.")
        return
    out = []
    out.append("# Model Comparison Report")
    out.append(f"\nModels: {', '.join(models)}\n")
    out.append(f"Hardware: Intel Arc A770 16GB via OpenVINO Model Server (OVMS)\n")
    # Mark the timestamps + merge provenance
    out.append("| Model | Latest run | Merged from |")
    out.append("|-------|------------|-------------|")
    for m in models:
        sources = latest[m].get("_merged_from", [])
        src_str = f"{len(sources)} run(s)" if len(sources) > 1 else (sources[0] if sources else "?")
        out.append(f"| {m} | {latest[m].get('started_at', '?')} | {src_str} |")

    # ---- Throughput sweep ----
    out.append(section_header("Throughput (tok/s, output-length sweep)"))
    headers = ["Output tokens"] + models
    lengths = ["128", "256", "512", "1024", "2048"]
    rows = []
    for L in lengths:
        vals = []
        for m in models:
            t = latest[m].get("tests", {}).get("throughput", {})
            v = (t.get(L) or {}).get("median_tok_s")
            vals.append(v)
        marks = winner_mark(vals, higher_better=True)
        rows.append([L] + [f"{vals[i]:.2f}{marks[i]}" if isinstance(vals[i], (int, float)) else "—" for i in range(len(models))])
    out.append(render_table(headers, rows))

    # ---- TTFT ----
    out.append(section_header("TTFT (Time To First Token, lower is better)"))
    headers = ["Metric"] + models
    vals = [(latest[m].get("tests", {}).get("ttft", {}) or {}).get("overall_median_ms") for m in models]
    marks = winner_mark(vals, higher_better=False)
    row = ["Median TTFT (ms)"] + [f"{vals[i]:.1f}{marks[i]}" if isinstance(vals[i], (int, float)) else "—" for i in range(len(models))]
    out.append(render_table(headers, [row]))

    # ---- Long context ----
    out.append(section_header("Long-context generation throughput (tok/s)"))
    headers = ["Approx ctx tokens"] + models
    rows = []
    for tgt in ["2000", "8000", "16000", "32000"]:
        vals = []
        for m in models:
            t = (latest[m].get("tests", {}).get("long_context", {}) or {}).get(tgt) or {}
            vals.append(t.get("gen_tok_s"))
        marks = winner_mark(vals, higher_better=True)
        rows.append([tgt] + [f"{vals[i]:.2f}{marks[i]}" if isinstance(vals[i], (int, float)) else "—" for i in range(len(models))])
    out.append(render_table(headers, rows))

    # ---- Prompt processing ----
    out.append(section_header("Prompt processing speed (input tok/s, higher better)"))
    headers = ["Input tokens"] + models
    rows = []
    for tgt in ["4000", "8000"]:
        vals = []
        for m in models:
            t = (latest[m].get("tests", {}).get("prompt_processing", {}) or {}).get(tgt) or {}
            vals.append(t.get("prompt_tok_per_s"))
        marks = winner_mark(vals, higher_better=True)
        rows.append([tgt] + [f"{vals[i]:.1f}{marks[i]}" if isinstance(vals[i], (int, float)) else "—" for i in range(len(models))])
    out.append(render_table(headers, rows))

    # ---- Correctness suite ----
    out.append(section_header("Verifiable correctness (accuracy %, higher better)"))
    accuracy_tests = [
        ("gsm8k", "GSM8K math (20 problems)"),
        ("arithmetic", "Multi-step arithmetic (10)"),
        ("logic", "Logic puzzles (10)"),
        ("code", "Code generation (10 HumanEval-style)"),
        ("instruction", "Instruction following (8)"),
        ("needle", "Needle-in-haystack (15 trials @ 4k ctx)"),
        ("rag", "RAG comprehension (8 questions)"),
        ("multihop", "Multi-hop reasoning (5)"),
        ("knowledge", "Knowledge accuracy (10)"),
        ("multimodal", "Multimodal vision (10 generated images)"),
    ]
    headers = ["Test"] + models
    rows = []
    for key, label in accuracy_tests:
        vals = []
        raws = []
        for m in models:
            t = latest[m].get("tests", {}).get(key, {})
            pct = t.get("accuracy_pct")
            raw = f"{t.get('correct')}/{t.get('total')}" if t.get("total") else "—"
            vals.append(pct); raws.append(raw)
        marks = winner_mark(vals, higher_better=True)
        rows.append([label] + [
            (f"{vals[i]:.1f}% ({raws[i]}){marks[i]}" if isinstance(vals[i], (int, float)) else "—")
            for i in range(len(models))
        ])
    out.append(render_table(headers, rows))

    # ---- Hallucination (refusal is good) ----
    out.append(section_header("Anti-hallucination (refusal % on unanswerable, higher better)"))
    headers = ["Metric"] + models
    vals = []
    for m in models:
        t = latest[m].get("tests", {}).get("hallucination", {})
        vals.append(t.get("refusal_pct"))
    marks = winner_mark(vals, higher_better=True)
    row = ["Refusal rate"] + [
        f"{vals[i]:.1f}%{marks[i]}" if isinstance(vals[i], (int, float)) else "—" for i in range(len(models))
    ]
    out.append(render_table(headers, [row]))

    # ---- Overall score (text-only, excludes multimodal for fairness) ----
    out.append(section_header("Overall text-correctness score (multimodal excluded for fairness)"))
    headers = ["Metric"] + models
    correctness_keys = [k for k, _ in accuracy_tests if k != "multimodal"]
    rows = []
    overall_vals = []
    for m in models:
        tests = latest[m].get("tests", {})
        # weighted by total problems in each test
        total_correct = 0; total = 0
        for k in correctness_keys:
            t = tests.get(k, {})
            if isinstance(t.get("correct"), int) and t.get("total"):
                total_correct += t["correct"]; total += t["total"]
        pct = (total_correct / total * 100) if total else None
        overall_vals.append(pct)
    marks = winner_mark(overall_vals, higher_better=True)
    out.append(render_table(headers, [["Weighted-avg correctness %"] + [
        f"{overall_vals[i]:.1f}%{marks[i]}" if isinstance(overall_vals[i], (int, float)) else "—"
        for i in range(len(models))
    ]]))

    print("\n".join(out))


def main():
    if len(sys.argv) < 2:
        print("usage: compare_runs.py results/compare/*.json")
        sys.exit(2)
    paths = []
    for arg in sys.argv[1:]:
        paths.extend(glob.glob(arg))
    if not paths:
        print("No matching files.", file=sys.stderr)
        sys.exit(1)
    latest = load_latest_per_model(paths)
    report(latest)


if __name__ == "__main__":
    main()
