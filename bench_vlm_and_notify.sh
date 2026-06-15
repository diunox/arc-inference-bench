#!/bin/bash
# Wrapper: run bench_vlm.py on supplied multimodal models, then send a
# truncated summary to Telegram. Mirrors bench_and_notify.sh.
#
# Usage:
#   bench_vlm_and_notify.sh OpenVINO/gemma-3-12b-it-int4-ov OpenVINO/Qwen3.5-9B-int4-ov
#   bench_vlm_and_notify.sh --only=chart,layout OpenVINO/gemma-3-12b-it-int4-ov

set -uo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${TELEGRAM_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
    echo "ERROR: TELEGRAM_TOKEN/TELEGRAM_CHAT_ID not set (looked in ./.env)" >&2
    exit 2
fi

ts=$(date +%Y%m%d-%H%M%S)
log="/tmp/bench-vlm-$ts.log"
exec > >(tee "$log") 2>&1

started=$(date +%s)
echo "=== bench_vlm.py $* ==="
python3 -u bench_vlm.py "$@"
bench_exit=$?

minutes=$((($(date +%s) - started) / 60))

if [ $bench_exit -ne 0 ]; then
    msg="<b>VLM Benchmark FAILED</b>
exit=$bench_exit | duration=${minutes}m
Args: <code>$*</code>
Log: <code>$log</code> on $(hostname)"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" -d parse_mode="HTML" \
        --data-urlencode "text=${msg}" > /dev/null
    exit $bench_exit
fi

# Summarize the per-model JSON files written this run (look at results/vlm/ for
# files newer than the start time).
echo ""
echo "=== summary ==="
report=$(python3 - <<PY
import glob, json, os, time
started = ${started}
files = [f for f in glob.glob("results/vlm/*.json") if os.path.getmtime(f) >= started]
files.sort(key=os.path.getmtime)
if not files:
    print("(no VLM result files written this run)")
else:
    suites = ["count", "spatial", "chart", "layout"]
    # header
    name_w = max(len(os.path.basename(f).rsplit("_", 1)[0]) for f in files)
    print(f"{'model'.ljust(name_w)}  " + "  ".join(s.ljust(8) for s in suites) + "  weighted")
    for f in files:
        d = json.load(open(f))
        short = os.path.basename(f).rsplit("_", 1)[0]
        cells = []
        total_c, total_t = 0, 0
        for s in suites:
            v = d.get(s)
            if v and "pct" in v:
                cells.append(f"{v['pct']:5.1f}%")
                total_c += v.get("correct", 0)
                total_t += v.get("total", 0)
            else:
                cells.append("  --   ")
        wpct = (100 * total_c / total_t) if total_t else 0
        print(f"{short.ljust(name_w)}  " + "  ".join(c.ljust(8) for c in cells) + f"  {wpct:5.1f}%")
PY
)
echo "$report"

# Telegram caps at 4096 chars; trim if necessary.
if [ ${#report} -gt 3700 ]; then
    head_part=$(echo "$report" | head -c 1800)
    tail_part=$(echo "$report" | tail -c 1500)
    body="$head_part

[...truncated, full report at $log on $(hostname)...]

$tail_part"
else
    body="$report"
fi

header="<b>VLM Benchmark complete</b> (${minutes}m on $(hostname))
Args: <code>$*</code>
"
full="${header}
<pre>${body}</pre>"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" -d parse_mode="HTML" \
    --data-urlencode "text=${full}" > /dev/null

echo "Telegram sent."
