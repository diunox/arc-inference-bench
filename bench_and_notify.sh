#!/bin/bash
# Wrapper: run bench_compare on supplied models, then compare_runs against ALL
# results, then send a truncated summary to Telegram.
#
# Usage:
#   bench_and_notify.sh OpenVINO/Qwen3-14B-int4-ov OpenVINO/gemma-3-12b-it-int4-ov
#   bench_and_notify.sh --only=multimodal OpenVINO/gemma-3-12b-it-int4-ov

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
log="/tmp/bench-and-notify-$ts.log"
exec > >(tee "$log") 2>&1

started=$(date +%s)
echo "=== bench_compare.py $* ==="
python3 -u bench_compare.py "$@"
bench_exit=$?

if [ $bench_exit -ne 0 ]; then
    msg="<b>Benchmark FAILED</b>
exit=$bench_exit | duration=$((($(date +%s) - started) / 60))m
Args: <code>$*</code>
Log: <code>$log</code> on $(hostname)"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" -d parse_mode="HTML" \
        --data-urlencode "text=${msg}" > /dev/null
    exit $bench_exit
fi

echo ""
echo "=== compare_runs.py ==="
report=$(python3 compare_runs.py results/compare/*.json)
echo "$report"

# Telegram caps messages at 4096 chars. Trim from the middle if too long.
if [ ${#report} -gt 3700 ]; then
    head_part=$(echo "$report" | head -c 1800)
    tail_part=$(echo "$report" | tail -c 1500)
    body="$head_part

[...truncated, full report at $log on $(hostname)...]

$tail_part"
else
    body="$report"
fi

minutes=$((($(date +%s) - started) / 60))
header="<b>Benchmark complete</b> (${minutes}m on $(hostname))
Args: <code>$*</code>
"
full="${header}
<pre>${body}</pre>"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" -d parse_mode="HTML" \
    --data-urlencode "text=${full}" > /dev/null

echo "Telegram sent."
