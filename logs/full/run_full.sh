#!/usr/bin/env bash
# Full-run driver: same watchdog/run_eval as scripts/run_midterm.sh, parametrised
# by model short name. usage: logs/full/run_full.sh <short> <eval> [rep]
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export OPENAI_API_KEY="$LITELLM_API_KEY" OPENAI_BASE_URL="$LITELLM_BASE_URL/v1"
short=$1; what=$2; rep=${3:-}
case $short in haiku) M=openai-api/litellm/claude-haiku-5-5;; sonnet) M=openai-api/litellm/claude-sonnet-5-5;; luna) M=openai-api/litellm/luna;; sol) M=openai-api/litellm/sol;; *) echo bad model; exit 1;; esac
PF="private_fields=[capability,cost_of_conflict]"
RUNS=logs/full/$short/RUNS.md
COMMON=(-T a=model -T b=model -T n_episodes=20 -T "$PF"
        --model-role a=$M --model-role b=$M --display plain --max-connections 4 --timeout 120 --max-retries 6)
STALL=${STALL:-300}; MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}
mkdir -p logs/full/$short
[ -f "$RUNS" ] || printf '# Full run: %s\n\nModel `%s`, both seats, no reasoning tokens.\n\n| run | command | log | wall time |\n|---|---|---|---|\n' "$short" "$M" > "$RUNS"
watched() { local dir=$1; shift; "$@" > "$dir/stdout.txt" 2>&1 & local pid=$! last=0 idle=0
  while kill -0 $pid 2>/dev/null; do sleep 15
    local now; now=$(stat -f %m "$dir"/*.eval 2>/dev/null | sort -n | tail -1); now=${now:-0}
    if [ "$now" != "$last" ]; then last=$now; idle=0; else idle=$((idle + 15)); fi
    if [ $idle -ge "$STALL" ]; then echo "stalled, killing"; pkill -P $pid; kill $pid; wait $pid 2>/dev/null; return 2; fi
  done; wait $pid; }
run_eval() { local name=$1; shift; local dir=logs/full/$short/$name
  [ -f "$dir/done" ] && { echo "skip $name (done)"; return; }
  mkdir -p "$dir"; local cmd="uv run inspect eval src/spar_harness/task.py@crisis_bargaining ${COMMON[*]} $* --log-dir $dir"
  echo ">> $short/$name"; local t0=$SECONDS attempts=0; rm -f "$dir"/*.eval
  watched "$dir" uv run inspect eval src/spar_harness/task.py@crisis_bargaining "${COMMON[@]}" "$@" --log-dir "$dir" || true
  while ! uv run python -c "
import glob,os,sys; from inspect_ai.log import read_eval_log
fs=sorted(glob.glob('$dir/*.eval'), key=os.path.getmtime)
l=read_eval_log(fs[-1]); sys.exit(0 if l.status=='success' else 1)" 2>/dev/null; do
    attempts=$((attempts + 1)); [ $attempts -gt "$MAX_ATTEMPTS" ] && { echo "giving up on $name"; printf '| %s | `%s` | `%s` | %ds (GAVE UP after %d stall retries) |\n' "$name" "$cmd" "$dir" "$((SECONDS - t0))" "$attempts" >> "$RUNS"; return 1; }
    echo "   retry $attempts"
    watched "$dir" uv run inspect eval-retry "$(ls -t "$dir"/*.eval | head -1)" --log-dir "$dir" --display plain --max-connections 4 --timeout 120 || true
  done
  local secs=$((SECONDS - t0)); touch "$dir/done"
  printf '| %s | `%s` | `%s` | %ds (%d stall retries) |\n' "$name" "$cmd" "$(ls -t "$dir"/*.eval | head -1)" "$secs" "$attempts" >> "$RUNS"; }
case "$what" in
  arm1) run_eval "arm1_rep$rep" -T version=lottery -T no_deal=nothing ;;
  arm3) run_eval "arm3_rep$rep" -T version=lottery -T no_deal=nothing -T commitment_a=2 -T commitment_b=2 ;;
  strict) run_eval "strict_nothing" -T version=strict -T no_deal=nothing ;;
  statusquo) run_eval "lottery_statusquo" -T version=lottery -T no_deal=status_quo -T status_quo=0.5 ;;
  *) echo "usage: $0 model arm1|arm3 REP | strict | statusquo"; exit 1 ;;
esac
