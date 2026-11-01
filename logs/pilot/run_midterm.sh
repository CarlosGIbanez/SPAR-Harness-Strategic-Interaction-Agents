#!/usr/bin/env bash
# Midterm runs: result 3 (arm 1 x3 vs arm 3 x3, lottery, nothing), then the
# nice-to-haves. Reuses run_eval/watched from run_pilot.sh via source of the
# shared prefix; evals requested on the command line only.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export OPENAI_API_KEY="$LITELLM_API_KEY" OPENAI_BASE_URL="$LITELLM_BASE_URL/v1"
M=openai-api/litellm/claude-sonnet-5-5
PF="private_fields=[capability,cost_of_conflict]"
RUNS=logs/pilot/RUNS.md
COMMON=(-T a=model -T b=model -T n_episodes=20 -T "$PF"
        --model-role a=$M --model-role b=$M --display plain --max-connections 4 --timeout 120 --max-retries 6)
STALL=${STALL:-300}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}

watched() {
  local dir=$1; shift
  "$@" > "$dir/stdout.txt" 2>&1 &
  local pid=$! last=0 idle=0
  while kill -0 $pid 2>/dev/null; do
    sleep 15
    local now; now=$(stat -f %m "$dir"/*.eval 2>/dev/null | sort -n | tail -1); now=${now:-0}
    if [ "$now" != "$last" ]; then last=$now; idle=0; else idle=$((idle + 15)); fi
    if [ $idle -ge "$STALL" ]; then echo "stalled, killing"; pkill -P $pid; kill $pid; wait $pid 2>/dev/null; return 2; fi
  done
  wait $pid
}

run_eval() {
  local name=$1; shift
  local dir=logs/pilot/$name
  [ -f "$dir/done" ] && { echo "skip $name (done)"; return; }
  mkdir -p "$dir"
  local cmd="uv run inspect eval src/spar_harness/task.py@crisis_bargaining ${COMMON[*]} $* --log-dir $dir"
  echo ">> $name"; local t0=$SECONDS attempts=0
  rm -f "$dir"/*.eval
  watched "$dir" uv run inspect eval src/spar_harness/task.py@crisis_bargaining "${COMMON[@]}" "$@" --log-dir "$dir" || true
  while ! uv run python -c "
import glob,os,sys; from inspect_ai.log import read_eval_log
fs=sorted(glob.glob('$dir/*.eval'), key=os.path.getmtime)
l=read_eval_log(fs[-1]); sys.exit(0 if l.status=='success' else 1)" 2>/dev/null; do
    attempts=$((attempts + 1)); [ $attempts -gt "$MAX_ATTEMPTS" ] && { echo "giving up on $name"; printf '| %s | `%s` | `%s` | %ds (GAVE UP after %d stall retries) |\n' "$name" "$cmd" "$dir" "$((SECONDS - t0))" "$attempts" >> "$RUNS"; return 1; }
    echo "   retry $attempts"
    watched "$dir" uv run inspect eval-retry "$(ls -t "$dir"/*.eval | head -1)" --log-dir "$dir" --display plain --max-connections 4 --timeout 120 || true
  done
  tail -3 "$dir/stdout.txt"
  local secs=$((SECONDS - t0))
  touch "$dir/done"
  printf '| %s | `%s` | `%s` | %ds (%d stall retries) |\n' "$name" "$cmd" "$(ls -t "$dir"/*.eval | head -1)" "$secs" "$attempts" >> "$RUNS"
}

case "${1:-}" in
  arm1) run_eval "result3_arm1_rep$2" -T version=lottery -T no_deal=nothing ;;
  arm3) run_eval "result3_arm3_rep$2" -T version=lottery -T no_deal=nothing -T commitment_a=2 -T commitment_b=2 ;;
  payment) MAX_ATTEMPTS=0 run_eval "result2_payment_nothing" -T version=payment -T no_deal=nothing -T n_episodes=10 ;;
  statusquo) MAX_ATTEMPTS=0 run_eval "result2_lottery_statusquo" -T version=lottery -T no_deal=status_quo -T status_quo=0.5 -T n_episodes=10 ;;
  *) echo "usage: $0 arm1|arm3 REP | payment | statusquo"; exit 1 ;;
esac
