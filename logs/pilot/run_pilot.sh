#!/usr/bin/env bash
# The F3 pilot: result 1 (floors only), result 2 (baseline, arm 1) and
# result 3 (arm 1 vs arm 3 with level-2 floors on both seats, three repeats).
# Every Inspect log goes under logs/pilot/<name>/; RUNS.md records each
# command, its log path and wall time.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export OPENAI_API_KEY="$LITELLM_API_KEY" OPENAI_BASE_URL="$LITELLM_BASE_URL/v1"

M=openai-api/litellm/claude-sonnet-5-5
PF="private_fields=[capability,cost_of_conflict]"
RUNS=logs/pilot/RUNS.md
COMMON=(-T a=model -T b=model -T n_episodes=20 -T "$PF"
        --model-role a=$M --model-role b=$M --display plain --max-connections 4 --timeout 120 --max-retries 6)

[ -f "$RUNS" ] || printf '# Pilot runs\n\nModel `%s`, both seats, no reasoning tokens.\n\n| run | command | log | wall time |\n|---|---|---|---|\n' "$M" > "$RUNS"

# Runs stall now and then (the HTTP connection to the proxy sits open without
# progress), so each eval runs under a watchdog: if the log file has not changed
# for STALL seconds the process is killed and the eval resumed with eval-retry.
STALL=${STALL:-300}

watched() {  # cmd...  -> runs it, kills on stall, returns 0 on success
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

run_eval() {  # name, extra -T args...
  local name=$1; shift
  local dir=logs/pilot/$name
  [ -f "$dir/done" ] && { echo "skip $name (done)"; return; }
  mkdir -p "$dir"
  local cmd="uv run inspect eval src/spar_harness/task.py@crisis_bargaining ${COMMON[*]} $* --log-dir $dir"
  echo ">> $name"; local t0=$SECONDS attempts=0
  rm -f "$dir"/*.eval
  watched "$dir" uv run inspect eval src/spar_harness/task.py@crisis_bargaining "${COMMON[@]}" "$@" --log-dir "$dir" || true
  while ! uv run python -c "
import glob,sys; from inspect_ai.log import read_eval_log
l=read_eval_log(glob.glob('$dir/*.eval')[0]); sys.exit(0 if l.status=='success' else 1)" 2>/dev/null; do
    attempts=$((attempts + 1)); [ $attempts -gt 6 ] && { echo "giving up on $name"; break; }
    echo "   retry $attempts"
    watched "$dir" uv run inspect eval-retry "$dir"/*.eval --display plain --max-connections 4 --timeout 120 || true
  done
  tail -3 "$dir/stdout.txt"
  local secs=$((SECONDS - t0))
  touch "$dir/done"
  printf '| %s | `%s` | `%s` | %ds (%d stall retries) |\n' "$name" "$cmd" "$(ls "$dir"/*.eval)" "$secs" "$attempts" >> "$RUNS"
}

# Result 1: floors only
if [ ! -f logs/pilot/result1_floors.csv ]; then
  cmd="uv run python scripts/commitment_floors.py --model $M --levels 1 2 3 4 5 --versions strict lottery payment --seeds 20 --no-deal nothing --private-fields capability cost_of_conflict --concurrency 8 --out logs/pilot/result1_floors.csv"
  echo ">> result1"; t0=$SECONDS
  $cmd 2> logs/pilot/result1_calls.log | tee logs/pilot/result1_summary.txt
  printf '| result1_floors | `%s` | `logs/pilot/result1_floors.csv` (calls: `result1_calls.log`) | %ds |\n' "$cmd" "$((SECONDS - t0))" >> "$RUNS"
fi

# Result 2: baseline arm 1
for v in strict lottery payment; do
  run_eval "result2_${v}_nothing" -T version=$v -T no_deal=nothing
done
run_eval "result2_lottery_statusquo" -T version=lottery -T no_deal=status_quo -T status_quo=0.5

# Result 3: lottery, nothing; arm 1 x3 and arm 3 (level 2 both) x3
for r in 1 2 3; do
  run_eval "result3_arm1_rep$r" -T version=lottery -T no_deal=nothing
  run_eval "result3_arm3_rep$r" -T version=lottery -T no_deal=nothing -T commitment_a=2 -T commitment_b=2
done

echo ">> seed_diff"
uv run python scripts/seed_diff.py --a logs/pilot/result3_arm1_rep*/*.eval --b logs/pilot/result3_arm3_rep*/*.eval --measure fight --delta 0.05 | tee logs/pilot/result3_seed_diff.txt
echo DONE
