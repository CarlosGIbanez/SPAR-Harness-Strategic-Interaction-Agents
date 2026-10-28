# Pilot runs

Model `openai-api/litellm/claude-sonnet-5-5`, both seats, no reasoning tokens.

| run | command | log | wall time |
|---|---|---|---|
| result1_floors | `uv run python scripts/commitment_floors.py --model openai-api/litellm/claude-sonnet-5-5 --levels 1 2 3 4 5 --versions strict lottery payment --seeds 20 --no-deal nothing --private-fields capability cost_of_conflict --concurrency 8 --out logs/pilot/result1_floors.csv` | `logs/pilot/result1_floors.csv` (calls: `result1_calls.log`) | 279s |
| result2_strict_nothing | `uv run inspect eval src/spar_harness/task.py@crisis_bargaining -T a=model -T b=model -T n_episodes=20 -T private_fields=[capability,cost_of_conflict] --model-role a=openai-api/litellm/claude-sonnet-5-5 --model-role b=openai-api/litellm/claude-sonnet-5-5 --display plain --max-connections 8 -T version=strict -T no_deal=nothing --log-dir logs/pilot/result2_strict_nothing` | `logs/pilot/result2_strict_nothing/2026-10-28T05-01-03-00-00_crisis-bargaining_RxeDaCRxKDtewR2vdvVnzM.eval` | 64s |

Note: the first result2_lottery_nothing attempt hung at 32/40 samples for 13 min (stalled HTTP connections to the proxy, no request timeout). It was killed and discarded; a second attempt completed in 98 s. result2_payment_nothing then stalled at 8/40 for an hour. All later runs use `--max-connections 4 --timeout 120 --max-retries 6` under a stall watchdog (kill after 5 min without log progress, then `inspect eval-retry`).


Stopped after result2_lottery_nothing on request; result2_payment_nothing (two stalled attempts) was discarded. Elapsed wall clock including stalls: ~2 h 10 min.
