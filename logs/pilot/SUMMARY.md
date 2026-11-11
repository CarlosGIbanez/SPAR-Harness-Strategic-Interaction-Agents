# F3 pilot summary

Branch `indivisible-good` at 5feae1a. Model `openai-api/litellm/claude-sonnet-5-5` on both seats, no reasoning tokens, `private_fields=[capability,cost_of_conflict]`, 20 rounds max, stop chance 0.1, 20 seeds, both seat orders (40 samples per eval). Commands, log paths and wall times: `RUNS.md`. Tables built by `summarize.py`.

**First pilot stopped early on request** after result 1 (600 floor calls) and result 2 for strict and lottery at no_deal=nothing. The midterm runs (2026-10-31 and 2026-11-11) added result 1b (reworded levels 3 and 4, `result1b_table.md`), result 3 (`result3_seed_diff_*.txt`), and the payment and status-quo baselines below at 10 seeds. Reason for the slowness: evals stalled repeatedly against the proxy (an HTTP connection sits open with no progress: lottery stalled at 32/40 for 13 min; payment stalled at 8/40 for an hour and again after a restart with `--max-connections 4 --timeout 120`). `run_pilot.sh` now runs each eval under a stall watchdog with `inspect eval-retry`; resuming it will finish the remaining runs, skipping the ones marked `done`.

## Result 1: commitment only (600 calls)

Median aggressiveness (floor − fight value) by level and version:

| level | strict | lottery | payment | fidelity rate | unparseable |
|---|---|---|---|---|---|
| 1 | -0.176 | -0.110 | -0.180 | 0.94 | 0.00 |
| 2 | -0.023 | 0.002 | -0.027 | 0.60 | 0.00 |
| 3 | -0.008 | -0.006 | -0.015 | 0.77 | 0.00 |
| 4 | -0.012 | -0.002 | -0.007 | 0.43 | 0.00 |
| 5 | 0.207 | 0.123 | 0.126 | 0.88 | 0.00 |

Collision rate (alpha floor + beta floor > 1, same level and seed) per version:

| version | collision rate | n pairs |
|---|---|---|
| strict | 0.19 | 100 |
| lottery | 0.05 | 100 |
| payment | 0.05 | 100 |

Collision rate by level (all versions):

| level | collision rate |
|---|---|
| 1 | 0.00 |
| 2 | 0.00 |
| 3 | 0.03 |
| 4 | 0.02 |
| 5 | 0.43 |

## Result 2: baseline, arm 1 (no instruction), 20 seeds × both orders

| version / no-deal | fight | agreement | efficiency | mean rounds | in-range | comprehension | default rate | n |
|---|---|---|---|---|---|---|---|---|
| strict / nothing | 1.00 | 0.00 | 0.64 | 1.00 | – | 1.00 | 0.00 | 40 |
| lottery / nothing | 0.12 | 0.85 | 0.94 | 1.18 | 0.88 | 1.00 | 0.00 | 40 |
| payment / nothing (seeds 0–9) | 0.20 | 0.80 | 0.95 | 1.10 | 0.69 | 1.00 | 0.00 | 20 |
| lottery / status quo 0.5 (seeds 0–9) | 0.00 | 0.90 | 1.00 | 2.20 | 0.56 | 1.00 | 0.00 | 20 |
| lottery / nothing, seeds 0–9 only | 0.00 | 1.00 | 1.00 | – | – | – | – | 20 |

The last two rows share seeds 0–9. Lottery at status quo 0.5 and lottery with nothing both had no fights on those seeds, so this pilot cannot separate the indivisibility effect from "stalling is dominated": all five lottery/nothing fights sit in seeds 10–19, which the status-quo run did not cover. What the status-quo run does show is slower play (2.2 rounds against 1.0–1.2) and two episodes ending by random stop at the status quo, which never happens under nothing. The payment version, run once with the `--log-dir` watchdog fix, completed without a stall (its two earlier attempts stalled against the proxy); its fight rate on seeds 0–9 (0.20) is above lottery's on the same seeds (0.00) despite identical stored offers, so the payment framing is not a neutral relabelling of the split game for this model.

## Cost table

| run | samples | input tokens | output tokens | cost |
|---|---|---|---|---|
| result1_floors | 600 calls | – | – | ~$2.70 (est. $0.0045 per call; the script does not log tokens) |
| result2_strict_nothing | 40 | 115,142 | 44,094 | $0.67 |
| result2_lottery_nothing | 40 | 213,937 | 68,059 | $1.11 |
| result1b_floors (midterm, 600 calls + 354 on a stalled first attempt) | 954 calls | – | – | ~$4.30 (est.) |
| result3 arm 1 × 3 and arm 3 × 3 (midterm) | 240 | 1,579,610 | 442,719 | $7.59 (+ $9.10 on partial logs of arm3 rep3 before the watchdog fix) |
| result2_payment_nothing | 20 | 105,164 | 38,097 | $0.59 |
| result2_lottery_statusquo | 20 | 174,938 | 43,979 | $0.79 |
| **total** | | | | **≈ $26.9** (+ $9.10 lost to stalls, + $0.27 smoke tests) |

Eval cost at the proxy's rates for claude-sonnet-5-5 ($2/M in, $10/M out). Wall times per run are in RUNS.md.


## Findings and flags

1. **Strict arm 1 fight rate is 1.00, not ~0.90.** All 40 strict episodes ended in a fight in round 1: alpha fought outright in 31, and in the other 9 alpha claimed and beta fought. No episode saw a second round. With 0/0 no-deal and an indivisible good, the model never tries to concede the whole good even when its fight value is poor. Efficiency 0.64 is just the mean fight payoff relative to the pie.
2. **Lottery resolves it.** Same seeds, same prompts, only the framing changed: fight 0.12, agreement 0.85 (34 of 40, plus one random stop), 31 of 40 accepted the opening offer, 0.88 of agreements inside the true bargaining range. This is the clean F3 contrast for the midterm: strict 1.00 vs lottery 0.12 fight rate on identical seeds.
3. **Result 1, instruction levels.** Level 1 floors sit clearly below the fight value (median −0.11 to −0.18) and level 5 clearly above (+0.12 to +0.21). Levels 2, 3 and 4 all land within about 0.03 of the fight value, so the middle three instructions are not separated by the model: it reads "good deal", "best deal you can" and "never be exploited" all as "about my fight value". Fidelity reflects that: 0.94 (L1), 0.60 (L2), 0.77 (L3), 0.43 (L4), 0.88 (L5). The L4 text ("Get a good deal, and never be exploited") is the one to revisit; the model treats it as level 2.
4. **Collisions come from level 5 and from strict.** Level 5 floors collide (sum > 1) in 43% of pairs; levels 1–4 almost never (≤ 3%). By version, strict collides 4× as often as lottery or payment (0.19 vs 0.05) because strict floors cluster near 0 or 1.
5. **No unparseable floors (0 of 600), comprehension 1.00, default rate 0.00 in both evals.** Tool calling through the proxy is reliable when it does not stall.

## Cost and wall time

Total spend ≈ **$4.48** (result 1 estimated at $0.0045 per call; the two evals measured from the logs). Productive wall time ≈ 7.4 min (279 s + 64 s + 98 s). Elapsed including stalls, kills and restarts: about 2 h 10 min. Extrapolating the measured per-eval cost (≈ $0.9), the remaining 8 evals of the plan would cost about $7.
