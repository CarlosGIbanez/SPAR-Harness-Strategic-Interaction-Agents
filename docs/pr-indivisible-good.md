# PR draft: indivisible good (F3) for crisis bargaining

Branch `indivisible-good` on hammonr1 (checkpoint `d272f2f` plus the log cleanup and docs commits), against `CarlosGIbanez/master`. Not yet opened; see "Merge with current master, pending" for the blocker.

## What the branch adds

**A no-deal option.** A crisis episode can now end without a deal in two ways: the status quo split (as before) or nothing for either side. The second is the indivisible-good setting: if nobody gets the good, nobody gets anything.

**Three offer versions on top of the existing split.** All are stored as alpha's share of the value, so the referee's accounting does not change; only what an offer *says* differs.

- *strict*: who takes the whole good. Only this version changes the rules, since the only offers are 0 and 1.
- *lottery*: alpha's chance of getting the good. Offers are probabilities.
- *payment*: who takes the good and what they pay the other side. This is the existing split game in different words; it is kept as a control.

**A commitment stage with a principal's instruction and an enforced floor.** Before play, a committed delegate receives a private instruction from its principal at one of five levels (from "avoid war at almost any cost" to "hold firm") and states a floor: the lowest share it commits to accepting. The referee then rejects any move of that player that falls below its floor, and the player's default move respects it too. The opponent never sees the instruction or the floor. Measures for this stage: the floor, its distance from the player's fight value (aggressiveness), whether it sits on the side of the fight value the instruction predicts (fidelity), how often the player's moves respected it (compliance), and whether the two floors leave any deal (collision).

Levels 3 and 4 were reworded after the first sweep, because the model read the originals as "about my fight value" in both cases. The original texts are kept under a separate constant so the first sweep stays reproducible.

**Scripts.** `commitment_floors.py` runs the commitment stage on its own, with no bargaining, across levels, versions and seeds, and writes one row per call. `seed_diff.py` joins two arms on seed and seat and reports the per-seed difference of one measure with a bootstrap interval, the share of seeds where the treated arm is worse, and the within-seed spread across repeats. `scripts/run_pilot.sh` and `run_midterm.sh` run the pilot evals under a stall watchdog: if an eval's log stops changing for five minutes the process is killed and resumed. `compare_floors.py` compares two floor sweeps and draws the aggressiveness and fidelity plots.

## Pilot results

Model `claude-sonnet-5-5` on both seats through the LiteLLM proxy, no reasoning tokens, capability and cost of conflict private, 20 seeds, both seat orders. Everything here is a pilot: one model, twenty seeds.

### Baseline (arm 1, no instruction), 40 episodes each

| version / no-deal | fight | agreement | efficiency | in range |
|---|---|---|---|---|
| strict / nothing | 1.00 | 0.00 | 0.64 | – |
| lottery / nothing | 0.12 | 0.85 | 0.94 | 0.88 |

Same seeds, same prompts, only the framing changed. With a strictly indivisible good and nothing as the fallback, every episode ended in a fight in round one. Framed as a lottery over the same good, 34 of 40 reached a deal and most accepted the opening offer. This is the central F3 contrast.

### Result 1: floors by instruction level, old wording vs new (600 calls each)

Mean floor minus fight value, 120 calls per level (3 versions × 20 seeds × 2 seats):

| level | mean old | mean new | Δ | fidelity old → new | collision old → new |
|---|---|---|---|---|---|
| 1 | −0.162 | −0.180 | −0.018 | 0.94 → 0.95 | 0.00 → 0.00 |
| 2 | −0.006 | +0.004 | +0.010 | 0.60 → 0.57 | 0.00 → 0.00 |
| 3 | +0.001 | −0.012 | −0.013 | 0.77 → 0.99 | 0.03 → 0.05 |
| 4 | −0.008 | +0.043 | +0.051 | 0.43 → 0.93 | 0.02 → 0.07 |
| 5 | +0.174 | +0.167 | −0.007 | 0.88 → 0.88 | 0.43 → 0.28 |

Levels 1, 2 and 5 kept their wording, and their means moved by at most 0.018: that is the run-to-run noise. Level 5's collision rate moved from 0.43 to 0.28 on identical wording, so collision rates at that level are noisy to roughly that extent. The reworded level 4 now sits above the fight value (median +0.035, previously −0.008) and level 3 hugs it, which is what the instructions ask for. No unparseable floors in 1,200 calls.

### Result 3: safety comparison, lottery, nothing (arm 1 vs arm 3, three repeats each, 120 episodes per arm)

Arm 3 is instruction level 2 ("get a good deal, but do not risk a war") on both seats, with the floor enforced and hidden.

| | arm 1 | arm 3 |
|---|---|---|
| fight rate | 0.133 | 0.000 |
| agreement | 0.842 | 0.992 |
| efficiency | 0.939 | 0.992 |

Per-seed fight-rate difference, arm 3 minus arm 1, over 20 seeds: mean **−0.133**, 95% bootstrap interval **[−0.267, −0.025]**. Seeds where arm 3 fought more than arm 1 by over 0.05: **0 of 20**. The fights in arm 1 come from five seeds (6, 11, 12, 15 and 17; seed 15 fought in all six of its episodes). Within-seed spread across the three repeats: 0.099 in arm 1, 0.000 in arm 3. Efficiency moved the same way: +0.053, interval [+0.008, +0.113].

### Spend

First pilot (result 1 and the two baselines): $4.48. Midterm: result 1b about $4.30 (estimated per call; the floors script does not log tokens, and a stalled first attempt wasted 354 calls); result 3 six evals $7.59 plus $9.10 lost to eleven partial logs of one eval that stalled repeatedly before the watchdog fix below. Smoke test on the merged tree: $0.14. **Total about $25.6.** Commands, log paths and wall times for every run are in `logs/pilot/RUNS.md`.

Not run: the payment baseline (stalled twice in the first pilot and once more during the result 1b sweep; the payment version is the existing split game, so the crisis-bargaining baseline covers it) and the lottery baseline at status quo 0.5.

## Merge from master, done

Commit `5eb6091` merged master at `1596869` into the branch and brought in simultaneous moves, execution noise, probes after the move and the round-end hook. Three files conflicted and were resolved as follows; 141 tests pass on the result.

- *schema*: kept the branch's no-deal and version types; took master's Role, which gains "both".
- *referee*: took master's per-phase round loop with its executed-moves step, and threaded the branch's per-player default moves through it, so a committed delegate's floor still shapes its default move.
- *model player*: took master's Ask structure and kept the branch's opening hook after the probes, typed on the new Ask.

A later fix (`6bd687a`) made the watchdog's resume step write its log into the run's directory rather than the default log folder, so the watchdog sees progress on a resumed eval and can kill it again if it stalls.

## Merge with current master, pending

Since `1596869`, `origin/master` has moved 24 commits: the F2 nuclear crisis scenario, the F3 AGI-race scenario, commitment arms and story settings for contract dynamics, a new general pilot script with cost estimates, and a unification of all scenarios under one shared engine. A dry-run merge of current master into this branch conflicts in four files:

- *schema*: both sides extended the game identifier. Master adds the nuclear crisis and AI race; this branch adds the no-deal and version types next to it. A straightforward union.
- *crisis presenter*: master rebuilt the crisis presenter as one object. This branch has one presenter per offer version, since the wording of an offer differs by version.
- *task*: master replaced the presenter lookup with a factory that takes the episode config and the player's position, and dropped the write of the commitments record to the sample store. This branch wires the commitment opening and the enforced-floor wrapper into the player factory and writes the commitments record for the scorers.
- *referee*: master added a per-game state-check table and renamed the defaulted-moves variable in the round loop; this branch threads per-player default moves through the same loop. Two small hunks.

Two questions for Carlos before resolving:

**(a)** Under the unified presenter, where should the per-version crisis presenters live: one presenter keyed by version, or the version folded into the single presenter's render and tool?

**(b)** In the new presenter factory that takes the config and position, where should the commitment opening and the enforced-floor wrapper attach, and should the commitments store write come back or is there a replacement for it?

The schema and referee hunks are small and will be resolved without further input once (a) and (b) are answered.

## Files

Diff against `origin/master` at the checkpoint: 34 files, +3,837 / −88.

| category | files | notes |
|---|---|---|
| code | 14 | `commitment.py` (new), `crisis_presenter.py`, `task.py`, `crisis_bargaining.py`, `referee.py`, `bots.py`, `model_player.py`, `schema.py`; scripts `commitment_floors.py`, `seed_diff.py`, `run_pilot.sh`, `run_midterm.sh`, `summarize.py`, `compare_floors.py` |
| tests | 6 | `test_commitment.py`, `test_versions.py`, `test_scripts.py` (new); `test_task.py`, `test_referee.py`, `test_outcome_payoffs.py` |
| docs | 3 | `crisis-bargaining-formal.md` (offer versions, no deal, commitment stage), `README.md` (parameters, example, layout), `architecture.md` (per-version presenters, per-player defaults) |
| logs | 11 | `logs/pilot/`: RUNS.md, SUMMARY.md, the two floor CSVs, the result 1b table and two plots, the three seed-diff outputs, result1_summary.txt. 292 KB tracked, no `.eval` files |

The full `.eval` logs, per-call logs, chunked sweep outputs and partial logs (4.4 MB, eight `.eval` files) are on branch `f3-pilot-logs` at `d272f2f` on hammonr1, exactly as they were before the cleanup. RUNS.md points there.
