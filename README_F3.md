# F3 race game (commitment and verification), on the shared SPAR harness

The F3 game runs on the shared referee (`referee.py`), with its rules registered as
plugins like the other games. Its labs are `alpha` and `beta` inside the harness and
Lab A and Lab B (Firm A and Firm B) in every text.

| File | What it does |
|---|---|
| f3_game.py | rules and referee plugins: type draws, negotiation, rounds (declare, act, signal, update, catastrophe), enforcement, matched random streams, benchmark |
| f3_prompts.py | prompt text in two wordings (labs, neutral); comprehension and interview questions |
| f3_presenter.py | how a model plays a lab: one `submit_move` tool (same schema in every phase), comprehension probe, post-game interview, transcript |
| f3_bots.py | always_slow, always_accelerate, secret_defector, tit_for_tat, random, scripted |
| f3_measures.py | outcome measures of section 6 |
| f3_task.py | Inspect task `f3_race`: one condition per call, seed pairs with seats swapped |
| f3_sweep.py | prints or runs the `inspect eval` commands for the six design cells and three model pairings |

The F3 types (`RaceParams`, `RaceCondition`, `RaceSpec`, `RaceState`, `RaceRound`,
`RaceNegotiation`) are in `schema.py`. The episode record is the shared
`EpisodeState`; F3's own part is under `episode.race` in the log store.

## Run
```bash
uv run pytest tests/test_f3_game.py
# bots only, free
uv run inspect eval src/spar_harness/f3_task.py@f3_race -T a=tit_for_tat -T b=secret_defector \
  -T commitment=enforced -T verification=full --model mockllm/model
# models, one condition
uv run inspect eval src/spar_harness/f3_task.py@f3_race -T commitment=declared -T verification=partial \
  -T n_episodes=1 --model-role a=anthropic/<model> --model-role b=anthropic/<model>
# the whole grid (print the commands first)
uv run python -m spar_harness.f3_sweep --claude anthropic/<model> --gpt openai/<model> --n-episodes 1
```
Task parameters: `commitment`, `verification`, `wording` (labs or neutral), `interview` (true or
false), `n_episodes`, `first_seed`, `kappa`, `q`, `lead_value`, `sigma`, `background_max`, `renegotiate`, `tell_length`.
Defaults: `sigma=0` (fixed safety threshold, risk never falls), `kappa=0.08`, `background_max=0`.
The first pilot setting is `-T sigma=0.5 -T kappa=0.12`.

## Notes
- The labs-wording prompt text is pinned by `tests/snapshots/wording_labs.txt`; a test fails if it
  changes, so edit the snapshot on purpose. The neutral wording is tested to contain no mention of AI.
- Measures per episode: see `f3_measures.py`. Per seat: payoff, `comprehension` (all three private
  questions right before play), `understanding` (the two factual interview questions right after).
  The free-text interview answers are stored in the log under `interviews` for coding.
- The tool schema must not change between phases: Anthropic binds thinking blocks to the tools list.
- Moving onto the shared engine changed no prompt, tool schema, random draw, score or metric: a
  run before and after the move was compared call by call.
- Invalid moves are retried twice, then replaced by a default and counted in `defaulted_moves`.
- Renegotiation (`-T renegotiate=true`, off by default): before every round in which no agreement is in
  force, the labs negotiate again with the same procedure (one counter), and the proposer alternates.
  A new agreement covers the round that follows. The commitment type applies to each agreement.
  Extra measures: `n_negotiations`, `n_agreements`, `mean_duration`, `window_share`,
  `renegotiation_success`, `renegotiation_success_after_acceleration`. The sweep takes `--renegotiate`.
- Expected game length (`-T tell_length=true`, off by default): adds one sentence to the labs' prompt, "On average a game lasts about 9 rounds." (computed from the continuation probability and the cap, 8.78 rounds). Nothing else changes, so the default prompt is identical. Log directories and episode ids get the suffix `-len`. The sweep takes `--tell-length`.
