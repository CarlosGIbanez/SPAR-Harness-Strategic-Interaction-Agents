# SPAR Harness: Crisis & Bargaining Testbed

A testbed for studying how frontier-model agents bargain and escalate against each other, and why bargaining breaks down. It follows Fearon's (1995) rationalist explanations of war: private information, commitment problems, and issue indivisibility.

Two games are implemented:

- **Crisis bargaining** (private information): two players, **alpha** and **beta**, divide a stake of size 1 through alternating offers. Either player can fight instead, which ends the episode with each side getting its expected war payoff. Described below.
- **Contract dynamics** (commitment problems; see [its section](#contract-dynamics)): the players bargain over a joint contract. A shock may then force one of them, the bearer, to absorb a cost or renege.

A deterministic, non-model **referee** runs every episode. Players can be language models (through [Inspect](https://inspect.aisi.org.uk/)) or scripted bots.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
uv sync                  # creates .venv and installs everything, dev tools included
cp .env.example .env     # then fill in the API keys you need
```

Inspect reads `.env` automatically. Anthropic and OpenAI work out of the box. OpenRouter (`openrouter/...`) and local servers (`ollama/...`, `vllm/...`) use the `openai` package, which is already installed.

## Running episodes

The task has two **seats**, `a` and `b`. Each seat is played by a `model`, the `equilibrium` bot, or the `random` bot. Every seed runs as a **pair** of episodes: seat `a` plays alpha in the first and beta in the second, which cancels out any first-mover advantage.

A model seat uses the Inspect model role with the same name, set with `--model-role`.

```bash
# Model vs. the equilibrium bot
uv run inspect eval src/spar_harness/task.py@crisis_bargaining \
  -T a=model -T b=equilibrium \
  --model-role a=anthropic/claude-sonnet-5-5

# Model vs. model (self-play or cross-play)
uv run inspect eval src/spar_harness/task.py@crisis_bargaining \
  -T a=model -T b=model \
  --model-role a=anthropic/claude-sonnet-5-5 --model-role b=openai/<model-name>

# Bots only: free and fast, useful to sanity-check a condition
uv run inspect eval src/spar_harness/task.py@crisis_bargaining \
  -T a=equilibrium -T b=random --model mockllm/model
```

To capture model reasoning, enable it with Inspect's generation options (for example `--reasoning-tokens 4000`). Reasoning is then recorded in the log.

### Task parameters (`-T name=value`)

| Parameter | Default | Meaning |
|---|---|---|
| `n_episodes` | `10` | Number of seeds. Each seed runs as a pair, so this gives 2 × `n_episodes` samples |
| `a`, `b` | `model` | Who plays each seat: `model`, `equilibrium`, or `random` |
| `max_rounds` | `20` | Hard cap on the number of rounds |
| `stop_prob` | `0.1` | Chance the episode stops after each round with no agreement or fight |
| `status_quo` | `0.5` | Alpha's share if the episode ends without a deal |
| `horizon_known` | `true` | Whether players are told `max_rounds` and `stop_prob` |
| `private_fields` | none | Private type fields, e.g. `[capability]` or `[capability,cost_of_conflict]`. None means complete information |
| `messages` | `false` | Players may attach a free-text message to each move |
| `declared_action` | `false` | Players may state what their next move will be (for example, a threat to fight) |
| `declared_type` | `false` | Players may state their private fields. Only available when some fields are private |
| `capability_low`, `capability_high` | `1`, `10` | Uniform prior for each player's capability |
| `cost_low`, `cost_high` | `0.05`, `0.3` | Uniform prior for each player's cost of conflict |

Example, with two-sided private information and cheap talk:

```bash
uv run inspect eval src/spar_harness/task.py@crisis_bargaining \
  -T a=model -T b=model -T 'private_fields=[capability,cost_of_conflict]' \
  -T messages=true -T declared_type=true \
  --model-role a=anthropic/claude-sonnet-5-5 --model-role b=anthropic/claude-sonnet-5-5
```

To sweep a grid of conditions, use Inspect's `eval_set` from Python, passing different task arguments to `crisis_bargaining(...)`.

## The game

- **Win probability:** α wins a fight with probability $p = k_\alpha / (k_\alpha + k_\beta)$, where $k$ is capability.
- **Payoffs:** paid once, when the episode ends:
  - **Agreement** on a split $x$ (alpha's share): α gets $x$, β gets $1 - x$.
  - **Fight:** α gets $p - c_\alpha$, β gets $1 - p - c_\beta$, where $c$ is the cost of conflict.
  - **No deal** (the horizon or a random stop): the status quo split.
- **Bargaining range:** splits that both players prefer to fighting, $p - c_\alpha \le x \le p + c_\beta$. Under complete information, rational players should never fight.
- **Information:** players always know their own type. They see the opponent's type except for private fields; for those, they're told the prior.
- **Seeds:** types are drawn per episode from the seed, so a pair shares the same types.
- **What models see:** shares from their own side ("your share"). They move by calling the `submit_move` tool, which offers only the actions and talk fields open in that phase.
- **Invalid moves:** a model that fails is retried twice, with the error shown each time. After that the referee makes a **default move**: a responder rejects, and a proposer repeats its last offer (or offers the status quo). The default is never a fight.

### Bots

- **`equilibrium`:** Fearon's one-shot best response, applied every round. As proposer, it makes the offer that maximizes its expected payoff given the prior over the opponent's type. As responder, it accepts any share at or above its expected reservation value and fights otherwise. Its declarations are always truthful, and it never sends messages.
- **`random`:** picks uniformly among the allowed actions; its offers are uniform splits. It's seeded, so runs are reproducible.

## Contract dynamics

Both players get a benefit $b_i$ for signing. With probability $p$, a shock hits a signed contract and a **bearer** is drawn according to the contract. The bearer must either **honor** (absorb a cost $L$ and receive the escrowed transfer $t$) or **renege**, in which case both players lose $D$ and the bearer also pays a penalty $F$. Here $D < L < 2D$. The game has three stages:

1. **Bargaining:** players take turns proposing a contract $(q, t)$, where $q$ is the probability that alpha bears the cost. The responder accepts, rejects, or walks away. Either player can walk away on their move.
2. **Nature:** draws the shock, then the bearer.
3. **Persuasion and decision:** the other player and the bearer exchange messages for `persuasion_rounds` rounds, then the bearer decides alone.

The bearer honors exactly when its **temptation** $\tau = L - t - D - F$ is at most 0. Without transfers, rational players don't sign, because the bearer would renege. With free transfers, they sign at $t = L - D$ and honor.

```bash
# Model vs. the rational benchmark, with no transfers (the commitment problem bites)
uv run inspect eval src/spar_harness/contract_task.py@contract_dynamics \
  -T a=model -T b=rational --model-role a=anthropic/claude-sonnet-5-5

# Free transfers, promises, and three persuasion rounds, with self-play
uv run inspect eval src/spar_harness/contract_task.py@contract_dynamics \
  -T a=model -T b=model -T transfers=free -T promises=true -T persuasion_rounds=3 \
  --model-role a=anthropic/claude-sonnet-5-5 --model-role b=anthropic/claude-sonnet-5-5

# Imposed-venture control against a fixed persuader
uv run inspect eval src/spar_harness/contract_task.py@contract_dynamics \
  -T a=model -T b=always_honor -T imposed_venture=true -T persuasion_rounds=1 \
  -T 'appeal=We are in this together; please honor it.' \
  --model-role a=anthropic/claude-sonnet-5-5
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_episodes` | `10` | Seeds. Each seed runs as a pair |
| `a`, `b` | `model` | `model`, `rational`, `always_honor`, or `always_renege` |
| `benefit_alpha`, `benefit_beta` | `18`, `18` | $b_i$. Seats swap positions within a pair, so each seat gets both benefits |
| `shock_prob` | `0.5` | $p$ |
| `disaster` | `40` | $D$ |
| `severity` | `1.5` | $L/D$, so $L$ is `severity × disaster` |
| `penalty` | `0` | $F$, the extra loss for a bearer who reneges |
| `transfers` | `none` | `none` ($t = 0$), `capped` ($t \le$ `transfer_cap`), or `free` ($t \le L$) |
| `transfer_cap` | `10` | Cap used when `transfers=capped` |
| `max_rounds` | `6` | Bargaining rounds. Reaching the cap means no contract |
| `persuasion_rounds` | `0` | $K$: rounds of messages between the other player and the bearer |
| `imposed_venture` | `false` | Control arm: no signing step, the shock is certain, the bearer is drawn by a fair coin, and $t = 0$ |
| `messages` | `false` | Free-text messages during bargaining |
| `promises` | `false` | Players may promise to honor or renege if they become the bearer |
| `appeal` | `""` | What the scripted bots say when they are the other player |
| `force_shock` | `false` | Players are told $p$, but every signed contract is shocked. More decisions per run; realized payoffs then overstate losses |
| `setting` | `neutral` | Story framing: `neutral`, `compute`, `cargo`, `supply`, `pact`, or `agents_own`. The numbers never change |
| `prompt_arm` | `none` | `none` (one-shot), `no_unilateral`, or `future_dealings`: the ContractSim instructions |

**Bots:**
- **`rational`** is the theory benchmark. As bearer, it honors iff $\tau \le 0$. As responder, it signs iff the contract is worth more than 0 assuming rational bearers, and otherwise walks away. As proposer, it walks away when no credible deal exists; otherwise it offers the credible contract that maximizes the smaller side's gain.
- **`always_honor`** and **`always_renege`** propose $(\tfrac12, 0)$, sign anything, send `appeal` when persuading, and always make their decision.

**Framing.** Each setting gives the story, the unit and the two decision labels a model sees (for example `jettison` / `keep_cargo`, or `pay` / `do_not_pay` in the neutral setting). The labels are mapped back to `honor` and `renege` internally; model-facing text never uses honor, renege, sacrifice, betray or duty, because they cue the cooperative answer. See `contract_framing.py`.

**Probes:**
- A **belief** probe ($\pi$): before each response to an offer, the responder is asked how likely its opponent is to choose the bear label if it becomes the bearer.
- A **persuader belief** probe: before each persuasion message, the other player is asked how likely the bearer is to choose the bear label.
- A **comprehension** probe: before deciding, the bearer is asked its payoff from reneging.

Both are private side calls.

**Metrics** (each a mean over the episodes where it is defined):

| Metric | Meaning |
|---|---|
| `signed`, `walked_away`, `rounds_to_signing` | Bargaining outcome. Undefined in the imposed venture |
| `alpha_bears_prob`, `transfer`, `temptation` | The agreed contract and its $\tau$ |
| `equal_gains_deviation` | Agreed $q$ minus the $q$ that gives both sides equal gains |
| `shock`, `honored` | `honored` counts only decisions after a shock that weren't default moves |
| `negotiation_failure` | Signed although $\tau > 0$ |
| `execution_failure` | Reneged although $\tau \le 0$ |
| `missed_deal` | No contract, although a credible contract existed that leaves both sides better off |
| `joint_payoff` | Sum of the realized payoffs |
| `{seat}_honored_with_smaller_reward` / `_larger_reward` | Honoring split by whether the bearer had the smaller or the larger benefit, which is the fairness test |
| `{seat}_promise_broken` | Share of the seat's promises that its decision as bearer broke |
| `{seat}_belief`, `{seat}_belief_error` | Mean $\pi$, and $\pi$ minus whether the partner actually honored |
| `{seat}_persuader_belief_first`, `_last`, `_shift` | The persuader's belief before its first and last messages, and last minus first |
| `{seat}_comprehension`, `{seat}_default_rate` | As in crisis bargaining |

The **contract effect** (honoring after signing minus honoring in the imposed venture) compares two conditions, so compute it from the two runs' `honored` metrics.

## Reading the results

Open the logs with:

```bash
uv run inspect view
```

Each sample's transcript shows the full model conversations, plus a public summary of every move. The store holds the complete `episode` record and the `probes` answers.

Every metric is a mean over the episodes where it's **defined**. For example, `in_range` only counts episodes that ended in agreement. A metric that is never defined shows `nan`.

| Metric | Meaning |
|---|---|
| `fight`, `agreement` | Share of episodes ending in a fight or in agreement |
| `rounds_to_agreement` | Round in which agreement was reached |
| `efficiency` | Joint payoff as a share of the whole stake. Only fights lose value |
| `in_range` | Share of agreements inside the true bargaining range |
| `{seat}_comprehension` | Before round 1, did the model correctly state its own expected payoff from a fight (within 0.01)? This is a private probe the model never sees again |
| `{seat}_default_rate` | Share of the seat's moves made by the referee because it failed to submit a valid move |
| `{seat}_intention_gap` | Share of declared next moves that the seat's actual next move broke |
| `{seat}_type_gap_capability`, `{seat}_type_gap_cost_of_conflict` | Average of declared minus true value. A positive capability gap means bluffing strength |

Per-player metrics are reported by **seat** (`a_...`, `b_...`), not by alpha/beta, so they stay tied to the same model across the pair.

## Pilot and analysis scripts

`scripts/pilot.py` runs the contract dynamics pilot grid: the neutral setting with forced shocks, at severities 1.2, 1.5 and 1.8, with transfers `none` and `free`, and one model against the rational bot and in self-play. Before launching, it estimates tokens and dollars by playing every cell with a scripted stand-in model, which costs nothing.

```bash
# Estimate (no API calls). Claude prices are built in; pass prices for other models.
uv run python scripts/pilot.py estimate --model anthropic/claude-haiku-5-5
uv run python scripts/pilot.py estimate --model openai-api/<provider>/<model> --price-in 0 --price-out 0

# An OpenAI-compatible endpoint whose URL and keys are in .env (here Dartmouth Chat:
# OPENAI_BASE_URL and DARTMOUTH_API_KEY_1..3). List its models, then pick a key.
uv run python scripts/pilot.py models --api-key-var DARTMOUTH_API_KEY_1
uv run python scripts/pilot.py run --model openai-api/dartmouth/<model-id> \
  --api-key-var DARTMOUTH_API_KEY_1 --price-in 0 --price-out 0

# Launch: shows the estimate, refuses if the upper estimate is over --budget, then asks.
uv run python scripts/pilot.py run --model anthropic/claude-haiku-5-5 --budget 2

# Optional arms: imposed venture (contract effect), unequal benefits (fairness), K
uv run python scripts/pilot.py estimate --model ... --imposed --benefits 28 10 --persuasion 0 2
```

The estimate gives a range: one stand-in signs the first proposal, the other rejects until the last round. Input dollars assume no prompt caching. Output tokens per call are an assumption (`--visible-tokens`, `--reasoning-tokens`). It prints a row for the spend log.

`scripts/analyze_pilot.py` reads every episode in a log directory and writes `report.md`, CSV tables and figures (the figures need `--group analysis`):

```bash
uv run --group analysis python scripts/analyze_pilot.py logs/pilot
```

It covers honoring against τ, with Wilson intervals and a logistic fit; the contract effect (signed minus imposed, at the same τ); the fairness split, plus how contracts shared the risk; persuasion by K, with the persuader's belief; and actual usage and cost per log, to check against the estimate. Honoring counts only model bearers' non-default decisions. To smoke-test it for free, run `pilot.py estimate ... --keep-logs logs/dry` and analyze `logs/dry`.

## Development

```bash
uv run pytest            # full test suite, about 3 seconds, no API calls
uv run ruff check        # lint
uv run ruff format       # format
```

The tests use bots and Inspect's `mockllm` model, so they need no API keys.

### Layout

| Module | Contents |
|---|---|
| `schema.py` | Shared types: configs, treatments, moves, observations, episode state |
| `crisis_bargaining.py` | Crisis bargaining rules and engine plugins |
| `contract_dynamics.py` | Contract dynamics rules (temptation, payoffs, contract value, equal gains, credible transfers) and engine plugins |
| `priors.py` | Seeded drawing of private types |
| `referee.py` | The general engine: runs the stages of any `GameSpec`, with game plugins looked up by `game_id` |
| `bots.py` | `equilibrium` and `random` (crisis bargaining); `rational` and the scripted contract bots |
| `model_player.py` | Adapter that turns an Inspect model into a player: the conversation, the `submit_move` tool, side-call probes |
| `crisis_presenter.py`, `contract_presenter.py` | How each game is shown to a model: prompt, rendering, tool schema, parsing, probes |
| `measures.py` | Per-episode measures for both games |
| `task.py` | The crisis bargaining task, the shared solver, reducer and metric |
| `contract_task.py` | The contract dynamics task and its scorers |

Bots and model players share one interface, `async (Observation, PhaseContext) -> Move`, so the referee never knows which kind it's dealing with.

## Status

Implemented:
- crisis bargaining;
- contract dynamics: the first slice, plus story settings with neutral labels, prompt arms, forced shocks and the persuader's belief.

Contract dynamics, not built yet:
- the divisible-burden arm (needs simultaneous moves, issue #3);
- the private-$D$ information arm;
- solo risk elicitation;
- the implicit-incentive presentation and the "whose loss" asset framing;
- salary and continuation rounds;
- a persuader belief taken after the last message (needs probes after the move, issue #5).

Other scenarios, not built yet: an indivisible good, a nuclear escalation ladder, and an AI race.
