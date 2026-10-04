# SPAR Harness: Crisis & Bargaining Testbed

A testbed for studying how frontier-model agents bargain and escalate against each other, and why bargaining breaks down. It follows Fearon's (1995) rationalist explanations of war: private information, commitment problems, and issue indivisibility.

The first game implemented is **crisis bargaining**. Two players, **alpha** and **beta**, divide a stake of size 1 through alternating offers. Either player can fight instead, which ends the episode with each side getting its expected war payoff. A deterministic, non-model **referee** runs every episode. Players can be language models (through [Inspect](https://inspect.aisi.org.uk/)) or scripted bots.

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
| `crisis_bargaining.py` | Game rules: win probability, reservation values, bargaining range, payoffs, default moves |
| `priors.py` | Seeded drawing of private types |
| `referee.py` | The general phase engine that runs one episode for any `GameSpec` |
| `bots.py` | `equilibrium` and `random` bots |
| `model_player.py` | Adapter that turns an Inspect model into a player: prompts, the `submit_move` tool, the comprehension probe |
| `measures.py` | Per-episode measures |
| `task.py` | The Inspect task, solver, scorers, and reducer |

Bots and model players share one interface, `async (Observation, PhaseContext) -> Move`, so the referee never knows which kind it's dealing with.

## Status

The first slice, crisis bargaining, is implemented. The other planned scenarios (an indivisible good, a nuclear escalation ladder, an AI race) and the remaining treatments (commitments, contracts, side payments, an RNG tool, noise) are not built yet.
