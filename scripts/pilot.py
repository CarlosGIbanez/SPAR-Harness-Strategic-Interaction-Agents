"""Pilot grid for contract dynamics: estimate tokens and dollars, then launch.

The grid uses forced shocks, at each severity and transfer regime, with one
model against the rational bot and in self-play. The setting is neutral unless
--setting names a story; --fixed-bearer lets only one side bear the cost (the
captain setting needs it), and --shock-prob is the chance players are told.

    # Estimate only (free: no API calls). Prints tokens and dollars per cell.
    uv run python scripts/pilot.py estimate --model anthropic/claude-sonnet-5-5

    # Launch, after logging the estimate. Refuses if the estimate is over budget.
    uv run python scripts/pilot.py run --model anthropic/claude-sonnet-5-5 --budget 2

    # An OpenAI-compatible endpoint, such as Dartmouth Chat: list its models,
    # then run one with a chosen key from .env.
    uv run python scripts/pilot.py models --api-key-var DARTMOUTH_API_KEY_1
    uv run python scripts/pilot.py run --model openai-api/dartmouth/<model-id> \
        --api-key-var DARTMOUTH_API_KEY_1 --price-in 0 --price-out 0

The estimate plays every cell with a scripted stand-in model that counts the
prompt it is sent, so call counts and conversation growth match the real
harness. Two stand-ins bracket the real model: "quick" signs the first
proposal, "slow" rejects until the last bargaining round. The dollar figure
has no prompt caching, so it is an upper bound on input cost; output tokens
per call are an assumption (--reasoning-tokens, --visible-tokens). Compare
it with the real usage that analyze_pilot.py reports after the run.
"""

import argparse
import json
import os
import random
import re
import sys
import tempfile
import urllib.request
import zlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from inspect_ai import eval, eval_set, task_with
from inspect_ai.model import ChatMessage, ModelOutput, ModelUsage, get_model
from inspect_ai.tool import ToolInfo

from spar_harness.contract_framing import SETTINGS
from spar_harness.contract_task import contract_dynamics

# Dollars per million tokens (input, output), first-party API, no caching.
# Free models (local, or a free institutional endpoint) cost 0: pass
# --price-in 0 --price-out 0. Anything not listed needs explicit prices.
PRICES = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}
CHARS_PER_TOKEN = 3.5  # conservative for English prose and JSON
PER_CALL_OVERHEAD = 350  # tokens a provider adds for tool use


@dataclass(frozen=True)
class Cell:
    severity: float
    transfers: str  # "none" or "free"
    opponent: str  # "rational" or "self"
    imposed_venture: bool = False
    persuasion_rounds: int = 0
    fixed_transfer: float | None = None  # set by the game, not bargained over
    penalty: float = 0.0  # extra loss for a bearer who reneges (maybe uncertain)

    @property
    def label(self) -> str:
        arm = "imposed" if self.imposed_venture else f"t-{self.transfers}"
        if self.fixed_transfer is not None:
            arm += f"-fixed{self.fixed_transfer:g}"
        if self.penalty:
            arm += f"-pen{self.penalty:g}"
        return (
            f"sev{self.severity:g}_{arm}_vs-{self.opponent}_K{self.persuasion_rounds}"
        )


def grid(args: argparse.Namespace) -> list[Cell]:
    cells = []
    for k in args.persuasion:
        for severity in args.severities:
            for opponent in args.opponents:
                if args.fixed_transfers:  # each value is a signed and an imposed cell
                    for t in args.fixed_transfers:
                        cells.append(Cell(severity, "none", opponent, False, k, t))
                        if args.imposed:
                            cells.append(Cell(severity, "none", opponent, True, k, t))
                    continue
                for transfers in args.transfers:
                    cells.append(Cell(severity, transfers, opponent, False, k))
                if args.imposed:  # no transfers in the imposed venture
                    cells.append(Cell(severity, "none", opponent, True, k))
    if args.penalties:  # every cell at every swept penalty
        cells = [replace(cell, penalty=p) for p in args.penalties for cell in cells]
    return cells


def cell_task(cell: Cell, args: argparse.Namespace):
    alpha, beta = args.benefits
    task = contract_dynamics(
        n_episodes=args.seeds,
        a="model",
        b="model" if cell.opponent == "self" else cell.opponent,
        benefit_alpha=alpha,
        benefit_beta=beta,
        severity=cell.severity,
        transfers=cell.transfers,
        persuasion_rounds=cell.persuasion_rounds,
        imposed_venture=cell.imposed_venture,
        shock_prob=args.shock_prob,
        force_shock=True,
        setting=args.setting,
        fixed_bearer=args.fixed_bearer,
        stated_promise=args.stated_promise,
        fixed_transfer=cell.fixed_transfer,
        penalty=cell.penalty,
        penalty_prob=args.penalty_prob,
        honor_cost_prob=args.honor_cost_prob,
    )
    return task_with(task, name=f"pilot_{cell.label}", metadata={"cell": asdict(cell)})


# The stand-in model -----------------------------------------------------------


def stand_in(policy: str, visible_tokens: int, reasoning_tokens: int, calls: list):
    """A scripted model that plays legal moves and reports estimated usage."""

    def generate(input: list[ChatMessage], tools: list[ToolInfo], tool_choice, config):
        prompt = [message for message in input if message.role == "user"][-1].text
        transcript = "\n".join(message.text for message in input)
        if "Private question" in prompt:
            output = ModelOutput.for_tool_call(
                "stand-in", "answer_probe", {"answer": 0.5}
            )
            kind = "probe"
        else:
            output = ModelOutput.for_tool_call(
                "stand-in", "submit_move", _move(policy, prompt, transcript)
            )
            kind = "move"
        input_tokens = _input_tokens(input, tools, visible_tokens)
        output_tokens = visible_tokens + reasoning_tokens
        output.usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        calls.append((kind, input_tokens, output_tokens))
        return output

    return generate


def _move(policy: str, prompt: str, transcript: str) -> dict:
    if "send a message" in prompt:
        return {"message": "Please consider both of our payoffs before you decide."}
    if "You are the proposer" in prompt:
        move = {"action": "propose", "probability_you_bear": 0.5}
        if "+ transfer" in transcript:  # transfers allowed: offer the credible one
            loss = float(re.search(r"benefit - ([\d.]+) \+ transfer", transcript)[1])
            disaster, penalty = map(
                float,
                re.search(r"benefit - ([\d.]+) - ([\d.]+) \|", transcript).groups(),
            )
            move["transfer"] = max(0.0, loss - disaster - penalty)
        return move
    if "You are the responder" in prompt:
        last = int(re.search(r"at most (\d+) rounds", transcript)[1])
        round_ = int(re.search(r"Bargaining round (\d+)", prompt)[1])
        signs = policy == "quick" or round_ >= last
        return {"action": "accept" if signs else "reject"}
    # The decision: either label, at random, so smoke-test logs have both.
    options = prompt.split("Your options: ")[-1].rstrip(".").split(", ")
    return {"action": random.Random(zlib.crc32(transcript.encode())).choice(options)}


def _input_tokens(input: list[ChatMessage], tools: list[ToolInfo], visible: int) -> int:
    chars = sum(len(message.text) for message in input if message.role != "assistant")
    chars += sum(len(json.dumps(tool.model_dump(exclude_none=True))) for tool in tools)
    earlier_outputs = sum(1 for message in input if message.role == "assistant")
    return int(chars / CHARS_PER_TOKEN) + PER_CALL_OVERHEAD + earlier_outputs * visible


# Estimate -----------------------------------------------------------------------


def estimate(args: argparse.Namespace) -> dict:
    price_in, price_out = prices(args)
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        for cell in grid(args):
            row = {"cell": cell.label}
            for policy in ("quick", "slow"):
                calls: list = []
                model = get_model(
                    "mockllm/model",
                    custom_outputs=stand_in(
                        policy, args.visible_tokens, args.reasoning_tokens, calls
                    ),
                )
                [log] = eval(
                    cell_task(cell, args),
                    model=model,
                    model_roles={"a": model, "b": model},
                    # Only the quick logs are kept, so each episode appears once.
                    log_dir=args.keep_logs
                    if args.keep_logs and policy == "quick"
                    else temporary,
                    display="none",
                )
                if log.status != "success":
                    sys.exit(f"Dry run failed for {cell.label}: {log.error.message}")
                tokens_in = sum(call[1] for call in calls)
                tokens_out = sum(call[2] for call in calls)
                row[policy] = {
                    "calls": len(calls),
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "dollars": (tokens_in * price_in + tokens_out * price_out) / 1e6,
                }
            rows.append(row)
    totals = {
        policy: {
            key: sum(row[policy][key] for row in rows)
            for key in ("calls", "input_tokens", "output_tokens", "dollars")
        }
        for policy in ("quick", "slow")
    }
    margin = 1 + args.margin
    return {
        "date": datetime.now().astimezone().date().isoformat(),
        "model": args.model,
        "price_per_mtok": {"input": price_in, "output": price_out},
        "assumed_output_tokens_per_call": args.visible_tokens + args.reasoning_tokens,
        "seeds": args.seeds,
        "episodes": len(rows) * 2 * args.seeds,
        "benefits": args.benefits,
        "setting": args.setting,
        "fixed_bearer": args.fixed_bearer,
        "shock_prob": args.shock_prob,
        "stated_promise": args.stated_promise,
        "penalties": args.penalties,
        "penalty_prob": args.penalty_prob,
        "honor_cost_prob": args.honor_cost_prob,
        "margin": args.margin,
        "cells": rows,
        "total": totals,
        "dollars_typical": totals["quick"]["dollars"] * margin,
        "dollars_upper": totals["slow"]["dollars"] * margin,
    }


def prices(args: argparse.Namespace) -> tuple[float, float]:
    if args.price_in is not None and args.price_out is not None:
        return args.price_in, args.price_out
    name = args.model.split("/")[-1]
    if name in PRICES:
        return PRICES[name]
    sys.exit(
        f"No price known for {args.model}. Pass --price-in and --price-out "
        "(dollars per million tokens; 0 for a free model)."
    )


def print_estimate(result: dict) -> None:
    print(f"Pilot estimate for {result['model']}, {result['date']}")
    print(
        f"{result['episodes']} episodes ({result['seeds']} seeds x 2 per cell), "
        f"benefits {result['benefits']}, forced shocks (told p = "
        f"{result['shock_prob']:g}), {result['setting']} setting"
        + (f", fixed bearer {result['fixed_bearer']}" if result["fixed_bearer"] else "")
        + (", stated promise" if result["stated_promise"] else "")
        + (
            f", penalties {result['penalties']} at p {result['penalty_prob']:g}"
            if result["penalties"]
            else ""
        )
        + (
            f", honoring costs with p {result['honor_cost_prob']:g}"
            if result["honor_cost_prob"] < 1
            else ""
        )
        + "\n"
    )
    print(f"{'cell':<36} {'calls':>11} {'input tok':>17} {'output tok':>17} {'$':>13}")
    for row in result["cells"]:
        quick, slow = row["quick"], row["slow"]
        print(
            f"{row['cell']:<36} {quick['calls']:>5}-{slow['calls']:<5} "
            f"{quick['input_tokens']:>8}-{slow['input_tokens']:<8} "
            f"{quick['output_tokens']:>8}-{slow['output_tokens']:<8} "
            f"{quick['dollars']:>6.3f}-{slow['dollars']:<6.3f}"
        )
    total = result["total"]
    print(
        f"\nTotal: {total['quick']['calls']}-{total['slow']['calls']} calls, "
        f"{total['quick']['input_tokens']:,}-{total['slow']['input_tokens']:,} input "
        f"and {total['quick']['output_tokens']:,}-{total['slow']['output_tokens']:,} "
        "output tokens"
    )
    print(
        f"Dollars, with a {result['margin']:.0%} margin for retries: "
        f"typical ${result['dollars_typical']:.2f}, upper ${result['dollars_upper']:.2f}"
    )
    print(
        "Ranges are quick-slow stand-ins. Input dollars assume no prompt caching; "
        f"output assumes {result['assumed_output_tokens_per_call']} tokens per call."
    )
    print(
        "\nSpend log row: "
        f"{result['date']},contract-dynamics pilot,{result['model']},"
        f"{result['episodes']} episodes,${result['dollars_typical']:.2f}"
        f" (upper ${result['dollars_upper']:.2f})"
    )


# Run --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    result = estimate(args)
    print_estimate(result)
    if args.budget is not None and result["dollars_upper"] > args.budget:
        sys.exit(
            f"\nThe upper estimate ${result['dollars_upper']:.2f} is over the budget "
            f"${args.budget:.2f}. Lower --seeds, drop cells, or raise --budget."
        )
    if not args.yes:
        answer = input("\nLaunch the pilot with real model calls? Type yes: ")
        if answer.strip().lower() != "yes":
            sys.exit("Not launched.")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "pilot_estimate.json").write_text(json.dumps(result, indent=2))
    generate_args = {}
    if args.reasoning_effort:
        generate_args["reasoning_effort"] = args.reasoning_effort
    # openai-api sends tools with strict: true by default, which OpenAI rejects
    # because submit_move's fields are optional.
    model = args.model
    if args.model.startswith("openai-api/"):
        accept_responses_usage()
        model = get_model(args.model, strict_tools=False)
    success, _ = eval_set(
        [cell_task(cell, args) for cell in grid(args)],
        log_dir=str(log_dir),
        model=model,
        model_roles={"a": model, "b": model},
        max_connections=args.max_connections,
        **generate_args,
    )
    print(
        f"\nLogs in {log_dir}. Next: uv run python scripts/analyze_pilot.py {log_dir}"
    )
    if not success:
        sys.exit("Some tasks did not finish; rerun the same command to resume them.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["estimate", "run", "models"])
    parser.add_argument(
        "--model", help="Inspect model, e.g. anthropic/claude-sonnet-5-5"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="seeds per cell; each is a pair of episodes",
    )
    parser.add_argument("--severities", type=float, nargs="+", default=[1.2, 1.5, 1.8])
    parser.add_argument(
        "--transfers",
        nargs="+",
        default=["none", "free"],
        choices=["none", "capped", "free"],
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["rational", "self"],
        choices=["rational", "self"],
    )
    parser.add_argument(
        "--persuasion", type=int, nargs="+", default=[0], help="values of K"
    )
    parser.add_argument(
        "--imposed",
        action="store_true",
        help="add imposed-venture cells (contract effect)",
    )
    parser.add_argument(
        "--benefits",
        type=float,
        nargs=2,
        default=[18.0, 18.0],
        metavar=("ALPHA", "BETA"),
        help="unequal benefits are needed for the fairness split",
    )
    parser.add_argument(
        "--setting", choices=sorted(SETTINGS), default="neutral", help="story framing"
    )
    parser.add_argument(
        "--fixed-bearer",
        choices=["alpha", "beta"],
        help="only this side can be the bearer; q is not negotiated",
    )
    parser.add_argument(
        "--fixed-transfers",
        type=float,
        nargs="+",
        help="sweep transfers set by the game (replaces --transfers); with "
        "--imposed, the imposed venture carries each one too",
    )
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        help="sweep the penalty for reneging; every cell runs at each value",
    )
    parser.add_argument(
        "--penalty-prob",
        type=float,
        default=1.0,
        help="chance the penalty is applied to a bearer who reneges",
    )
    parser.add_argument(
        "--honor-cost-prob",
        type=float,
        default=1.0,
        help="chance that honoring costs the bearer anything",
    )
    parser.add_argument(
        "--stated-promise",
        action="store_true",
        help="contracts state the bearer agrees to take the cost, with a reminder",
    )
    parser.add_argument(
        "--shock-prob",
        type=float,
        default=0.5,
        help="chance of the shock players are told (every contract is shocked)",
    )
    parser.add_argument(
        "--api-key-var",
        default="DARTMOUTH_API_KEY_1",
        help="openai-api models: the .env variable holding the key",
    )
    parser.add_argument(
        "--base-url-var",
        default="OPENAI_BASE_URL",
        help="openai-api models: the .env variable holding the base URL",
    )
    parser.add_argument(
        "--service", default="dartmouth", help="models: the endpoint's name"
    )
    parser.add_argument(
        "--price-in", type=float, help="dollars per million input tokens"
    )
    parser.add_argument(
        "--price-out", type=float, help="dollars per million output tokens"
    )
    parser.add_argument(
        "--visible-tokens",
        type=int,
        default=150,
        help="assumed visible output per call",
    )
    parser.add_argument(
        "--reasoning-tokens",
        type=int,
        default=600,
        help="assumed reasoning output per call",
    )
    parser.add_argument(
        "--margin", type=float, default=0.15, help="extra share for retries"
    )
    parser.add_argument(
        "--budget", type=float, help="refuse to run if the upper estimate exceeds this"
    )
    parser.add_argument("--log-dir", default="logs/pilot")
    parser.add_argument("--max-connections", type=int, default=10)
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--yes", action="store_true", help="launch without asking")
    parser.add_argument("--json", action="store_true", help="estimate: print JSON")
    parser.add_argument(
        "--keep-logs",
        help="estimate: keep the stand-in logs here, to smoke-test analyze_pilot.py",
    )
    return parser.parse_args(argv)


def use_endpoint(args: argparse.Namespace) -> None:
    """Point an openai-api/<service>/<model> at a key and base URL from .env.

    Inspect reads <SERVICE>_API_KEY and <SERVICE>_BASE_URL. This copies the
    variables named by --api-key-var and --base-url-var into those names, so
    .env can hold several keys for one endpoint.
    """
    load_dotenv(find_dotenv(usecwd=True))  # the .env where you run it; set ones win
    if args.command != "models" and not args.model.startswith("openai-api/"):
        return
    service = args.model.split("/")[1] if args.model else args.service
    prefix = service.upper().replace("-", "_")
    for suffix, var in (("API_KEY", args.api_key_var), ("BASE_URL", args.base_url_var)):
        target = f"{prefix}_{suffix}"
        if var in os.environ:
            os.environ[target] = os.environ[var]
        elif target not in os.environ:
            sys.exit(f"Set {var} (or {target}) in .env.")
    args.endpoint = prefix


def accept_responses_usage() -> None:
    """Let Inspect read usage that names its fields the Responses-API way.

    The Dartmouth proxy answers some chat completions (e.g. gpt-6-luna) with
    input_tokens/output_tokens and a null prompt_tokens, which Inspect cannot
    subtract from. This copies them into the chat-completions names first.
    """
    from inspect_ai.model._providers import openai_compatible
    from openai.types.completion_usage import CompletionTokensDetails

    original = openai_compatible.model_output_from_openai

    def model_output_from_openai(completion, choices):
        usage = completion.usage
        if usage is not None and usage.prompt_tokens is None:
            extra = usage.model_extra or {}
            usage.prompt_tokens = extra.get("input_tokens") or 0
            usage.completion_tokens = extra.get("output_tokens") or 0
            reasoning = (extra.get("output_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            if usage.completion_tokens_details is None and reasoning is not None:
                usage.completion_tokens_details = CompletionTokensDetails(
                    reasoning_tokens=reasoning
                )
        return original(completion, choices)

    openai_compatible.model_output_from_openai = model_output_from_openai


def list_models(args: argparse.Namespace) -> None:
    """Print the model ids an OpenAI-compatible endpoint serves."""
    base = os.environ[f"{args.endpoint}_BASE_URL"].rstrip("/")
    key = os.environ[f"{args.endpoint}_API_KEY"]
    request = urllib.request.Request(
        f"{base}/models", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        models = json.load(response).get("data", [])
    for model in sorted(models, key=lambda m: m.get("id", "")):
        print(f"openai-api/{args.endpoint.lower()}/{model.get('id')}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command != "models" and not args.model:
        sys.exit("--model is required.")
    use_endpoint(args)
    if args.command == "models":
        list_models(args)
    elif args.command == "estimate":
        result = estimate(args)
        print(json.dumps(result, indent=2) if args.json else "", end="")
        if not args.json:
            print_estimate(result)
    else:
        run(args)


if __name__ == "__main__":
    main()
