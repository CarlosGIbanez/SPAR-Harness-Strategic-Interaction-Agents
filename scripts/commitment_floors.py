"""Result 1: the commitment stage on its own, with no bargaining.

For each instruction level, version and seed, each position gets the briefing
a real episode would give it, is asked the commitment question in the same
side call the episode uses, and its floor is logged next to its expected
fight value, its aggressiveness (floor minus fight value) and its fidelity
(1 if the floor sits on the side of the fight value the level predicts). Each
call is also logged as one line on stderr as it completes, and its row is
appended to the CSV at once, so a rerun with the same --out skips the rows
already there and finishes a partial sweep. Nothing else is played, so a run
is one short call per row.

Example:
    uv run python scripts/commitment_floors.py --model anthropic/claude-sonnet-5-5 \\
        --levels 1 2 3 4 5 --versions strict lottery payment --seeds 20 \\
        --private-fields capability cost_of_conflict --out floors.csv

The first observation of a real episode also carries the round prompt, and for
beta the opponent's opening offer. Here both positions see the briefing and
the round-1 prompt only, so beta is asked before any offer is on the table.
"""

import argparse
import asyncio
import csv
import statistics
import sys
from pathlib import Path

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model

from spar_harness.commitment import (
    COMMITMENT_QUESTION,
    Commitment,
    fidelity,
    instruction_text,
)
from spar_harness.crisis_bargaining import (
    briefing,
    crisis_bargaining_game,
    expected_reservation_value,
)
from spar_harness.crisis_presenter import CRISIS_PRESENTERS
from spar_harness.model_player import (
    ANSWER_PROBE_TOOL,
    SUBMIT_MOVE,
    ask_probe,
)
from spar_harness.priors import draw_types
from spar_harness.schema import (
    EpisodeConfig,
    EpisodeState,
    HorizonTreatment,
    Observation,
    PhaseContext,
    PrivateInfoTreatment,
    Treatments,
    TypePrior,
    UniformPrior,
)

FIELDS = [
    "level",
    "version",
    "seed",
    "repeat",
    "position",
    "capability",
    "cost_of_conflict",
    "opponent_capability",
    "opponent_cost_of_conflict",
    "fight_value",
    "floor",
    "aggressiveness",
    "fidelity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="an Inspect model name")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--versions", nargs="+", default=["strict", "lottery", "payment"]
    )
    parser.add_argument("--seeds", type=int, default=20, help="seeds 0 .. seeds-1")
    parser.add_argument("--repeats", type=int, default=1, help="calls per row")
    parser.add_argument(
        "--no-deal", default="nothing", choices=["status_quo", "nothing"]
    )
    parser.add_argument("--status-quo", type=float, default=0.5)
    parser.add_argument(
        "--private-fields",
        nargs="*",
        default=[],
        choices=["capability", "cost_of_conflict"],
        help="type fields hidden from the opponent, as in the episode treatment",
    )
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--stop-prob", type=float, default=0.1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("floors.csv"))
    return parser.parse_args()


def config_for(args: argparse.Namespace, version: str, seed: int) -> EpisodeConfig:
    prior = TypePrior(
        capability=UniformPrior(low=1.0, high=10.0),
        cost_of_conflict=UniformPrior(low=0.05, high=0.3),
    )
    private = (
        PrivateInfoTreatment(private_fields=set(args.private_fields))
        if args.private_fields
        else None
    )
    return EpisodeConfig(
        episode_id=f"{seed}-floors",
        game=crisis_bargaining_game(
            max_rounds=args.max_rounds,
            stop_prob=args.stop_prob,
            status_quo=args.status_quo,
            no_deal=args.no_deal,
            version=version,
        ),
        treatments=Treatments(
            horizon=HorizonTreatment(known=True), private_info=private
        ),
        priors={"alpha": prior, "beta": prior},
        seed=seed,
    )


async def ask_floor(
    model, config: EpisodeConfig, position: str, level: int
) -> tuple[float, float | None]:
    """The position's expected fight value and the floor it states."""
    state = EpisodeState(types=draw_types(config.priors, config.seed))
    first = briefing(config, state, position)
    role = "proposer" if position == "alpha" else "responder"
    context = PhaseContext(
        round=1,
        role=role,
        action_kind="offer" if role == "proposer" else "response",
        allowed_actions={"offer", "fight"}
        if role == "proposer"
        else {"accept", "reject", "fight"},
    )
    observation = Observation(recipient=position, round=1, briefing=first)
    presenter = CRISIS_PRESENTERS[config.game.version]
    messages = [
        ChatMessageSystem(content=presenter.system_prompt.format(tool=SUBMIT_MOVE)),
        ChatMessageUser(content=presenter.render(position, observation, context)),
    ]
    tools = [presenter.tool(observation, context), ANSWER_PROBE_TOOL]
    question = COMMITMENT_QUESTION.format(instruction=instruction_text(level))
    answer = await ask_probe(model, messages, tools, question)
    floor = None if answer is None else min(max(answer, 0.0), 1.0)
    return expected_reservation_value(position, first), floor


KEY = ("level", "version", "seed", "repeat", "position")


def existing_rows(path: Path) -> list[dict]:
    """The rows already in the CSV, typed like fresh rows; none if no file."""
    if not path.exists():
        return []
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in ("level", "seed", "repeat"):
            row[field] = int(row[field])
        for field in FIELDS[5:12]:
            row[field] = None if row[field] in ("", "None") else float(row[field])
        row["fidelity"] = (
            None if row["fidelity"] in ("", "None") else int(row["fidelity"])
        )
    return rows


def row_key(row: dict) -> tuple:
    return tuple(row[field] for field in KEY)


async def main() -> None:
    args = parse_args()
    model = get_model(args.model)
    semaphore = asyncio.Semaphore(args.concurrency)
    done = existing_rows(args.out)
    have = {row_key(row) for row in done}
    jobs = [
        job
        for job in (
            (level, version, seed, repeat, position)
            for level in args.levels
            for version in args.versions
            for seed in range(args.seeds)
            for repeat in range(args.repeats)
            for position in ("alpha", "beta")
        )
        if job not in have
    ]
    if done:
        print(
            f"{len(done)} rows already in {args.out}, {len(jobs)} to go",
            file=sys.stderr,
        )
    write_lock = asyncio.Lock()
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    file = args.out.open("a", newline="")
    writer = csv.DictWriter(file, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        file.flush()

    async def run(job):
        level, version, seed, repeat, position = job
        config = config_for(args, version, seed)
        async with semaphore:
            fight_value, floor = await ask_floor(model, config, position, level)
        types = draw_types(config.priors, seed)
        other = "beta" if position == "alpha" else "alpha"
        row = {
            "level": level,
            "version": version,
            "seed": seed,
            "repeat": repeat,
            "position": position,
            "capability": types[position].capability,
            "cost_of_conflict": types[position].cost_of_conflict,
            "opponent_capability": types[other].capability,
            "opponent_cost_of_conflict": types[other].cost_of_conflict,
            "fight_value": fight_value,
            "floor": floor,
            "aggressiveness": None if floor is None else floor - fight_value,
            "fidelity": fidelity(Commitment(level=level, floor=floor), fight_value),
        }
        log_call(row)
        async with write_lock:
            writer.writerow(row)
            file.flush()
        return row

    try:
        fresh = await asyncio.gather(*(run(job) for job in jobs))
    finally:
        file.close()
    rows = sorted(done + list(fresh), key=row_key)
    print(f"{len(fresh)} rows written to {args.out} ({len(rows)} in total)")
    summarize(rows)


def log_call(row: dict) -> None:
    """One line per call, as it completes."""

    def fmt(value: float | None) -> str:
        return "none" if value is None else f"{value:.3f}"

    print(
        f"level={row['level']} {row['version']} seed={row['seed']} "
        f"rep={row['repeat']} {row['position']}: floor={fmt(row['floor'])} "
        f"fight_value={row['fight_value']:.3f} "
        f"aggressiveness={fmt(row['aggressiveness'])} fidelity={row['fidelity']}",
        file=sys.stderr,
    )


def summarize(rows: list[dict]) -> None:
    """By level: floor minus fight value (median, mean, share above zero), the
    fidelity rate and the number of unparseable floors."""
    print(
        f"{'level':>5} {'n':>4} {'missing':>7} {'median':>7} {'mean':>7} "
        f"{'above':>6} {'fidelity':>8}"
    )
    for level in sorted({row["level"] for row in rows}):
        at_level = [row for row in rows if row["level"] == level]
        stated = [row for row in at_level if row["floor"] is not None]
        gaps = [row["aggressiveness"] for row in stated]
        missing = len(at_level) - len(gaps)
        if not gaps:
            print(f"{level:>5} {len(at_level):>4} {missing:>7}")
            continue
        above = sum(gap > 0 for gap in gaps) / len(gaps)
        faithful = sum(row["fidelity"] for row in stated) / len(stated)
        print(
            f"{level:>5} {len(at_level):>4} {missing:>7} "
            f"{statistics.median(gaps):>7.3f} {statistics.mean(gaps):>7.3f} "
            f"{above:>6.2f} {faithful:>8.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
