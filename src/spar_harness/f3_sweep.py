"""Print (or run) the `inspect eval` commands for the F3 grid of conditions.

uv run python -m spar_harness.f3_sweep --claude anthropic/<model> --gpt openai/<model> [--run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

CELLS = [
    ("none", "partial"),
    ("declared", "none"),
    ("declared", "partial"),
    ("declared", "full"),
    ("enforced", "partial"),
    ("enforced", "full"),
]
PAIRINGS = {
    "claude-claude": ("claude", "claude"),
    "gpt-gpt": ("gpt", "gpt"),
    "claude-gpt": ("claude", "gpt"),
}
TASK = f"{Path(__file__).with_name('f3_task.py')}@f3_race"


def commands(
    claude: str,
    gpt: str,
    pairings: list[str],
    wording: str = "labs",
    n_episodes: int = 10,
    first_seed: int = 0,
    log_root: str = "logs/f3",
    cells: list[tuple[str, str]] = CELLS,
    interview: bool = True,
    renegotiate: bool = False,
    tell_length: bool = False,
) -> list[list[str]]:
    models = {"claude": claude, "gpt": gpt}
    extra = (["-T", "renegotiate=true"] if renegotiate else []) + (
        ["-T", "tell_length=true"] if tell_length else []
    )
    suffix = ("-reneg" if renegotiate else "") + ("-len" if tell_length else "")
    out = []
    for pairing in pairings:
        seat_a, seat_b = PAIRINGS[pairing]
        for commitment, verification in cells:
            out.append(
                [
                    "inspect",
                    "eval",
                    TASK,
                    "-T",
                    f"commitment={commitment}",
                    "-T",
                    f"verification={verification}",
                    "-T",
                    f"wording={wording}",
                    "-T",
                    f"n_episodes={n_episodes}",
                    "-T",
                    f"first_seed={first_seed}",
                    "-T",
                    f"interview={str(interview).lower()}",
                    *extra,
                    "--model-role",
                    f"a={models[seat_a]}",
                    "--model-role",
                    f"b={models[seat_b]}",
                    "--log-dir",
                    f"{log_root}/{wording}/{pairing}/{commitment}-{verification}{suffix}",
                ]
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--claude", required=True, help="Inspect model name for the Anthropic model"
    )
    parser.add_argument(
        "--gpt", required=True, help="Inspect model name for the OpenAI model"
    )
    parser.add_argument(
        "--pairings", nargs="+", default=list(PAIRINGS), choices=list(PAIRINGS)
    )
    parser.add_argument("--wording", default="labs", choices=["labs", "neutral"])
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="seeds per cell; each seed runs twice",
    )
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--log-root", default="logs/f3")
    parser.add_argument("--no-interview", action="store_true")
    parser.add_argument(
        "--tell-length",
        action="store_true",
        help="tell the labs the expected number of rounds",
    )
    parser.add_argument(
        "--renegotiate",
        action="store_true",
        help="a new negotiation before every round without an agreement",
    )
    parser.add_argument(
        "--run", action="store_true", help="run the commands instead of printing them"
    )
    args = parser.parse_args()
    for command in commands(
        args.claude,
        args.gpt,
        args.pairings,
        args.wording,
        args.n_episodes,
        args.first_seed,
        args.log_root,
        interview=not args.no_interview,
        renegotiate=args.renegotiate,
        tell_length=args.tell_length,
    ):
        if args.run:
            print("running:", " ".join(command), flush=True)
            if subprocess.run(command, check=False).returncode != 0:
                sys.exit(
                    "stopped: a run failed (see its log); fix it and rerun with a later --pairings or cell."
                )
        else:
            print(" ".join(command))


if __name__ == "__main__":
    main()
