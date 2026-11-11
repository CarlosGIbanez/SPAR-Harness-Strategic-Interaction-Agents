"""Per-seed comparison of two arms run on the same seeds.

Joins the samples of the two arms on seed and seat, takes the per-seed
difference of one measure (arm B minus arm A, averaged over the pair of
episodes of each seed and over repeated runs), and reports the mean difference
with a bootstrap interval over seeds, the share of seeds where arm B is worse
by more than delta, and the within-seed spread of each arm across its repeats.

Example, three repeats per arm:
    uv run python scripts/seed_diff.py --a logs/arm1-*.eval --b logs/arm3-*.eval \\
        --measure fight --delta 0.05

Measures are the keys of the outcome scorer (fight, agreement, efficiency,
...), which belong to the seed, or of the seat scorer (a_fidelity, b_floor,
...), which belong to one seat. A seat measure is compared for that seat.
"""

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

SCORERS = ("outcome_scorer", "seat_scorer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--a", nargs="+", required=True, type=Path, help="arm A logs")
    parser.add_argument("--b", nargs="+", required=True, type=Path, help="arm B logs")
    parser.add_argument("--measure", default="fight")
    parser.add_argument(
        "--delta", type=float, default=0.05, help="a seed is unsafe when B - A > delta"
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0, help="for the bootstrap")
    return parser.parse_args()


def values_by_seed(paths: list[Path], measure: str) -> dict[int, list[float]]:
    """Every value of the measure per seed: one per episode of each pair, from
    every run. None values (the measure was undefined) are left out."""
    values: dict[int, list[float]] = defaultdict(list)
    for path in paths:
        log = read_eval_log(str(path))
        for sample in log.samples:
            seed = int(str(sample.id).split("-")[0])
            for scorer in SCORERS:
                score = sample.scores.get(scorer)
                if score is not None and measure in score.value:
                    value = score.value[measure]
                    if value is not None:
                        values[seed].append(float(value))
    return values


def within_seed_sd(values: dict[int, list[float]]) -> float | None:
    """Mean over seeds of the standard deviation across a seed's values."""
    spreads = [
        statistics.stdev(seed_values)
        for seed_values in values.values()
        if len(seed_values) > 1
    ]
    return statistics.mean(spreads) if spreads else None


def bootstrap(
    differences: list[float], statistic, draws: int, rng: random.Random
) -> tuple[float, float]:
    """A 95% percentile interval of the statistic over resampled seeds."""
    samples = sorted(
        statistic(rng.choices(differences, k=len(differences))) for _ in range(draws)
    )
    return samples[int(0.025 * draws)], samples[min(int(0.975 * draws), draws - 1)]


def main() -> None:
    args = parse_args()
    a = values_by_seed(args.a, args.measure)
    b = values_by_seed(args.b, args.measure)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise SystemExit(f"no seed has {args.measure!r} defined in both arms")
    differences = [
        statistics.mean(b[seed]) - statistics.mean(a[seed]) for seed in seeds
    ]
    rng = random.Random(args.seed)
    mean = statistics.mean(differences)
    mean_low, mean_high = bootstrap(differences, statistics.mean, args.bootstrap, rng)

    def unsafe_share(diffs: list[float]) -> float:
        return sum(diff > args.delta for diff in diffs) / len(diffs)

    unsafe = unsafe_share(differences)
    unsafe_low, unsafe_high = bootstrap(differences, unsafe_share, args.bootstrap, rng)

    print(f"measure: {args.measure}, B minus A per seed, {len(seeds)} seeds")
    print(f"{'seed':>4} {'A':>7} {'B':>7} {'diff':>7} {'n_A':>4} {'n_B':>4}")
    for seed, diff in zip(seeds, differences):
        print(
            f"{seed:>4} {statistics.mean(a[seed]):>7.3f} {statistics.mean(b[seed]):>7.3f} "
            f"{diff:>7.3f} {len(a[seed]):>4} {len(b[seed]):>4}"
        )
    print()
    print(
        f"mean difference: {mean:.3f}  95% bootstrap [{mean_low:.3f}, {mean_high:.3f}]"
    )
    print(
        f"unsafe share (diff > {args.delta:g}): {unsafe:.2f}  "
        f"95% bootstrap [{unsafe_low:.2f}, {unsafe_high:.2f}]"
    )
    for name, values in (("A", a), ("B", b)):
        spread = within_seed_sd({seed: values[seed] for seed in seeds})
        shown = "n/a (one value per seed)" if spread is None else f"{spread:.3f}"
        print(f"within-seed sd, arm {name}: {shown}")


if __name__ == "__main__":
    main()
