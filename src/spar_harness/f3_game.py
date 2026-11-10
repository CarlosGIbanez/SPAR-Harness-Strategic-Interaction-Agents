"""Rules of the F3 game: type draws, dynamics, risk, payoffs, matched random draws."""

import math
import random

from spar_harness.f3_schema import (
    Act,
    EpisodeConfig,
    LabId,
    LabType,
    Params,
    rival_of,
)


def rng(purpose: str, seed: int, *keys: object) -> random.Random:
    """A random stream fixed by (purpose, seed, keys), so that every condition
    sees the same draws (matched seeds) whatever the labs do."""
    return random.Random("-".join([purpose, str(seed), *map(str, keys)]))


def draw_types(config: EpisodeConfig) -> dict[LabId, LabType]:
    """Private k and c, uniform on the ranges, rounded to 2 decimals so that the
    numbers in the prompts are exactly the true values."""
    if config.fixed_types is not None:
        return config.fixed_types
    p, draw = config.params, rng("types", config.seed)
    return {
        lab: LabType(
            k=round(draw.uniform(p.k_low, p.k_high), 2),
            c=round(draw.uniform(p.c_low, p.c_high), 2),
        )
        for lab in ("A", "B")
    }


def draw_proposer(config: EpisodeConfig) -> LabId:
    return rng("proposer", config.seed).choice(["A", "B"])


def risk(
    capability: dict[LabId, float],
    threshold: float,
    kappa: float,
    background: float = 0.0,
) -> tuple[float, float]:
    """Frontier distance D and catastrophe probability p = 1 - exp(-(kappa D + b)),
    where b is the accumulated background hazard (zero unless switched on)."""
    distance = max(0.0, max(capability.values()) - threshold)
    return distance, 1.0 - math.exp(-(kappa * distance + background))


def background_step(config: EpisodeConfig, round_number: int) -> float:
    """This round's random increase of the background hazard, the same in every condition."""
    return rng("background", config.seed, round_number).uniform(
        0.0, config.params.background_max
    )


def round_payoff(
    lab: LabId,
    capability: dict[LabId, float],
    action: Act,
    types: dict[LabId, LabType],
    a: float,
) -> float:
    """u = 1 + a (own - rival capability) - c (1 if slowed down), after the update."""
    lead = capability[lab] - capability[rival_of(lab)]
    return 1.0 + a * lead - (types[lab].c if action == "slow" else 0.0)


def report_about(
    verification: str,
    actual: Act,
    q: float,
    seed: int,
    round_number: int,
    receiver: LabId,
) -> Act | None:
    """The signal the receiver gets about the rival's actual action."""
    if verification == "none":
        return None
    if (
        verification == "full"
        or rng("signal", seed, round_number, receiver).random() < q
    ):
        return actual
    return "slow" if actual == "accelerate" else "accelerate"


def mutual_slowdown_benchmark(types: dict[LabId, LabType], params: Params) -> float:
    """Joint payoff if both labs slow down in every round, expected over the
    stopping rule (no catastrophe, since mutual slowdown keeps the frontier at
    the threshold)."""
    weight = sum(params.continue_prob**t for t in range(params.max_rounds))
    return weight * sum(1.0 - types[lab].c for lab in ("A", "B"))
