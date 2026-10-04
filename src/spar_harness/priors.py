"""Drawing private types from priors."""

import random

from spar_harness.schema import PlayerId, PrivateType, TypePrior


def draw_types(
    priors: dict[PlayerId, TypePrior], seed: int
) -> dict[PlayerId, PrivateType]:
    rng = random.Random(seed)
    return {
        player: PrivateType(
            capability=rng.uniform(
                priors[player].capability.low, priors[player].capability.high
            ),
            cost_of_conflict=rng.uniform(
                priors[player].cost_of_conflict.low,
                priors[player].cost_of_conflict.high,
            ),
        )
        for player in ("alpha", "beta")
    }
