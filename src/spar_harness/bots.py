"""Scripted players. Each bot is a Player: it sees only what a model player sees."""

import bisect
import itertools
import random

from spar_harness.crisis_bargaining import (
    expected_reservation_value,
    possible_opponent_types,
    reservation_against,
)
from spar_harness.referee import ALLOWED_ACTIONS, Player
from spar_harness.schema import (
    Briefing,
    Choice,
    Move,
    Observation,
    Offer,
    PartialType,
    PhaseContext,
    PlayerId,
    PrivateType,
)

TOLERANCE = 1e-9  # so that offering exactly the reservation value is accepted


def random_bot(seed: int) -> Player:
    """Picks uniformly among the allowed actions; offers are uniform splits."""
    rng = random.Random(seed)

    async def play(observation: Observation, context: PhaseContext) -> Move:
        kind = rng.choice(sorted(ALLOWED_ACTIONS[context.action_kind]))
        if kind == "offer":
            return Move(action=Offer(split=rng.random()))
        return Move(action=Choice(kind=kind))

    return play


def equilibrium_bot() -> Player:
    """Fearon's one-shot best response, applied every round.

    As proposer it makes the offer that maximizes its expected payoff given the
    prior over the opponent's type, treating a rejection as a fight. As responder
    it accepts any share at or above its (expected) reservation value and fights
    otherwise. It declares its private fields truthfully whenever it may, and
    never sends messages.
    """
    briefing: Briefing | None = None

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal briefing
        if observation.briefing is not None:
            briefing = observation.briefing
        me = observation.recipient
        declared_type = _truthful_declaration(briefing.own_type, context)
        if context.role == "proposer":
            opponents = possible_opponent_types(briefing)
            my_share = _risk_return_share(me, briefing.own_type, opponents)
            split = my_share if me == "alpha" else 1 - my_share
            return Move(action=Offer(split=split), declared_type=declared_type)
        offer = observation.opponent_moves[-1].action
        my_share = offer.split if me == "alpha" else 1 - offer.split
        reservation = expected_reservation_value(me, briefing)
        kind = "accept" if my_share >= reservation - TOLERANCE else "fight"
        return Move(action=Choice(kind=kind), declared_type=declared_type)

    return play


def _truthful_declaration(
    own: PrivateType, context: PhaseContext
) -> PartialType | None:
    if "declared_type" not in context.talk_fields:
        return None
    return PartialType(**own.model_dump(include=context.declarable_fields))


def _risk_return_share(
    me: PlayerId, own: PrivateType, opponents: list[PrivateType]
) -> float:
    """The share to keep that maximizes expected payoff against the opponent prior.

    Assumes each possible opponent accepts exactly when its share reaches its own
    reservation value. Expected payoff only changes at those thresholds, so the
    best offer is one of them.
    """
    outcomes = sorted(
        (
            reservation_against(_other(me), opponent, own),
            reservation_against(me, own, opponent),
        )
        for opponent in opponents
    )
    thresholds = [threshold for threshold, _ in outcomes]
    fight_payoffs = list(itertools.accumulate(fight for _, fight in outcomes))
    total_fight = fight_payoffs[-1]

    def expected_payoff(offered: float) -> float:
        accepted = bisect.bisect_right(thresholds, offered + TOLERANCE)
        rejected_fight = total_fight - (
            fight_payoffs[accepted - 1] if accepted else 0.0
        )
        return (accepted * (1 - offered) + rejected_fight) / len(outcomes)

    candidates = sorted(
        {min(max(threshold, 0.0), 1.0) for threshold in thresholds} | {0.0}
    )
    best_payoff = max(expected_payoff(offered) for offered in candidates)
    # Among (near-)equally good offers, make the most generous: it risks the least war.
    best_offer = max(
        offered
        for offered in candidates
        if expected_payoff(offered) >= best_payoff - TOLERANCE
    )
    return 1 - best_offer


def _other(player: PlayerId) -> PlayerId:
    return "beta" if player == "alpha" else "alpha"
