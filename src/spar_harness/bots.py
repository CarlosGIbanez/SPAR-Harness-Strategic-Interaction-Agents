"""Scripted players. Each bot is a Player: it sees only what a model player sees."""

import bisect
import itertools
import random
from typing import Literal

from spar_harness.contract_dynamics import (
    contract_value,
    credible_transfers,
    deal_exists,
    equal_gains_share,
    temptation,
)
from spar_harness.crisis_bargaining import (
    expected_reservation_value,
    possible_opponent_types,
    reservation_against,
)
from spar_harness.referee import Player
from spar_harness.schema import (
    Briefing,
    Choice,
    ContractBriefing,
    ContractOffer,
    ContractParams,
    Move,
    Observation,
    Offer,
    PartialType,
    PhaseContext,
    PlayerId,
    PrivateType,
)

TOLERANCE = 1e-9  # so that offering exactly the reservation value is accepted
TRANSFER_GRID = 101  # transfers the rational contract bot considers


def random_bot(seed: int) -> Player:
    """Picks uniformly among the allowed actions; offers are uniform splits."""
    rng = random.Random(seed)

    async def play(observation: Observation, context: PhaseContext) -> Move:
        kind = rng.choice(sorted(context.allowed_actions))
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


def rational_contract_bot() -> Player:
    """The theory benchmark for contract dynamics.

    Proposer: walks away if no deal exists; otherwise proposes the credible
    contract that maximizes the smaller side's value (the cheapest transfer on
    ties). Responder: signs iff the contract is worth more than nothing, assuming
    the bearer honors exactly when its temptation is at most zero, and otherwise
    walks away. Bearer: honors iff its temptation is at most zero. It sends no
    messages.
    """
    briefing: ContractBriefing | None = None

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal briefing
        if observation.briefing is not None:
            briefing = observation.briefing
        params = briefing.params
        me = observation.recipient
        if context.action_kind == "message":
            return Move(message="")
        if context.action_kind == "decision":
            honors = temptation(params, observation.contract.transfer) <= 0
            return Move(action=Choice(kind="honor" if honors else "renege"))
        if context.role == "proposer":
            offer = _best_credible_contract(params)
            return Move(action=offer if offer else Choice(kind="walk_away"))
        offer = observation.opponent_moves[-1].action
        value = contract_value(
            me,
            params,
            offer.alpha_bears_prob,
            offer.transfer,
            bearer_honors=temptation(params, offer.transfer) <= 0,
        )
        return Move(action=Choice(kind="accept" if value > TOLERANCE else "walk_away"))

    return play


def _best_credible_contract(params: ContractParams) -> ContractOffer | None:
    if not deal_exists(params):
        return None
    low, high = credible_transfers(params)
    best, best_value = None, 0.0
    for i in range(TRANSFER_GRID):
        transfer = low + (high - low) * i / (TRANSFER_GRID - 1)
        q = equal_gains_share(params, transfer)
        smaller_value = min(
            contract_value(side, params, q, transfer, bearer_honors=True)
            for side in ("alpha", "beta")
        )
        if smaller_value > best_value + TOLERANCE:
            best = ContractOffer(alpha_bears_prob=q, transfer=transfer)
            best_value = smaller_value
    return best


def scripted_contract_bot(
    decision: Literal["honor", "renege"], appeal: str = ""
) -> Player:
    """A fixed counterparty: proposes the fair coin without transfer, signs any
    offer, sends `appeal` in every persuasion round, and as bearer always makes
    `decision`."""

    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.action_kind == "message":
            return Move(message=appeal if context.role == "other" else "")
        if context.action_kind == "decision":
            return Move(action=Choice(kind=decision))
        if context.role == "proposer":
            return Move(action=ContractOffer(alpha_bears_prob=0.5, transfer=0.0))
        return Move(action=Choice(kind="accept"))

    return play
