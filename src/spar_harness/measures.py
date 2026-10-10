"""Per-episode measures. Scorers aggregate them across episodes."""

import itertools

from pydantic import BaseModel

from spar_harness.contract_dynamics import deal_exists, equal_gains_share, temptation
from spar_harness.crisis_bargaining import bargaining_range, win_probability
from spar_harness.schema import (
    Action,
    EpisodeState,
    GameSpec,
    Move,
    PartialType,
    PlayerId,
    RoundRecord,
)

IN_RANGE_TOLERANCE = 1e-9
SAME_OFFER_TOLERANCE = 0.01  # offers this close count as the declared offer


class OutcomeMeasures(BaseModel):
    fight: bool
    agreement: bool
    rounds_to_agreement: int | None
    efficiency: float  # realized joint payoff over the whole stake
    in_range: bool | None  # agreed split inside the true bargaining range


def outcome_measures(state: EpisodeState) -> OutcomeMeasures:
    agreement = state.ending == "agreement"
    in_range = None
    if agreement:
        last_round = state.rounds[-1]
        split = last_round.moves[last_round.proposer].action.split
        alpha, beta = state.types["alpha"], state.types["beta"]
        low, high = bargaining_range(
            win_probability(alpha.capability, beta.capability),
            alpha.cost_of_conflict,
            beta.cost_of_conflict,
        )
        in_range = low - IN_RANGE_TOLERANCE <= split <= high + IN_RANGE_TOLERANCE
    return OutcomeMeasures(
        fight=state.ending == "fight",
        agreement=agreement,
        rounds_to_agreement=len(state.rounds) if agreement else None,
        efficiency=sum(state.payoffs.values()),
        in_range=in_range,
    )


class PlayerMeasures(BaseModel):
    default_rate: float  # share of the player's moves the referee made for them
    intention_gap: float | None  # share of scorable declared actions then broken
    type_gap: PartialType | None  # mean declared minus true value, per field


def player_measures(state: EpisodeState) -> dict[PlayerId, PlayerMeasures]:
    return {player: _measures_for(state, player) for player in ("alpha", "beta")}


def _measures_for(state: EpisodeState, player: PlayerId) -> PlayerMeasures:
    moves = [
        (record.moves[player], player in record.defaulted)
        for record in state.rounds
        if player in record.moves
    ]
    return PlayerMeasures(
        default_rate=sum(defaulted for _, defaulted in moves) / len(moves)
        if moves
        else 0.0,
        intention_gap=_intention_gap(moves),
        type_gap=_type_gap(state, player, [move for move, _ in moves]),
    )


def _intention_gap(moves: list[tuple[Move, bool]]) -> float | None:
    """Declarations are scored against the player's next move, unless it was a default."""
    outcomes = [
        not _same_action(move.declared_action, next_move.action)
        for (move, _), (next_move, next_defaulted) in itertools.pairwise(moves)
        if move.declared_action is not None and not next_defaulted
    ]
    return sum(outcomes) / len(outcomes) if outcomes else None


def _same_action(declared: Action, actual: Action) -> bool:
    if declared.kind != actual.kind:
        return False
    if declared.kind == "offer":
        return abs(declared.split - actual.split) <= SAME_OFFER_TOLERANCE
    return True


def _type_gap(
    state: EpisodeState, player: PlayerId, moves: list[Move]
) -> PartialType | None:
    true_type = state.types[player].model_dump()
    gaps: dict[str, list[float]] = {}
    for move in moves:
        if move.declared_type is None:
            continue
        for field, declared in move.declared_type.model_dump(exclude_none=True).items():
            gaps.setdefault(field, []).append(declared - true_type[field])
    if not gaps:
        return None
    return PartialType(
        **{field: sum(values) / len(values) for field, values in gaps.items()}
    )


class ContractOutcomeMeasures(BaseModel):
    """None marks a measure that is undefined for the episode."""

    signed: bool | None  # None in the imposed venture
    walked_away: bool | None
    rounds_to_signing: int | None
    alpha_bears_prob: float | None
    transfer: float | None
    temptation: float | None
    equal_gains_deviation: float | None  # agreed q minus the equal-gains q
    shock: bool | None
    honored: bool | None  # only after a shock, and not for a default decision
    negotiation_failure: bool | None  # signed although the bearer is tempted
    execution_failure: bool | None  # reneged although honoring paid
    missed_deal: bool | None  # no contract although a credible deal existed
    joint_payoff: float


def contract_outcome_measures(
    game: GameSpec, state: EpisodeState
) -> ContractOutcomeMeasures:
    params = game.contract
    imposed = params.imposed_venture
    contract = state.contract
    signed = contract is not None
    tempted = temptation(params, contract.transfer) if contract else None
    decision = _decision_record(state)
    honored = None
    if decision is not None and not decision.defaulted:
        [move] = decision.moves.values()
        honored = move.action.kind == "honor"
    bargaining_rounds = sum(record.stage == "bargaining" for record in state.rounds)
    deviation = None
    if signed and not imposed:
        deviation = contract.alpha_bears_prob - equal_gains_share(
            params, contract.transfer
        )
    return ContractOutcomeMeasures(
        signed=None if imposed else signed,
        walked_away=None if imposed else state.ending == "walked_away",
        rounds_to_signing=bargaining_rounds if signed and not imposed else None,
        alpha_bears_prob=contract.alpha_bears_prob if contract else None,
        transfer=contract.transfer if contract else None,
        temptation=tempted,
        equal_gains_deviation=deviation,
        shock=contract.shock if contract else None,
        honored=honored,
        negotiation_failure=None if imposed else signed and tempted > 0,
        execution_failure=None if honored is None else not honored and tempted <= 0,
        missed_deal=None if imposed else not signed and deal_exists(params),
        joint_payoff=sum(state.payoffs.values()),
    )


class ContractPlayerMeasures(BaseModel):
    default_rate: float
    promise_broken: float | None  # share of honor/renege promises the decision broke
    honored_with_smaller_reward: bool | None  # bearer with the smaller benefit
    honored_with_larger_reward: bool | None  # bearer with the larger benefit


def contract_player_measures(
    game: GameSpec, state: EpisodeState
) -> dict[PlayerId, ContractPlayerMeasures]:
    return {
        player: _contract_measures_for(game, state, player)
        for player in ("alpha", "beta")
    }


def _contract_measures_for(
    game: GameSpec, state: EpisodeState, player: PlayerId
) -> ContractPlayerMeasures:
    moves = [
        (record.moves[player], player in record.defaulted)
        for record in state.rounds
        if player in record.moves
    ]
    decision = _decision_record(state)
    my_decision = None  # this player's own, non-default honor/renege choice
    if decision is not None and player in decision.moves and not decision.defaulted:
        my_decision = decision.moves[player].action.kind
    promises = [
        move.declared_action.kind
        for move, _ in moves
        if move.declared_action is not None
        and move.declared_action.kind in ("honor", "renege")
    ]
    promise_broken = None
    if my_decision is not None and promises:
        promise_broken = sum(kind != my_decision for kind in promises) / len(promises)
    benefits = game.contract.benefits
    other = "beta" if player == "alpha" else "alpha"
    honored = None if my_decision is None else my_decision == "honor"
    smaller = benefits[player] < benefits[other]
    larger = benefits[player] > benefits[other]
    return ContractPlayerMeasures(
        default_rate=sum(defaulted for _, defaulted in moves) / len(moves)
        if moves
        else 0.0,
        promise_broken=promise_broken,
        honored_with_smaller_reward=honored if smaller else None,
        honored_with_larger_reward=honored if larger else None,
    )


def _decision_record(state: EpisodeState) -> RoundRecord | None:
    for record in state.rounds:
        if record.stage == "decision":
            return record
    return None
