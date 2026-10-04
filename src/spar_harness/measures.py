"""Per-episode measures. Scorers aggregate them across episodes."""

import itertools

from pydantic import BaseModel

from spar_harness.crisis_bargaining import bargaining_range, win_probability
from spar_harness.schema import Action, EpisodeState, Move, PartialType, PlayerId

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
