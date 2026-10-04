"""Rules of the crisis bargaining game (Fearon 1995)."""

import itertools

from spar_harness.schema import (
    Briefing,
    Choice,
    Ending,
    EpisodeState,
    GameSpec,
    Move,
    Offer,
    PhaseContext,
    PhaseSpec,
    PlayerId,
    PrivateType,
    TerminationSpec,
    UniformPrior,
)

GRID_POINTS = 200  # per unknown type field when integrating over a prior


def win_probability(capability_alpha: float, capability_beta: float) -> float:
    """Chance that alpha wins a fight."""
    return capability_alpha / (capability_alpha + capability_beta)


def reservation_value(
    player: PlayerId, win_probability: float, cost_of_conflict: float
) -> float:
    """Lowest share of the stake the player accepts rather than fight."""
    own_win_probability = win_probability if player == "alpha" else 1 - win_probability
    return own_win_probability - cost_of_conflict


def bargaining_range(
    win_probability: float, cost_alpha: float, cost_beta: float
) -> tuple[float, float]:
    """Alpha's shares of the stake that both players prefer to fighting."""
    low = reservation_value("alpha", win_probability, cost_alpha)
    high = 1 - reservation_value("beta", win_probability, cost_beta)
    return max(low, 0.0), min(high, 1.0)


def outcome_payoffs(
    ending: Ending,
    types: dict[PlayerId, PrivateType],
    status_quo: float,
    split: float | None = None,
) -> dict[PlayerId, float]:
    """Each player's payoff when the episode ends."""
    if ending == "fight":
        p = win_probability(types["alpha"].capability, types["beta"].capability)
        return {
            player: reservation_value(player, p, types[player].cost_of_conflict)
            for player in ("alpha", "beta")
        }
    if ending == "agreement":
        return {"alpha": split, "beta": 1 - split}
    return {"alpha": status_quo, "beta": 1 - status_quo}


def crisis_bargaining_game(
    max_rounds: int, stop_prob: float, status_quo: float
) -> GameSpec:
    """Alternating offers: each round the proposer offers and the responder answers."""
    return GameSpec(
        game_id="crisis_bargaining",
        phases=[
            PhaseSpec(kind="act", actors="proposer", action_kind="offer"),
            PhaseSpec(kind="act", actors="responder", action_kind="response"),
        ],
        proposer_rule="alternate",
        termination=TerminationSpec(max_rounds=max_rounds, stop_prob=stop_prob),
        status_quo=status_quo,
    )


def ending_after_move(move: Move) -> Ending | None:
    if move.action.kind == "accept":
        return "agreement"
    if move.action.kind == "fight":
        return "fight"
    return None


def episode_payoffs(game: GameSpec, state: EpisodeState) -> dict[PlayerId, float]:
    split = None
    if state.ending == "agreement":
        last_round = state.rounds[-1]
        split = last_round.moves[last_round.proposer].action.split
    return outcome_payoffs(state.ending, state.types, game.status_quo, split)


def default_move(
    game: GameSpec, state: EpisodeState, player: PlayerId, context: PhaseContext
) -> Move:
    """The move made for a player who keeps failing. Never a fight."""
    if context.role == "responder":
        return Move(action=Choice(kind="reject"))
    for record in reversed(state.rounds):
        own_move = record.moves.get(player)
        if record.proposer == player and own_move and own_move.action.kind == "offer":
            return own_move
    return Move(action=Offer(split=game.status_quo))


def expected_reservation_value(player: PlayerId, briefing: Briefing) -> float:
    """The player's reservation value, averaged over what it does not know."""
    opponents = possible_opponent_types(briefing)
    own = briefing.own_type
    return sum(
        reservation_against(player, own, opponent) for opponent in opponents
    ) / len(opponents)


def reservation_against(
    player: PlayerId, own: PrivateType, opponent: PrivateType
) -> float:
    """The player's reservation value if the two types are as given."""
    if player == "alpha":
        p = win_probability(own.capability, opponent.capability)
    else:
        p = win_probability(opponent.capability, own.capability)
    return reservation_value(player, p, own.cost_of_conflict)


def possible_opponent_types(briefing: Briefing) -> list[PrivateType]:
    """A grid over the opponent's prior, with known fields fixed to their values."""
    known = briefing.opponent_type
    prior = briefing.opponent_prior
    capabilities = (
        [known.capability] if known.capability is not None else _grid(prior.capability)
    )
    costs = (
        [known.cost_of_conflict]
        if known.cost_of_conflict is not None
        else _grid(prior.cost_of_conflict)
    )
    return [
        PrivateType(capability=k, cost_of_conflict=c)
        for k, c in itertools.product(capabilities, costs)
    ]


def _grid(prior: UniformPrior) -> list[float]:
    if prior.low == prior.high:
        return [prior.low]
    step = (prior.high - prior.low) / GRID_POINTS
    return [prior.low + (i + 0.5) * step for i in range(GRID_POINTS)]
