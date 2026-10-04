"""The referee: runs one episode of any game described by a GameSpec."""

import random
from collections.abc import Awaitable, Callable
from typing import Literal

from spar_harness import crisis_bargaining
from spar_harness.priors import draw_types
from spar_harness.schema import (
    Briefing,
    Ending,
    EpisodeConfig,
    EpisodeState,
    Move,
    Observation,
    PartialType,
    PhaseContext,
    PlayerId,
    RoundRecord,
    TalkField,
    TypeField,
)


class InvalidMove(Exception):
    """A move the referee cannot accept; the message says why."""


Player = Callable[[Observation, PhaseContext], Awaitable[Move]]

MAX_RETRIES = 2
TALK_FIELDS: tuple[TalkField, ...] = ("message", "declared_action", "declared_type")
ALLOWED_ACTIONS = {
    "offer": {"offer", "fight"},
    "response": {"accept", "reject", "fight"},
}

ENDING_RULES = {"crisis_bargaining": crisis_bargaining.ending_after_move}
PAYOFF_MODELS = {"crisis_bargaining": crisis_bargaining.episode_payoffs}
DEFAULT_MOVES = {"crisis_bargaining": crisis_bargaining.default_move}


async def run_episode(
    config: EpisodeConfig, players: dict[PlayerId, Player]
) -> EpisodeState:
    game = config.game
    state = EpisodeState(types=draw_types(config.priors, config.seed))
    stop_rng = random.Random(f"random_stop-{config.seed}")
    inbox: dict[PlayerId, list[Move]] = {"alpha": [], "beta": []}
    briefings = {
        player: _briefing(config, state, player) for player in ("alpha", "beta")
    }
    max_rounds = game.termination.max_rounds
    for round_number in range(1, max_rounds + 1):
        state.ending = await _play_round(
            config, players, state, inbox, briefings, round_number
        )
        if state.ending is not None:
            break
        if round_number < max_rounds and stop_rng.random() < game.termination.stop_prob:
            state.ending = "random_stop"
            break
    else:
        state.ending = "horizon"
    state.payoffs = PAYOFF_MODELS[game.game_id](game, state)
    return state


async def _play_round(
    config: EpisodeConfig,
    players: dict[PlayerId, Player],
    state: EpisodeState,
    inbox: dict[PlayerId, list[Move]],
    briefings: dict[PlayerId, Briefing],
    round_number: int,
) -> Ending | None:
    game = config.game
    proposer = _proposer(game.proposer_rule, round_number)
    record = RoundRecord(round=round_number, proposer=proposer)
    state.rounds.append(record)
    for phase in game.phases:
        player = proposer if phase.actors == "proposer" else _opponent(proposer)
        context = PhaseContext(
            round=round_number,
            role=phase.actors,
            action_kind=phase.action_kind,
            talk_fields=_talk_fields(config),
            declarable_fields=_private_fields(config),
        )
        observation = Observation(
            recipient=player,
            round=round_number,
            briefing=briefings.pop(player, None),
            opponent_moves=inbox[player],
        )
        inbox[player] = []
        move = await _request_move(players[player], observation, context)
        if move is None:
            move = DEFAULT_MOVES[game.game_id](game, state, player, context)
            record.defaulted.add(player)
        record.moves[player] = move
        inbox[_opponent(player)].append(move)
        ending = ENDING_RULES[game.game_id](move)
        if ending is not None:
            return ending
    return None


async def _request_move(
    player: Player, observation: Observation, context: PhaseContext
) -> Move | None:
    """Ask for a move, retrying with the reason whenever it is invalid.

    Returns None if every attempt was invalid.
    """
    for _ in range(1 + MAX_RETRIES):
        try:
            move = await player(observation, context)
            _validate(move, context)
            return move
        except InvalidMove as invalid:
            # A retry carries no new information, only the reason.
            observation = Observation(
                recipient=observation.recipient, round=observation.round
            )
            context = context.model_copy(update={"error": str(invalid)})
    return None


def _validate(move: Move, context: PhaseContext) -> None:
    allowed = ALLOWED_ACTIONS[context.action_kind]
    if move.action.kind not in allowed:
        raise InvalidMove(
            f"'{move.action.kind}' is not allowed for the {context.role}; "
            f"choose one of: {', '.join(sorted(allowed))}"
        )
    used_talk = {field for field in TALK_FIELDS if getattr(move, field) is not None}
    closed = used_talk - context.talk_fields
    if closed:
        raise InvalidMove(
            f"these fields are not available: {', '.join(sorted(closed))}"
        )
    if move.declared_type is not None:
        claimed = set(move.declared_type.model_dump(exclude_none=True))
        undeclarable = claimed - context.declarable_fields
        if undeclarable:
            raise InvalidMove(
                f"a declared type may only claim private fields, not: "
                f"{', '.join(sorted(undeclarable))}"
            )


def _talk_fields(config: EpisodeConfig) -> set[TalkField]:
    talk = config.treatments.talk
    if talk is None:
        return set()
    switches = {
        "message": talk.messages,
        "declared_action": talk.declared_action,
        "declared_type": talk.declared_type and bool(_private_fields(config)),
    }
    return {field for field, on in switches.items() if on}


def _private_fields(config: EpisodeConfig) -> set[TypeField]:
    private_info = config.treatments.private_info
    return private_info.private_fields if private_info else set()


def _briefing(config: EpisodeConfig, state: EpisodeState, player: PlayerId) -> Briefing:
    opponent = _opponent(player)
    private_info = config.treatments.private_info
    private_fields = _private_fields(config)
    public_opponent_type = state.types[opponent].model_dump(exclude=private_fields)
    return Briefing(
        own_type=state.types[player],
        opponent_type=PartialType(**public_opponent_type),
        opponent_prior=config.priors[opponent] if private_info else None,
        private_fields=private_fields,
        status_quo=config.game.status_quo,
        termination=config.game.termination
        if config.treatments.horizon.known
        else None,
    )


def _proposer(
    rule: Literal["alternate", "alpha", "beta"], round_number: int
) -> PlayerId:
    if rule == "alternate":
        return "alpha" if round_number % 2 == 1 else "beta"
    return rule


def _opponent(player: PlayerId) -> PlayerId:
    return "beta" if player == "alpha" else "alpha"
