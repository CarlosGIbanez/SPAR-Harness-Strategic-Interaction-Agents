"""The referee: runs one episode of any game described by a GameSpec.

A game is a sequence of stages. Each stage repeats its phase list for up to
max_rounds rounds. Everything game-specific is looked up in the plugin dicts
below, keyed by game_id.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Literal

from spar_harness import contract_dynamics, crisis_bargaining
from spar_harness.priors import draw_types
from spar_harness.schema import (
    Briefing,
    Ending,
    EpisodeConfig,
    EpisodeState,
    GameSpec,
    Move,
    Observation,
    PhaseContext,
    PhaseSpec,
    PlayerId,
    RoundRecord,
    StageSpec,
    TalkField,
    TypeField,
)


class InvalidMove(Exception):
    """A move the referee cannot accept; the message says why."""


Player = Callable[[Observation, PhaseContext], Awaitable[Move]]

MAX_RETRIES = 2
TALK_FIELDS: tuple[TalkField, ...] = ("message", "declared_action", "declared_type")

# Game plugins. Each maps a game_id to a function or table.
ALLOWED_ACTIONS = {
    "crisis_bargaining": crisis_bargaining.ALLOWED_ACTIONS,
    "contract_dynamics": contract_dynamics.ALLOWED_ACTIONS,
}
BRIEFINGS = {
    "crisis_bargaining": crisis_bargaining.briefing,
    "contract_dynamics": contract_dynamics.briefing,
}
ENDING_RULES = {
    "crisis_bargaining": crisis_bargaining.ending_after_move,
    "contract_dynamics": contract_dynamics.ending_after_move,
}
PAYOFF_MODELS = {
    "crisis_bargaining": crisis_bargaining.episode_payoffs,
    "contract_dynamics": contract_dynamics.episode_payoffs,
}
DEFAULT_MOVES = {
    "crisis_bargaining": crisis_bargaining.default_move,
    "contract_dynamics": contract_dynamics.default_move,
}
# nature's moves before each stage; games without any are absent
BEFORE_STAGE = {"contract_dynamics": contract_dynamics.before_stage}
# game-specific rules a move must satisfy; they return a problem or None
MOVE_CHECKS = {"contract_dynamics": contract_dynamics.move_problem}
# players for roles other than proposer and responder
ACTOR_RULES = {"contract_dynamics": contract_dynamics.actor_for}

StageOutcome = Ending | Literal["next_stage"] | None


async def run_episode(
    config: EpisodeConfig, players: dict[PlayerId, Player]
) -> EpisodeState:
    game = config.game
    types = draw_types(config.priors, config.seed) if config.priors else {}
    state = EpisodeState(types=types)
    nature_rng = random.Random(f"nature-{config.seed}")
    stop_rng = random.Random(f"random_stop-{config.seed}")
    inbox: dict[PlayerId, list[Move]] = {"alpha": [], "beta": []}
    briefings = {
        player: BRIEFINGS[game.game_id](config, state, player)
        for player in ("alpha", "beta")
    }
    for stage in game.stages:
        before_stage = BEFORE_STAGE.get(game.game_id)
        if before_stage is not None:
            state.ending = before_stage(game, state, stage, nature_rng)
            if state.ending is not None:
                break
        outcome = await _play_stage(
            config, players, state, inbox, briefings, stage, stop_rng
        )
        if outcome != "next_stage":
            state.ending = outcome
            break
    state.payoffs = PAYOFF_MODELS[game.game_id](game, state)
    return state


async def _play_stage(
    config: EpisodeConfig,
    players: dict[PlayerId, Player],
    state: EpisodeState,
    inbox: dict[PlayerId, list[Move]],
    briefings: dict[PlayerId, Briefing],
    stage: StageSpec,
    stop_rng: random.Random,
) -> StageOutcome:
    max_rounds = stage.termination.max_rounds
    for round_number in range(1, max_rounds + 1):
        outcome = await _play_round(
            config, players, state, inbox, briefings, stage, round_number
        )
        if outcome is not None:
            return outcome
        more_rounds = round_number < max_rounds
        if more_rounds and stop_rng.random() < stage.termination.stop_prob:
            return "random_stop"
    return stage.when_exhausted


async def _play_round(
    config: EpisodeConfig,
    players: dict[PlayerId, Player],
    state: EpisodeState,
    inbox: dict[PlayerId, list[Move]],
    briefings: dict[PlayerId, Briefing],
    stage: StageSpec,
    round_number: int,
) -> StageOutcome:
    game = config.game
    proposer = _proposer(stage.proposer_rule, round_number)
    record = RoundRecord(
        stage=stage.name,
        round=round_number,
        proposer=proposer if _has_offers(stage) else None,
    )
    state.rounds.append(record)
    for phase in stage.phases:
        actors = _actors(game.game_id, phase, proposer, state)
        # Simultaneous movers are asked together: each observation is built
        # before any move of the phase is in, so neither sees the other's.
        moves = await asyncio.gather(
            *(
                _phase_move(config, players, state, inbox, briefings, phase, record, p)
                for p in actors
            )
        )
        for player, move in zip(actors, moves):
            record.moves[player] = move
            inbox[_opponent(player)].append(move)
        for move in moves:
            outcome = ENDING_RULES[game.game_id](move)
            if outcome is not None:
                return outcome
    return None


async def _phase_move(
    config: EpisodeConfig,
    players: dict[PlayerId, Player],
    state: EpisodeState,
    inbox: dict[PlayerId, list[Move]],
    briefings: dict[PlayerId, Briefing],
    phase: PhaseSpec,
    record: RoundRecord,
    player: PlayerId,
) -> Move:
    """One player's move in a phase: a valid move, or the game's default."""
    game = config.game
    context = PhaseContext(
        round=record.round,
        role=phase.actors,
        action_kind=phase.action_kind,
        allowed_actions=ALLOWED_ACTIONS[game.game_id][phase.action_kind],
        talk_fields=_talk_fields(config),
        declarable_fields=_private_fields(config),
    )
    observation = Observation(
        recipient=player,
        round=record.round,
        briefing=briefings.pop(player, None),
        opponent_moves=inbox[player],
        contract=state.contract.model_copy() if state.contract else None,
    )
    inbox[player] = []
    move = await _request_move(game, players[player], observation, context)
    if move is None:
        move = DEFAULT_MOVES[game.game_id](game, state, player, context)
        record.defaulted.add(player)
    return move


async def _request_move(
    game: GameSpec, player: Player, observation: Observation, context: PhaseContext
) -> Move | None:
    """Ask for a move, retrying with the reason whenever it is invalid.

    Returns None if every attempt was invalid.
    """
    for _ in range(1 + MAX_RETRIES):
        try:
            move = await player(observation, context)
            _validate(game, move, context)
            return move
        except InvalidMove as invalid:
            # A retry carries no new information, only the reason.
            observation = Observation(
                recipient=observation.recipient, round=observation.round
            )
            context = context.model_copy(update={"error": str(invalid)})
    return None


def _validate(game: GameSpec, move: Move, context: PhaseContext) -> None:
    allowed = context.allowed_actions
    if move.action is None:
        if allowed:
            raise InvalidMove(
                f"an action is required; choose one of: {_listed(allowed)}"
            )
    elif move.action.kind not in allowed:
        raise InvalidMove(
            f"'{move.action.kind}' is not allowed for the {context.role}; "
            f"choose one of: {_listed(allowed)}"
        )
    used_talk = {field for field in TALK_FIELDS if getattr(move, field) is not None}
    # In a message phase (no actions) the message is the move itself.
    open_talk = context.talk_fields | (set() if allowed else {"message"})
    closed = used_talk - open_talk
    if closed:
        raise InvalidMove(f"these fields are not available: {_listed(closed)}")
    if move.declared_type is not None:
        claimed = set(move.declared_type.model_dump(exclude_none=True))
        undeclarable = claimed - context.declarable_fields
        if undeclarable:
            raise InvalidMove(
                f"a declared type may only claim private fields, not: "
                f"{_listed(undeclarable)}"
            )
    move_check = MOVE_CHECKS.get(game.game_id)
    problem = move_check(game, move) if move_check else None
    if problem is not None:
        raise InvalidMove(problem)


def _listed(names: set[str]) -> str:
    return ", ".join(sorted(names))


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


def _has_offers(stage: StageSpec) -> bool:
    return any(phase.actors in ("proposer", "responder") for phase in stage.phases)


def _actors(
    game_id: str, phase: PhaseSpec, proposer: PlayerId, state: EpisodeState
) -> list[PlayerId]:
    if phase.actors == "both":
        return ["alpha", "beta"]
    return [_actor(game_id, phase, proposer, state)]


def _actor(
    game_id: str, phase: PhaseSpec, proposer: PlayerId, state: EpisodeState
) -> PlayerId:
    if phase.actors == "proposer":
        return proposer
    if phase.actors == "responder":
        return _opponent(proposer)
    return ACTOR_RULES[game_id](phase.actors, state)


def _proposer(
    rule: Literal["alternate", "alpha", "beta"], round_number: int
) -> PlayerId:
    if rule == "alternate":
        return "alpha" if round_number % 2 == 1 else "beta"
    return rule


def _opponent(player: PlayerId) -> PlayerId:
    return "beta" if player == "alpha" else "alpha"
