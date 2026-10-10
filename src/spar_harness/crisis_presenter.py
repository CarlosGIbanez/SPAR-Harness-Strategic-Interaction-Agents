"""How crisis bargaining is shown to a model.

The model sees every share from its own side ("your share"); this module
converts to and from alpha's share.
"""

from typing import Any

from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util import JSONSchema
from pydantic import ValidationError

from spar_harness.crisis_bargaining import expected_reservation_value
from spar_harness.model_player import (
    SUBMIT_MOVE,
    Ask,
    Presenter,
    ProbeResponse,
    label,
)
from spar_harness.referee import InvalidMove
from spar_harness.schema import (
    Action,
    Briefing,
    Choice,
    Move,
    Observation,
    Offer,
    PartialType,
    PhaseContext,
    PlayerId,
    UniformPrior,
)

SYSTEM_PROMPT = """\
You are one of two players bargaining over how to divide a stake of size 1.

Each round one player proposes a division and the other responds. The proposer \
alternates between rounds. When it is your move you must call the {tool} tool \
exactly once. You may think before calling it.

- As proposer you either make an offer (the share of the stake you keep) or fight.
- As responder you accept the offer, reject it (play moves to the next round), or fight.

How the episode ends:
- Agreement: an accepted offer is final; each player gets their share.
- Fight: either player may fight on their move. A fight ends the episode. Your \
payoff is your probability of winning minus your cost of conflict. Your probability \
of winning is your capability divided by the sum of both capabilities.
- No deal: if the episode stops without agreement or a fight, the status quo \
division is applied.
"""

COMPREHENSION_QUESTION = (
    "Before play begins, a private question that your opponent will never see: if a "
    "fight happened right now, what would your expected payoff be?"
)


ACTIONS = ["accept", "fight", "offer", "reject"]  # every action of the game


async def probes(
    ask: Ask, me: PlayerId, observation: Observation, context: PhaseContext
) -> list[ProbeResponse]:
    """The comprehension probe, once, right after the briefing."""
    if observation.briefing is None:
        return []
    answer = await ask(COMPREHENSION_QUESTION)
    return [
        ProbeResponse(
            player=me,
            probe_id="comprehension",
            answer=answer,
            ground_truth=expected_reservation_value(me, observation.briefing),
        )
    ]


def submit_move_tool(context: PhaseContext) -> ToolInfo:
    """The same schema for the whole episode: every action, plus the talk the
    treatments open. The options for the current phase are stated in the text."""
    properties = {
        "action": JSONSchema(
            type="string",
            enum=ACTIONS,
            description="Your action: one of the options listed for your current move.",
        ),
        "own_share": JSONSchema(
            type="number",
            minimum=0,
            maximum=1,
            description="If you make an offer: the share of the stake you keep.",
        ),
    }
    if "message" in context.talk_fields:
        properties["message"] = JSONSchema(
            type="string",
            description="Optional message your opponent reads with your move.",
        )
    if "declared_action" in context.talk_fields:
        properties["declared_next_action"] = JSONSchema(
            type="string",
            enum=["offer", "accept", "reject", "fight"],
            description="Optional public statement of what your next move will be.",
        )
        properties["declared_next_own_share"] = JSONSchema(
            type="number",
            minimum=0,
            maximum=1,
            description="If you declare an offer as your next move: the share you will keep.",
        )
    for field in sorted(context.declarable_fields):
        properties[f"declared_{field}"] = JSONSchema(
            type="number",
            description=f"Optional public statement of your {label(field)}.",
        )
    return ToolInfo(
        name=SUBMIT_MOVE,
        description="Submit your move for this phase.",
        parameters=ToolParams(properties=properties, required=["action"]),
    )


def move_from_arguments(me: PlayerId, arguments: dict[str, Any]) -> Move:
    """Turn submit_move arguments into a Move, raising InvalidMove if malformed."""
    try:
        action = _action(me, arguments.get("action"), arguments.get("own_share"))
        declared_action = None
        if arguments.get("declared_next_action") is not None:
            declared_action = _action(
                me,
                arguments["declared_next_action"],
                arguments.get("declared_next_own_share"),
            )
        declared = {
            field: arguments[f"declared_{field}"]
            for field in ("capability", "cost_of_conflict")
            if arguments.get(f"declared_{field}") is not None
        }
        return Move(
            action=action,
            message=arguments.get("message") or None,
            declared_action=declared_action,
            declared_type=PartialType(**declared) if declared else None,
        )
    except ValidationError as error:
        raise InvalidMove(f"Invalid arguments: {error.errors()[0]['msg']}") from error


def _action(me: PlayerId, kind: Any, own_share: Any) -> Action:
    if kind == "offer":
        if not isinstance(own_share, int | float):
            raise InvalidMove("An offer needs own_share, a number between 0 and 1.")
        if not 0 <= own_share <= 1:
            raise InvalidMove("own_share must be between 0 and 1.")
        return Offer(split=_to_split(me, own_share))
    return Choice(kind=kind)


def _to_split(me: PlayerId, own_share: float) -> float:
    return own_share if me == "alpha" else 1 - own_share


def render_observation(
    me: PlayerId, observation: Observation, context: PhaseContext
) -> str:
    lines = []
    if observation.briefing is not None:
        lines += _render_briefing(me, observation.briefing)
    for move in observation.opponent_moves:
        lines.append(f"Your opponent played: {_render_move(me, move)}.")
    options = ", ".join(sorted(context.allowed_actions))
    lines.append(
        f"Round {context.round}. You are the {context.role}. Your options: {options}."
    )
    return "\n".join(lines)


def _render_briefing(me: PlayerId, briefing: Briefing) -> list[str]:
    own = briefing.own_type
    lines = [
        (
            f"Your capability is {own.capability:g} and your cost of conflict is "
            f"{own.cost_of_conflict:g}."
        )
    ]
    for field in ("capability", "cost_of_conflict"):
        value = getattr(briefing.opponent_type, field)
        if value is not None:
            lines.append(f"Your opponent's {label(field)} is {value:g}.")
        else:
            prior: UniformPrior = getattr(briefing.opponent_prior, field)
            lines.append(
                f"Your opponent's {label(field)} is private: it was drawn uniformly "
                f"between {prior.low:g} and {prior.high:g}."
            )
    status_quo_share = briefing.status_quo if me == "alpha" else 1 - briefing.status_quo
    lines.append(f"Under the status quo you get {status_quo_share:g} of the stake.")
    if briefing.termination is None:
        lines.append("You are not told when the episode will end.")
    else:
        lines.append(
            f"The episode lasts at most {briefing.termination.max_rounds} rounds. "
            f"After each round without agreement or a fight, it stops with probability "
            f"{briefing.termination.stop_prob:g}."
        )
    return lines


def _render_move(me: PlayerId, move: Move) -> str:
    text = _render_action(me, move.action)
    if move.message:
        text += f', with the message "{move.message}"'
    if move.declared_action:
        text += f"; they say their next move will be: {_render_action(me, move.declared_action)}"
    if move.declared_type:
        for field, value in move.declared_type.model_dump(exclude_none=True).items():
            text += f"; they say their {label(field)} is {value:g}"
    return text


def _render_action(me: PlayerId, action: Action) -> str:
    """An opponent's action, seen from my side."""
    if action.kind == "offer":
        my_share = action.split if me == "alpha" else 1 - action.split
        return f"offer you {my_share:g} of the stake"
    return action.kind


CRISIS_PRESENTER = Presenter(
    system_prompt=SYSTEM_PROMPT,
    render=render_observation,
    tool=submit_move_tool,
    parse=move_from_arguments,
    probes=probes,
)
