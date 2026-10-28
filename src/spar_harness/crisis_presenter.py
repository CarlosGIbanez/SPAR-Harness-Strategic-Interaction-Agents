"""How crisis bargaining is shown to a model.

The model sees every offer from its own side ("your share", "you take it and
pay X", "your chance is X"); this module converts to and from alpha's share.
Each version of the offer has its own system prompt and offer fields, so there
is one Presenter per version.
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
    Version,
)

SPLIT_PROMPT = """\
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
- No deal: if the episode stops without agreement or a fight, each player gets \
the no-deal payoff stated at the start.
"""

GOOD_PROMPT = """\
You are one of two players, Player A and Player B, bargaining over one good \
that cannot be split. Payoffs are in points: the good is worth 1 point to \
whoever ends up with it.

Each round one player makes an offer and the other responds. The proposer \
alternates between rounds. When it is your move you must call the {tool} tool \
exactly once. You may think before calling it.

- As proposer you either make an offer or fight.
- As responder you accept the offer, reject it (play moves to the next round), \
or fight.

{offer_rule}

How the episode ends:
- Agreement: an accepted offer is final and is carried out.
- Fight: either player may fight on their move. A fight ends the episode. Your \
payoff is your probability of winning minus your cost of conflict. Your probability \
of winning is your capability divided by the sum of both capabilities.
- No deal: if the episode stops without agreement or a fight, each player gets \
the no-deal payoff stated at the start.
"""

OFFER_RULES = {
    "strict": (
        "An offer says who takes the good. If it is accepted, the player who takes "
        "the good gets 1 point and the other gets 0."
    ),
    "lottery": (
        "An offer says each player's chance of getting the good. If it is accepted, "
        "each player's payoff is their chance (the good is not actually drawn): a "
        "chance of 0.3 pays 0.3 points."
    ),
    "payment": (
        "An offer says who takes the good and a payment, between 0 and 1 point, "
        "that the taker makes to the other player. If it is accepted, the taker "
        "gets 1 minus the payment and the other player gets the payment."
    ),
}

COMPREHENSION_QUESTION = (
    "Before play begins, a private question that your opponent will never see: if a "
    "fight happened right now, what would your expected payoff be?"
)

NAMES = {"alpha": "Player A", "beta": "Player B"}
ACTIONS = ["accept", "fight", "offer", "reject"]  # every action of the game


def crisis_presenter(version: Version) -> Presenter:
    if version == "split":
        system_prompt = SPLIT_PROMPT
    else:
        system_prompt = GOOD_PROMPT.replace("{offer_rule}", OFFER_RULES[version])
    return Presenter(
        system_prompt=system_prompt,
        render=lambda me, observation, context: render_observation(
            version, me, observation, context
        ),
        tool=lambda observation, context: submit_move_tool(
            version, observation, context
        ),
        parse=lambda me, arguments: move_from_arguments(version, me, arguments),
        probes=probes,
    )


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


def submit_move_tool(
    version: Version, observation: Observation, context: PhaseContext
) -> ToolInfo:
    """The same schema for the whole episode: every action, the version's offer
    fields, plus the talk the treatments open. The options for the current
    phase are stated in the text."""
    properties = {
        "action": JSONSchema(
            type="string",
            enum=ACTIONS,
            description="Your action: one of the options listed for your current move.",
        ),
        **_offer_fields(version, "If you make an offer: "),
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
        properties |= _offer_fields(
            version, "If you declare an offer as your next move: ", "declared_next_"
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


def _offer_fields(version: Version, lead: str, prefix: str = "") -> dict[str, Any]:
    """The offer fields of the version, each description starting with `lead`."""
    if version == "split":
        return {
            f"{prefix}own_share": JSONSchema(
                type="number",
                minimum=0,
                maximum=1,
                description=f"{lead}the share of the stake you keep.",
            )
        }
    if version == "lottery":
        return {
            f"{prefix}your_chance": JSONSchema(
                type="number",
                minimum=0,
                maximum=1,
                description=f"{lead}your chance of getting the good.",
            )
        }
    fields = {
        f"{prefix}you_take": JSONSchema(
            type="boolean",
            description=f"{lead}true if you take the good, false if your opponent does.",
        )
    }
    if version == "payment":
        fields[f"{prefix}payment"] = JSONSchema(
            type="number",
            minimum=0,
            maximum=1,
            description=f"{lead}what the taker pays the other player.",
        )
    return fields


def move_from_arguments(
    version: Version, me: PlayerId, arguments: dict[str, Any]
) -> Move:
    """Turn submit_move arguments into a Move, raising InvalidMove if malformed."""
    try:
        action = _action(version, me, arguments.get("action"), arguments)
        declared_action = None
        if arguments.get("declared_next_action") is not None:
            declared_action = _action(
                version,
                me,
                arguments["declared_next_action"],
                arguments,
                prefix="declared_next_",
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


def _action(
    version: Version,
    me: PlayerId,
    kind: Any,
    arguments: dict[str, Any],
    prefix: str = "",
) -> Action:
    if kind != "offer":
        return Choice(kind=kind)
    own_share = _own_share(version, arguments, prefix)
    return Offer(split=own_share if me == "alpha" else 1 - own_share)


def _own_share(version: Version, arguments: dict[str, Any], prefix: str) -> float:
    """The player's own share of the value in the offer it describes."""
    if version in ("split", "lottery"):
        field = f"{prefix}own_share" if version == "split" else f"{prefix}your_chance"
        return _unit_number(arguments, field)
    you_take = arguments.get(f"{prefix}you_take")
    if not isinstance(you_take, bool):
        raise InvalidMove(f"An offer needs {prefix}you_take, true or false.")
    if version == "strict":
        return 1.0 if you_take else 0.0
    payment = _unit_number(arguments, f"{prefix}payment")
    return 1 - payment if you_take else payment


def _unit_number(arguments: dict[str, Any], field: str) -> float:
    value = arguments.get(field)
    if not isinstance(value, int | float):
        raise InvalidMove(f"An offer needs {field}, a number between 0 and 1.")
    if not 0 <= value <= 1:
        raise InvalidMove(f"{field} must be between 0 and 1.")
    return float(value)


def render_observation(
    version: Version, me: PlayerId, observation: Observation, context: PhaseContext
) -> str:
    lines = []
    if observation.briefing is not None:
        lines += _render_briefing(version, me, observation.briefing)
    for move in observation.opponent_moves:
        lines.append(f"Your opponent played: {_render_move(version, me, move)}.")
    options = ", ".join(sorted(context.allowed_actions))
    lines.append(
        f"Round {context.round}. You are the {context.role}. Your options: {options}."
    )
    return "\n".join(lines)


def _render_briefing(version: Version, me: PlayerId, briefing: Briefing) -> list[str]:
    own = briefing.own_type
    lines = []
    if version != "split":
        lines.append(f"You are {NAMES[me]}; your opponent is {NAMES[_other(me)]}.")
    lines.append(
        f"Your capability is {own.capability:g} and your cost of conflict is "
        f"{own.cost_of_conflict:g}."
    )
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
    if briefing.no_deal == "nothing":
        lines.append("If the episode ends without a deal or a fight, you get nothing.")
    else:
        own_status_quo = (
            briefing.status_quo if me == "alpha" else 1 - briefing.status_quo
        )
        unit = "of the stake" if version == "split" else "points"
        lines.append(
            f"If the episode ends without a deal or a fight, you get "
            f"{own_status_quo:g} {unit} (the status quo)."
        )
    if briefing.termination is None:
        lines.append("You are not told when the episode will end.")
    else:
        lines.append(
            f"The episode lasts at most {briefing.termination.max_rounds} rounds. "
            f"After each round without agreement or a fight, it stops with probability "
            f"{briefing.termination.stop_prob:g}."
        )
    return lines


def _render_move(version: Version, me: PlayerId, move: Move) -> str:
    text = _render_action(version, me, move.action)
    if move.message:
        text += f', with the message "{move.message}"'
    if move.declared_action:
        text += (
            f"; they say their next move will be: "
            f"{_render_action(version, me, move.declared_action)}"
        )
    if move.declared_type:
        for field, value in move.declared_type.model_dump(exclude_none=True).items():
            text += f"; they say their {label(field)} is {value:g}"
    return text


def _render_action(version: Version, me: PlayerId, action: Action) -> str:
    """An opponent's action, seen from my side. In the payment version the
    taker is the player with the larger share; at exactly a half, the opponent
    (who made the offer) takes."""
    if action.kind != "offer":
        return action.kind
    my_share = action.split if me == "alpha" else 1 - action.split
    if version == "split":
        return f"offer you {my_share:g} of the stake"
    if version == "lottery":
        return f"offer you a chance of {my_share:g} of getting the good"
    if version == "strict":
        return "offer that you take the good" if my_share else "claim the good"
    if my_share > 0.5:
        return f"offer that you take the good and pay them {1 - my_share:g}"
    return f"offer that they take the good and pay you {my_share:g}"


def _other(player: PlayerId) -> PlayerId:
    return "beta" if player == "alpha" else "alpha"


CRISIS_PRESENTERS: dict[Version, Presenter] = {
    version: crisis_presenter(version)
    for version in ("split", "strict", "lottery", "payment")
}
CRISIS_PRESENTER = CRISIS_PRESENTERS["split"]
