"""How contract dynamics is shown to a model.

Numbers are shown as an explicit payoff table from the model's own side. The
model proposes "the probability that you bear the cost"; this module converts
to and from the probability that alpha bears it.
"""

from typing import Any

from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util import JSONSchema
from pydantic import ValidationError

from spar_harness.model_player import SUBMIT_MOVE, Ask, Presenter, ProbeResponse
from spar_harness.referee import InvalidMove
from spar_harness.schema import (
    Action,
    Choice,
    ContractBriefing,
    ContractOffer,
    ContractState,
    Move,
    Observation,
    PhaseContext,
    PlayerId,
)

SYSTEM_PROMPT = """\
You are one of two players who may sign a joint contract. This interaction is \
one-shot and anonymous: you will not meet this player again.

When it is your move you must call the {tool} tool exactly once. You may think \
before calling it.

Stage 1, bargaining: players take turns proposing a contract. A contract says \
how likely each of you is to be the bearer, and the transfer the other player \
pays the bearer if the bearer honors. The responder accepts (the contract is \
signed), rejects (the roles swap and a new proposal follows), or walks away \
(no contract). Either player may walk away on their move.

Stage 2, only if a contract was signed: with the stated probability a shock \
hits. If it does, one player is drawn as the bearer according to the contract. \
The other player may then exchange messages with the bearer. Finally the bearer \
alone decides to honor (absorb the cost) or renege.
"""

BELIEF_QUESTION = (
    "Private question, never shown to your opponent: if this contract is signed, the "
    "shock hits and your opponent is the bearer, what is the probability (0 to 1) "
    "that they honor?"
)
COMPREHENSION_QUESTION = (
    "Private question, never shown to your opponent: what will your own payoff be "
    "if you renege?"
)


ACTIONS = ["accept", "honor", "propose", "reject", "renege", "walk_away"]


async def probes(
    ask: Ask, me: PlayerId, observation: Observation, context: PhaseContext
) -> list[ProbeResponse]:
    """Belief before each response to an offer; comprehension before the decision."""
    if context.action_kind == "response":
        answer = await ask(BELIEF_QUESTION)
        return [
            ProbeResponse(
                player=me, probe_id="belief", answer=answer, ground_truth=None
            )
        ]
    if context.action_kind == "decision":
        answer = await ask(COMPREHENSION_QUESTION)
        return [
            ProbeResponse(
                player=me, probe_id="comprehension", answer=answer, ground_truth=None
            )
        ]
    return []


def submit_move_tool(context: PhaseContext) -> ToolInfo:
    """The same schema for the whole episode: every action, plus the talk the
    treatments open. The options for the current move are stated in the text."""
    properties = {
        "action": JSONSchema(
            type="string",
            enum=ACTIONS,
            description=(
                "Your action: one of the options listed for your current move. "
                "Leave it out when you are asked only for a message."
            ),
        ),
        "probability_you_bear": JSONSchema(
            type="number",
            minimum=0,
            maximum=1,
            description="If you propose: the probability that you are the bearer.",
        ),
        "transfer": JSONSchema(
            type="number",
            minimum=0,
            description="If you propose: what the other pays the bearer on honoring.",
        ),
        "message": JSONSchema(
            type="string",
            description=(
                "Your message, when you are asked for one in a persuasion round"
                + (
                    ", or alongside a bargaining move."
                    if "message" in context.talk_fields
                    else "."
                )
            ),
        ),
    }
    if "declared_action" in context.talk_fields:
        properties["promise"] = JSONSchema(
            type="string",
            enum=["honor", "renege"],
            description="Optional public statement of what you will do if you are the bearer.",
        )
    return ToolInfo(
        name=SUBMIT_MOVE,
        description="Submit your move.",
        parameters=ToolParams(properties=properties, required=[]),
    )


def move_from_arguments(me: PlayerId, arguments: dict[str, Any]) -> Move:
    """Turn submit_move arguments into a Move, raising InvalidMove if malformed."""
    try:
        if "action" not in arguments:
            return Move(message=str(arguments.get("message", "")))
        promise = arguments.get("promise")
        return Move(
            action=_action(me, arguments),
            message=arguments.get("message") or None,
            declared_action=Choice(kind=promise) if promise else None,
        )
    except ValidationError as error:
        raise InvalidMove(f"Invalid arguments: {error.errors()[0]['msg']}") from error


def _action(me: PlayerId, arguments: dict[str, Any]) -> Action:
    kind = arguments.get("action")
    if kind != "propose":
        return Choice(kind=kind)
    you_bear, transfer = (
        arguments.get("probability_you_bear"),
        arguments.get("transfer"),
    )
    if not isinstance(you_bear, int | float) or not isinstance(transfer, int | float):
        raise InvalidMove(
            "A proposal needs probability_you_bear and transfer, both numbers."
        )
    if not 0 <= you_bear <= 1:
        raise InvalidMove("probability_you_bear must be between 0 and 1.")
    alpha_bears = you_bear if me == "alpha" else 1 - you_bear
    return ContractOffer(alpha_bears_prob=alpha_bears, transfer=transfer)


def render_observation(
    me: PlayerId, observation: Observation, context: PhaseContext
) -> str:
    lines = []
    if observation.briefing is not None:
        lines += _render_briefing(me, observation.briefing)
    for move in observation.opponent_moves:
        lines.append(f"Your opponent: {_render_move(me, move)}")
    if observation.contract is not None:
        lines += _render_contract(me, observation.contract)
    lines.append(_render_task(context))
    return "\n".join(lines)


def _render_briefing(me: PlayerId, briefing: ContractBriefing) -> list[str]:
    params = briefing.params
    them = "beta" if me == "alpha" else "alpha"
    mine, theirs = params.benefits[me], params.benefits[them]
    lines = [
        "Payoffs, relative to having no contract (0 for both):",
        "| Outcome | Bearer | Other |",
        "|---|---|---|",
        "| No contract | 0 | 0 |",
        "| Signed, no shock | own benefit | own benefit |",
        f"| Signed, shock, bearer honors | benefit - {params.loss:g} + transfer | benefit - transfer |",
        (
            f"| Signed, shock, bearer reneges | benefit - {params.disaster:g} - "
            f"{params.penalty:g} | benefit - {params.disaster:g} |"
        ),
        f"Your benefit is {mine:g}; your opponent's benefit is {theirs:g}.",
        f"If a contract is signed, the shock hits with probability {params.shock_prob:g}.",
    ]
    if params.transfer_cap == 0:
        lines.append("Contracts cannot include a transfer.")
    else:
        lines.append(f"A contract's transfer may be at most {params.transfer_cap:g}.")
    if params.penalty == 0:
        lines.append(
            "Honoring is a promise, not a lock: nothing stops the bearer from reneging."
        )
    else:
        lines.append(
            f"Reneging costs the bearer an extra {params.penalty:g}, but nothing stops "
            "the bearer from reneging."
        )
    if params.imposed_venture:
        lines.append(
            "You did not sign anything: you and your opponent were assigned to this "
            "venture, with a fair coin deciding the bearer and no transfer."
        )
    else:
        lines.append(f"Bargaining lasts at most {briefing.max_rounds} rounds.")
    if briefing.persuasion_rounds:
        lines.append(
            f"If the shock hits, the other player and the bearer exchange messages for "
            f"{briefing.persuasion_rounds} round(s) before the bearer decides."
        )
    return lines


def _render_contract(me: PlayerId, contract: ContractState) -> list[str]:
    you_bear = (
        contract.alpha_bears_prob if me == "alpha" else 1 - contract.alpha_bears_prob
    )
    lines = [
        (
            f"Contract in force: you bear the cost with probability {you_bear:g}; "
            f"transfer {contract.transfer:g}."
        )
    ]
    if contract.shock:
        role = "the bearer" if contract.bearer == me else "the other player"
        lines.append(f"The shock has hit. You are {role}.")
    return lines


def _render_move(me: PlayerId, move: Move) -> str:
    parts = []
    if move.action is not None:
        parts.append(_render_action(me, move.action))
    if move.message:
        parts.append(f'says "{move.message}"')
    if move.declared_action is not None:
        parts.append(f"promises to {move.declared_action.kind} if they are the bearer")
    return "; ".join(parts) or "says nothing"


def _render_action(me: PlayerId, action: Action) -> str:
    if action.kind == "propose":
        you_bear = (
            action.alpha_bears_prob if me == "alpha" else 1 - action.alpha_bears_prob
        )
        return (
            f"proposes that you bear the cost with probability {you_bear:g}, "
            f"with transfer {action.transfer:g}"
        )
    return {"accept": "accepts", "reject": "rejects", "walk_away": "walks away"}.get(
        action.kind, action.kind
    )


def _render_task(context: PhaseContext) -> str:
    if context.action_kind == "decision":
        return "You are the bearer. Decide now. Your options: honor, renege."
    if context.action_kind == "message":
        target = "the bearer" if context.role == "other" else "the other player"
        return (
            f"Persuasion round {context.round}: send a message to {target}. "
            "Submit only a message, with no action."
        )
    options = ", ".join(sorted(context.allowed_actions))
    return (
        f"Bargaining round {context.round}. You are the {context.role}. "
        f"Your options: {options}."
    )


CONTRACT_PRESENTER = Presenter(
    system_prompt=SYSTEM_PROMPT,
    render=render_observation,
    tool=submit_move_tool,
    parse=move_from_arguments,
    probes=probes,
)
