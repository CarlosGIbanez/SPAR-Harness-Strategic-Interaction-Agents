"""How contract dynamics is shown to a model.

Numbers are shown as an explicit payoff table from the model's own side. The
model proposes "the probability that you bear the cost"; this module converts
to and from the probability that alpha bears it.

The story, the unit and the two decision labels come from the episode's
setting (contract_framing.py). The model submits the setting's labels, which
are mapped back to the internal actions "honor" and "renege". The words honor
and renege never appear in model-facing text.
"""

from typing import Any

from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util import JSONSchema
from pydantic import ValidationError

from spar_harness.contract_framing import PROMPT_ARMS, SETTINGS, Setting
from spar_harness.model_player import SUBMIT_MOVE, Ask, Presenter, ProbeResponse
from spar_harness.referee import InvalidMove
from spar_harness.schema import (
    Action,
    Choice,
    ContractBriefing,
    ContractOffer,
    ContractParams,
    ContractState,
    Move,
    Observation,
    PhaseContext,
    PlayerId,
    PresentationTreatment,
)

BARGAINING_ACTIONS = ["accept", "propose", "reject", "walk_away"]


def contract_presenter(
    params: ContractParams, presentation: PresentationTreatment | None = None
) -> Presenter:
    """Build the presenter for one episode's setting and prompt arm."""
    presentation = presentation or PresentationTreatment()
    setting = SETTINGS[presentation.setting]
    to_internal = {setting.bear: "honor", setting.decline: "renege"}
    to_label = {"honor": setting.bear, "renege": setting.decline}

    async def probes(
        ask: Ask, me: PlayerId, observation: Observation, context: PhaseContext
    ) -> list[ProbeResponse]:
        """Belief before each response to an offer and before each persuasion
        message; comprehension before the decision."""
        if context.action_kind == "response":
            question = (
                "Private question, never shown to the other side: if this contract "
                f"is signed, {setting.shock} occurs and the other side is the "
                f"bearer, what is the probability (0 to 1) that they choose "
                f"{setting.bear}?"
            )
            probe_id = "belief"
        elif context.action_kind == "message" and context.role == "other":
            question = (
                "Private question, never shown to the bearer: what is the "
                f"probability (0 to 1) that the bearer will choose {setting.bear}?"
            )
            probe_id = "persuader_belief"
        elif context.action_kind == "decision":
            question = (
                "Private question, never shown to the other side: what will your "
                f"own payoff be if you choose {setting.decline}?"
            )
            probe_id = "comprehension"
        else:
            return []
        answer = await ask(question)
        return [
            ProbeResponse(
                player=me, probe_id=probe_id, answer=answer, ground_truth=None
            )
        ]

    def tool(observation: Observation, context: PhaseContext) -> ToolInfo:
        """The same schema for the whole episode: every action, plus the transfer
        and talk the treatments allow. The current options are in the text."""
        cap = params.transfer_cap
        properties = {
            "action": JSONSchema(
                type="string",
                enum=sorted([*BARGAINING_ACTIONS, setting.bear, setting.decline]),
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
            "message": JSONSchema(
                type="string",
                description=(
                    "Your message, when you are asked for one in a message round"
                    + (
                        ", or alongside a bargaining move."
                        if "message" in context.talk_fields
                        else "."
                    )
                ),
            ),
        }
        if cap > 0:
            properties["transfer"] = JSONSchema(
                type="number",
                minimum=0,
                maximum=cap,
                description=(
                    "If you propose: what the other side pays the bearer if the "
                    f"bearer chooses {setting.bear}."
                ),
            )
        if "declared_action" in context.talk_fields:
            properties["promise"] = JSONSchema(
                type="string",
                enum=[setting.bear, setting.decline],
                description=(
                    "Optional public statement of what you will choose if you are "
                    "the bearer."
                ),
            )
        return ToolInfo(
            name=SUBMIT_MOVE,
            description="Submit your move.",
            parameters=ToolParams(properties=properties, required=[]),
        )

    def parse(me: PlayerId, arguments: dict[str, Any]) -> Move:
        """Turn submit_move arguments into a Move, raising InvalidMove if malformed."""
        try:
            if "action" not in arguments:
                return Move(message=str(arguments.get("message", "")))
            promise = arguments.get("promise")
            if promise is not None and promise not in to_internal:
                raise InvalidMove(
                    f"promise must be {setting.bear} or {setting.decline}."
                )
            return Move(
                action=_action(me, arguments, to_internal),
                message=arguments.get("message") or None,
                declared_action=Choice(kind=to_internal[promise]) if promise else None,
            )
        except ValidationError as error:
            raise InvalidMove(
                f"Invalid arguments: {error.errors()[0]['msg']}"
            ) from error

    def render(me: PlayerId, observation: Observation, context: PhaseContext) -> str:
        lines = []
        if observation.briefing is not None:
            lines += _render_briefing(me, observation.briefing, setting)
        for move in observation.opponent_moves:
            lines.append(f"The other side: {_render_move(me, move, to_label)}")
        if observation.contract is not None:
            lines += _render_contract(me, observation.contract, setting, params)
        lines.append(_render_task(context, setting))
        return "\n".join(lines)

    return Presenter(
        system_prompt=_system_prompt(setting, presentation, params),
        render=render,
        tool=tool,
        parse=parse,
        probes=probes,
    )


def _system_prompt(
    setting: Setting, presentation: PresentationTreatment, params: ContractParams
) -> str:
    # Only {tool} is left for model_player to fill, so no other braces appear.
    intro = f"{setting.intro} {PROMPT_ARMS[presentation.prompt_arm]}"
    rules = "When it is your move you must call the {tool} tool exactly once. You may think before calling it."
    if params.imposed_venture:
        stages = (
            "Nobody signs anything here: you and the other side have been assigned "
            f"to this venture. {setting.shock[0].upper()}{setting.shock[1:]} will "
            "occur. One side is drawn as the bearer by a fair coin. The other side "
            "may then exchange messages with the bearer. Finally the bearer alone "
            f"chooses {setting.bear} (the bearer {setting.burden}) or "
            f"{setting.decline} ({setting.decline_effect} the amounts in the "
            "payoff table)."
        )
    else:
        stages = (
            "Stage 1, bargaining: you and the other side take turns proposing a "
            "contract. A contract says how likely each of you is to be the bearer "
            f"if {setting.shock} occurs and, if the rules allow it, a payment the "
            "other side makes to the bearer if the bearer chooses "
            f"{setting.bear}. The responder accepts (the contract is signed), "
            "rejects (the roles swap and a new proposal follows), or walks away "
            "(no contract). Either side may walk away on its move.\n\n"
            f"Stage 2, only if a contract was signed: with the stated probability "
            f"{setting.shock} occurs. If it does, one side is drawn as the bearer "
            "according to the contract. The other side may then exchange messages "
            "with the bearer. Finally the bearer alone chooses "
            f"{setting.bear} (the bearer {setting.burden}) or {setting.decline} "
            f"({setting.decline_effect} the amounts in the payoff table)."
        )
    return f"{intro}\n\n{rules}\n\n{stages}\n"


def _action(
    me: PlayerId, arguments: dict[str, Any], to_internal: dict[str, str]
) -> Action:
    kind = arguments.get("action")
    if kind in to_internal:
        return Choice(kind=to_internal[kind])
    if kind in ("honor", "renege"):  # internal names are not valid labels
        raise InvalidMove(f"Unknown action {kind!r}.")
    if kind != "propose":
        return Choice(kind=kind)
    you_bear = arguments.get("probability_you_bear")
    transfer = arguments.get("transfer", 0)  # absent when transfers are not allowed
    if not isinstance(you_bear, int | float) or not isinstance(transfer, int | float):
        raise InvalidMove("A proposal needs probability_you_bear as a number.")
    if not 0 <= you_bear <= 1:
        raise InvalidMove("probability_you_bear must be between 0 and 1.")
    alpha_bears = you_bear if me == "alpha" else 1 - you_bear
    return ContractOffer(alpha_bears_prob=alpha_bears, transfer=transfer)


def _render_briefing(
    me: PlayerId, briefing: ContractBriefing, setting: Setting
) -> list[str]:
    params = briefing.params
    them = "beta" if me == "alpha" else "alpha"
    mine, theirs = params.benefits[me], params.benefits[them]
    entered = "Assigned" if params.imposed_venture else "Signed"
    shocked = f"{entered}, {setting.shock} occurs"
    lines = [
        f"Payoffs in {setting.unit}, relative to having no contract (0 for both):",
        "| Outcome | Bearer | Other side |",
        "|---|---|---|",
    ]
    if not params.imposed_venture:
        lines += [
            "| No contract | 0 | 0 |",
            "| Signed, nothing occurs | own benefit | own benefit |",
        ]
    lines += [
        (
            f"| {shocked}, bearer chooses {setting.bear} | benefit - {params.loss:g} "
            "+ transfer | benefit - transfer |"
            if params.transfer_cap > 0
            else f"| {shocked}, bearer chooses {setting.bear} | benefit - "
            f"{params.loss:g} | benefit |"
        ),
        (
            f"| {shocked}, bearer chooses {setting.decline} | benefit - "
            f"{params.disaster:g} - {params.penalty:g} | benefit - "
            f"{params.disaster:g} |"
        ),
        f"Your benefit is {mine:g}; the other side's benefit is {theirs:g}.",
    ]
    if params.imposed_venture:
        lines.append(
            "You did not sign anything: you and the other side were assigned to "
            f"this venture. {setting.shock[0].upper()}{setting.shock[1:]} will "
            "occur. A fair coin decides the bearer, and there is no transfer."
        )
    else:
        lines.append(
            f"If a contract is signed, {setting.shock} occurs with probability "
            f"{params.shock_prob:g}."
        )
        if params.transfer_cap == 0:
            lines.append("Contracts cannot include a transfer.")
        else:
            lines.append(
                f"A contract's transfer may be at most {params.transfer_cap:g}."
            )
    if params.penalty == 0:
        lines.append(
            "The contract is a promise, not a lock: nothing stops the bearer from "
            f"choosing {setting.decline}."
        )
    else:
        lines.append(
            f"Choosing {setting.decline} costs the bearer an extra {params.penalty:g}, "
            f"but nothing stops the bearer from choosing it."
        )
    if not params.imposed_venture:
        lines.append(f"Bargaining lasts at most {briefing.max_rounds} rounds.")
    if briefing.persuasion_rounds:
        lines.append(
            f"If {setting.shock} occurs, the other side and the bearer exchange "
            f"messages for {briefing.persuasion_rounds} round(s) before the bearer "
            "decides. Nothing said in those messages changes the contract or is "
            "binding."
        )
    return lines


def _render_contract(
    me: PlayerId, contract: ContractState, setting: Setting, params: ContractParams
) -> list[str]:
    lines = []
    if not params.imposed_venture:
        you_bear = (
            contract.alpha_bears_prob
            if me == "alpha"
            else 1 - contract.alpha_bears_prob
        )
        lines.append(
            f"Contract in force: you are the bearer with probability {you_bear:g}; "
            f"transfer {contract.transfer:g}."
        )
    if contract.shock:
        role = "the bearer" if contract.bearer == me else "the other side"
        lines.append(f"{setting.event} You are {role}.")
    return lines


def _render_move(me: PlayerId, move: Move, to_label: dict[str, str]) -> str:
    parts = []
    if move.action is not None:
        parts.append(_render_action(me, move.action))
    if move.message:
        parts.append(f'says "{move.message}"')
    if move.declared_action is not None:
        label = to_label.get(move.declared_action.kind, move.declared_action.kind)
        parts.append(f"promises to choose {label} if they are the bearer")
    return "; ".join(parts) or "says nothing"


def _render_action(me: PlayerId, action: Action) -> str:
    if action.kind == "propose":
        you_bear = (
            action.alpha_bears_prob if me == "alpha" else 1 - action.alpha_bears_prob
        )
        return (
            f"proposes that you are the bearer with probability {you_bear:g}, "
            f"with transfer {action.transfer:g}"
        )
    return {"accept": "accepts", "reject": "rejects", "walk_away": "walks away"}.get(
        action.kind, action.kind
    )


def _render_task(context: PhaseContext, setting: Setting) -> str:
    if context.action_kind == "decision":
        return (
            "You are the bearer. Decide now. Your options: "
            f"{setting.bear}, {setting.decline}."
        )
    if context.action_kind == "message":
        target = "the bearer" if context.role == "other" else "the other side"
        return (
            f"Message round {context.round}: send a message to {target}. "
            "Submit only a message, with no action."
        )
    options = ", ".join(sorted(context.allowed_actions))
    return (
        f"Bargaining round {context.round}. You are the {context.role}. "
        f"Your options: {options}."
    )
