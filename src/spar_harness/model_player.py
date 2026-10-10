"""Adapter that lets an Inspect model act as a Player.

The model keeps one persistent conversation and moves by calling the
submit_move tool. Everything game-specific (prompt, rendering, tool schema,
parsing, probes) comes from the game's Presenter.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Model,
)
from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util import JSONSchema
from pydantic import BaseModel

from spar_harness.referee import InvalidMove, Player
from spar_harness.schema import ContractState, Move, Observation, PhaseContext, PlayerId

SUBMIT_MOVE = "submit_move"
ANSWER_PROBE = "answer_probe"


class ProbeResponse(BaseModel):
    player: PlayerId
    probe_id: str
    answer: float | None  # None if the model gave no usable answer
    ground_truth: float | None  # None if only known after the episode


Probes = Callable[
    [Model, list[ChatMessage], PlayerId, Observation, PhaseContext],
    Awaitable[list[ProbeResponse]],
]


@dataclass(frozen=True)
class Presenter:
    """How one game is shown to a model and how its answers are read."""

    system_prompt: str  # may contain {tool}
    render: Callable[[PlayerId, Observation, PhaseContext], str]
    tool: Callable[[PhaseContext], ToolInfo]  # the submit_move schema for a phase
    parse: Callable[[PlayerId, dict[str, Any]], Move]  # raises InvalidMove
    probes: Probes  # private side calls made before the move


def model_player(
    model: Model, probe_log: list[ProbeResponse], presenter: Presenter
) -> Player:
    messages: list[ChatMessage] = []
    pending_tool_call_ids: list[str] = []
    last_contract: ContractState | None = None  # only new contract news is shown

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal last_contract
        me = observation.recipient
        if observation.briefing is not None:
            messages.append(
                ChatMessageSystem(
                    content=presenter.system_prompt.format(tool=SUBMIT_MOVE)
                )
            )
        feedback = context.error or "Move received."
        for tool_call_id in pending_tool_call_ids:
            messages.append(
                ChatMessageTool(
                    content=feedback, tool_call_id=tool_call_id, function=SUBMIT_MOVE
                )
            )
        if context.error and not pending_tool_call_ids:
            messages.append(
                ChatMessageUser(
                    content=f"{context.error} Call {SUBMIT_MOVE} exactly once."
                )
            )
        pending_tool_call_ids.clear()
        if context.error is None:
            if observation.contract == last_contract:
                shown = observation.model_copy(update={"contract": None})
            else:
                shown = observation
                last_contract = observation.contract
            messages.append(
                ChatMessageUser(content=presenter.render(me, shown, context))
            )
            probe_log.extend(
                await presenter.probes(model, messages, me, observation, context)
            )

        output = await model.generate(messages, tools=[presenter.tool(context)])
        messages.append(output.message)
        tool_calls = output.message.tool_calls or []
        pending_tool_call_ids.extend(call.id for call in tool_calls)
        if len(tool_calls) != 1 or tool_calls[0].function != SUBMIT_MOVE:
            raise InvalidMove(f"You must call {SUBMIT_MOVE} exactly once.")
        return presenter.parse(me, tool_calls[0].arguments)

    return play


async def ask_probe(
    model: Model,
    messages: list[ChatMessage],
    question: str,
    field: str,
    description: str,
) -> float | None:
    """Ask a private numeric question on a copy of the conversation.

    The copy is discarded, so the probe never enters the conversation.
    """
    tool = ToolInfo(
        name=ANSWER_PROBE,
        description="Answer the private question.",
        parameters=ToolParams(
            properties={field: JSONSchema(type="number", description=description)},
            required=[field],
        ),
    )
    prompt = ChatMessageUser(
        content=f"{question} Answer by calling the {ANSWER_PROBE} tool."
    )
    output = await model.generate([*messages, prompt], tools=[tool])
    for call in output.message.tool_calls or []:
        value = call.arguments.get(field)
        if call.function == ANSWER_PROBE and isinstance(value, int | float):
            return float(value)
    return None


def label(field: str) -> str:
    return field.replace("_", " ")
