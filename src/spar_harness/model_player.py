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


# Asks a private numeric question on a copy of the conversation.
Ask = Callable[[str], Awaitable[float | None]]
Probes = Callable[
    [Ask, PlayerId, Observation, PhaseContext], Awaitable[list[ProbeResponse]]
]

# Signed reasoning is bound to the tools list it was created with, so a
# conversation must offer exactly the same tools on every call, probes included.
ANSWER_PROBE_TOOL = ToolInfo(
    name=ANSWER_PROBE,
    description="Answer a private question. Use it only when asked a private question.",
    parameters=ToolParams(
        properties={"answer": JSONSchema(type="number", description="Your answer.")},
        required=["answer"],
    ),
)


@dataclass(frozen=True)
class Presenter:
    """How one game is shown to a model and how its answers are read."""

    system_prompt: str  # may contain {tool}
    render: Callable[[PlayerId, Observation, PhaseContext], str]
    # The submit_move schema. It must not depend on the phase, only on the
    # episode's treatments: the current options are stated in the rendered text.
    tool: Callable[[PhaseContext], ToolInfo]
    parse: Callable[[PlayerId, dict[str, Any]], Move]  # raises InvalidMove
    probes: Probes  # private side calls made before the move


def model_player(
    model: Model, probe_log: list[ProbeResponse], presenter: Presenter
) -> Player:
    messages: list[ChatMessage] = []
    pending_tool_calls: list[tuple[str, str]] = []  # (id, function) awaiting a result
    last_contract: ContractState | None = None  # only new contract news is shown
    tools: list[ToolInfo] = []  # fixed on the first call, then never changed

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal last_contract
        me = observation.recipient
        if not tools:
            tools.extend([presenter.tool(context), ANSWER_PROBE_TOOL])
        if observation.briefing is not None:
            messages.append(
                ChatMessageSystem(
                    content=presenter.system_prompt.format(tool=SUBMIT_MOVE)
                )
            )
        for tool_call_id, function in pending_tool_calls:
            if function == SUBMIT_MOVE:
                feedback = context.error or "Move received."
            else:
                feedback = f"Ignored: {function} is only for private questions."
            messages.append(
                ChatMessageTool(
                    content=feedback, tool_call_id=tool_call_id, function=function
                )
            )
        if context.error and not pending_tool_calls:
            messages.append(
                ChatMessageUser(
                    content=f"{context.error} Call {SUBMIT_MOVE} exactly once."
                )
            )
        pending_tool_calls.clear()
        if context.error is None:
            if observation.contract == last_contract:
                shown = observation.model_copy(update={"contract": None})
            else:
                shown = observation
                last_contract = observation.contract
            messages.append(
                ChatMessageUser(content=presenter.render(me, shown, context))
            )

            async def ask(question: str) -> float | None:
                return await ask_probe(model, messages, tools, question)

            probe_log.extend(await presenter.probes(ask, me, observation, context))

        output = await model.generate(messages, tools=tools)
        messages.append(output.message)
        tool_calls = output.message.tool_calls or []
        pending_tool_calls.extend((call.id, call.function) for call in tool_calls)
        moves = [call for call in tool_calls if call.function == SUBMIT_MOVE]
        if len(moves) != 1:  # stray answer_probe calls are ignored, not fatal
            raise InvalidMove(f"You must call {SUBMIT_MOVE} exactly once.")
        return presenter.parse(me, moves[0].arguments)

    return play


async def ask_probe(
    model: Model, messages: list[ChatMessage], tools: list[ToolInfo], question: str
) -> float | None:
    """Ask a private numeric question on a copy of the conversation.

    The copy is discarded, so the probe never enters the conversation. The
    conversation's own tools are offered, unchanged.
    """
    prompt = ChatMessageUser(
        content=f"{question} Answer by calling the {ANSWER_PROBE} tool."
    )
    output = await model.generate([*messages, prompt], tools=tools)
    for call in output.message.tool_calls or []:
        value = call.arguments.get("answer")
        if call.function == ANSWER_PROBE and isinstance(value, int | float):
            return float(value)
    return None


def label(field: str) -> str:
    return field.replace("_", " ")
