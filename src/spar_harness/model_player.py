"""Adapter that lets an Inspect model act as a Player.

The model keeps one persistent conversation and moves by calling the
submit_move tool. Everything game-specific (prompt, rendering, tool schema,
parsing, probes) comes from the game's Presenter.
"""

from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, field
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
    choice: str | None = None  # the option picked, for a question with options
    round: int | None = None  # the round of the move the probe went with


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
# The answer tool for presenters whose probes also ask the model to pick an option.
ANSWER_WITH_CHOICE_TOOL = ToolInfo(
    name=ANSWER_PROBE,
    description=ANSWER_PROBE_TOOL.description,
    parameters=ToolParams(
        properties={
            "answer": JSONSchema(type="number", description="Your answer."),
            "choice": JSONSchema(
                type="string",
                description="If the question lists options: the option you pick.",
            ),
        },
        required=["answer"],
    ),
)


@dataclass(frozen=True)
class Ask:
    """Private questions for probes; none of them enters the game conversation.

    `await ask(question)` asks on a copy of the conversation and returns the
    number. `ask.choice` also returns the option picked, and `ask.fresh` asks
    in a new conversation, with no role and no game history.
    """

    model: Model
    messages: list[ChatMessage]  # the conversation so far, never changed
    tools: list[ToolInfo]  # the conversation's own tools
    answer_tool: ToolInfo  # the only tool of a fresh conversation

    async def __call__(self, question: str) -> float | None:
        return await ask_probe(self.model, self.messages, self.tools, question)

    async def choice(
        self, question: str, options: Collection[str]
    ) -> tuple[str | None, float | None]:
        """The option picked (None if it isn't one of the options) and the number.

        The presenter's answer tool needs a choice field: ANSWER_WITH_CHOICE_TOOL.
        """
        number, picked = await _answer(self.model, self.messages, self.tools, question)
        return (picked if picked in options else None), number

    async def fresh(self, question: str) -> float | None:
        number, _ = await _answer(self.model, [], [self.answer_tool], question)
        return number


Probes = Callable[
    [Ask, PlayerId, Observation, PhaseContext], Awaitable[list[ProbeResponse]]
]


@dataclass(frozen=True)
class Presenter:
    """How one game is shown to a model and how its answers are read."""

    system_prompt: str  # may contain {tool}
    render: Callable[[PlayerId, Observation, PhaseContext], str]
    # The submit_move schema, built once from the player's first observation
    # (which carries the briefing). It may depend on the episode's treatments
    # but never on the phase: the current options are stated in the text.
    tool: Callable[[Observation, PhaseContext], ToolInfo]
    parse: Callable[[PlayerId, dict[str, Any]], Move]  # raises InvalidMove
    probes: Probes  # private side calls made before the move
    # private side calls made right after the move, before anything new is shown
    after_move: Probes | None = None
    # the tool probes are answered with; ANSWER_WITH_CHOICE_TOOL to pick options
    answer_tool: ToolInfo = field(default_factory=lambda: ANSWER_PROBE_TOOL)


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
            tools.extend([presenter.tool(observation, context), presenter.answer_tool])
        if observation.briefing is not None:
            messages.append(
                ChatMessageSystem(
                    content=presenter.system_prompt.format(tool=SUBMIT_MOVE)
                )
            )
        for tool_call_id, function in pending_tool_calls:
            messages.append(
                ChatMessageTool(
                    content=_feedback(function, context.error),
                    tool_call_id=tool_call_id,
                    function=function,
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
            ask = Ask(model, messages, tools, presenter.answer_tool)
            probes = await presenter.probes(ask, me, observation, context)
            probe_log.extend(_in_round(probes, context))

        output = await model.generate(messages, tools=tools)
        messages.append(output.message)
        tool_calls = output.message.tool_calls or []
        pending_tool_calls.extend((call.id, call.function) for call in tool_calls)
        moves = [call for call in tool_calls if call.function == SUBMIT_MOVE]
        if len(moves) != 1:  # stray answer_probe calls are ignored, not fatal
            raise InvalidMove(f"You must call {SUBMIT_MOVE} exactly once.")
        move = presenter.parse(me, moves[0].arguments)
        if presenter.after_move is not None:
            # As if the move were accepted: its calls answered, nothing new shown.
            answered = [
                ChatMessageTool(
                    content=_feedback(function, None),
                    tool_call_id=tool_call_id,
                    function=function,
                )
                for tool_call_id, function in pending_tool_calls
            ]
            after = Ask(model, [*messages, *answered], tools, presenter.answer_tool)
            probes = await presenter.after_move(after, me, observation, context)
            probe_log.extend(_in_round(probes, context))
        return move

    return play


def _feedback(function: str, error: str | None) -> str:
    """The tool result for one of the model's calls."""
    if function == SUBMIT_MOVE:
        return error or "Move received."
    return f"Ignored: {function} is only for private questions."


def _in_round(
    probes: list[ProbeResponse], context: PhaseContext
) -> list[ProbeResponse]:
    return [probe.model_copy(update={"round": context.round}) for probe in probes]


async def ask_probe(
    model: Model, messages: list[ChatMessage], tools: list[ToolInfo], question: str
) -> float | None:
    """Ask a private numeric question on a copy of the conversation.

    The copy is discarded, so the probe never enters the conversation. The
    conversation's own tools are offered, unchanged.
    """
    number, _ = await _answer(model, messages, tools, question)
    return number


async def _answer(
    model: Model, messages: list[ChatMessage], tools: list[ToolInfo], question: str
) -> tuple[float | None, str | None]:
    """Ask a private question after the given messages, which stay unchanged.

    Returns the number and, if the answer tool has a choice field, the choice.
    """
    prompt = ChatMessageUser(
        content=f"{question} Answer by calling the {ANSWER_PROBE} tool."
    )
    output = await model.generate([*messages, prompt], tools=tools)
    for call in output.message.tool_calls or []:
        value = call.arguments.get("answer")
        if call.function == ANSWER_PROBE and isinstance(value, int | float):
            choice = call.arguments.get("choice")
            return float(value), choice if isinstance(choice, str) else None
    return None, None


def label(field: str) -> str:
    return field.replace("_", " ")
