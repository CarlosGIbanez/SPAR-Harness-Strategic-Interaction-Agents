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
from spar_harness.schema import (
    ContractState,
    EpisodeState,
    Move,
    Observation,
    PhaseContext,
    PlayerId,
)

SUBMIT_MOVE = "submit_move"
ANSWER_PROBE = "answer_probe"


class ProbeResponse(BaseModel):
    player: PlayerId
    probe_id: str
    answer: float | bool | None  # None if the model gave no usable answer
    ground_truth: float | bool | None  # None if only known after the episode
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
    number. `ask.choice` also returns the option picked, `ask.fresh` asks in a
    new conversation, with no role and no game history, and `ask.with_tool`
    asks on a copy with a tool of the presenter's own.
    """

    model: Model
    messages: list[ChatMessage]  # the conversation so far, never changed
    tools: list[ToolInfo]  # the conversation's own tools
    answer_tool: ToolInfo | None  # the only tool of a fresh conversation

    async def __call__(self, question: str) -> float | None:
        return await ask_probe(self.model, self.messages, self.tools, question)

    async def choice(
        self, question: str, options: Collection[str]
    ) -> tuple[str | None, float | None]:
        """The option picked (None if it isn't one of the options) and the number.

        The presenter's answer tool needs a choice field: ANSWER_WITH_CHOICE_TOOL.
        """
        number, picked = await _answer(self.model, self.messages, self.tools, question)
        by_name = {option.lower(): option for option in options}
        return by_name.get((picked or "").strip().lower()), number

    async def fresh(self, question: str) -> float | None:
        number, _ = await _answer(self.model, [], [self.answer_tool], question)
        return number

    async def with_tool(self, question: str, tool: ToolInfo) -> dict[str, Any]:
        """The arguments of the model's last call to `tool`, or {} if none.

        The question is sent as written, and `tool` is the only tool offered.
        Only safe before the model's first move: signed reasoning in earlier
        turns is bound to the conversation's own tools.
        """
        prompt = ChatMessageUser(content=question)
        output = await self.model.generate([*self.messages, prompt], tools=[tool])
        arguments: dict[str, Any] = {}
        for call in output.message.tool_calls or []:
            if call.function == tool.name:
                arguments = call.arguments
        return arguments


Probes = Callable[
    [Ask, PlayerId, Observation, PhaseContext], Awaitable[list[ProbeResponse]]
]
# An opening side call made after the probes; the text it returns is added to the
# conversation as a user message the model keeps seeing (None adds nothing).
Opening = Callable[[Ask, PlayerId, Observation, PhaseContext], Awaitable[str | None]]


@dataclass(frozen=True)
class Presenter:
    """How one game is shown to a model and how its answers are read."""

    system_prompt: str  # may contain {tool}
    render: Callable[[PlayerId, Observation, PhaseContext], str]
    # The submit_move schema, built once from the player's first observation
    # (which carries the briefing). It may depend on the episode's treatments
    # but never on the phase: the current options are stated in the text.
    tool: Callable[[Observation, PhaseContext], ToolInfo]
    parse: Callable[[PlayerId, dict[str, Any], PhaseContext], Move]  # InvalidMove
    probes: Probes  # private side calls made before the move
    # private side calls made right after the move, before anything new is shown
    after_move: Probes | None = None
    # The tool probes are answered with, offered in the conversation next to
    # submit_move; ANSWER_WITH_CHOICE_TOOL to pick options. None: submit_move
    # is the conversation's only tool.
    answer_tool: ToolInfo | None = field(default_factory=lambda: ANSWER_PROBE_TOOL)
    # Rewrites a referee error in the game's own words before the model reads
    # it, e.g. the setting's action labels in place of internal action names.
    error: Callable[[str], str] = lambda message: message
    # After a default move, tell the model its last move was replaced (REPLACED).
    announce_default: bool = True
    # When the model answered without a tool call, add "Call submit_move exactly
    # once." even if the error already says so. Contract dynamics doesn't.
    repeat_reminder: bool = True
    # Accept a reply whose only submit_move call is valid, and answer its other
    # calls with "Ignored". Off: anything but exactly one call, to submit_move,
    # is an invalid move, and every call is answered as a move.
    ignore_stray_calls: bool = True
    # Reject a submit_move call that lacks a required argument before parsing.
    check_required: bool = True
    # A question for after the episode, asked on a copy of the conversation;
    # the model player then has a `debrief(state)` coroutine returning the reply.
    debrief: Callable[[EpisodeState, PlayerId], str] | None = None


def model_player(
    model: Model,
    probe_log: list[ProbeResponse],
    presenter: Presenter,
    opening: Opening | None = None,
) -> Player:
    messages: list[ChatMessage] = []
    pending_tool_calls: list[tuple[str, str]] = []  # (id, function) awaiting a result
    last_contract: ContractState | None = None  # only new contract news is shown
    tools: list[ToolInfo] = []  # fixed on the first call, then never changed
    position: list[PlayerId] = []  # the player's position, known from its first call
    # A retry's observation is empty, so probes after the move use the round's.
    round_observation: Observation | None = None
    last_failed = False  # the last call raised, so a new round means a default move

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal last_contract, round_observation, last_failed
        replaced = context.error is None and last_failed and presenter.announce_default
        me = observation.recipient
        error = presenter.error(context.error) if context.error else None
        if not tools:
            tools.append(presenter.tool(observation, context))
            if presenter.answer_tool is not None:
                tools.append(presenter.answer_tool)
            position.append(observation.recipient)
        if observation.briefing is not None:
            messages.append(
                ChatMessageSystem(
                    content=presenter.system_prompt.format(tool=SUBMIT_MOVE)
                )
            )
        for tool_call_id, function in pending_tool_calls:
            if not presenter.ignore_stray_calls:
                function = SUBMIT_MOVE  # every call is answered as a move
            messages.append(
                ChatMessageTool(
                    content=_feedback(
                        function, error or (REPLACED if replaced else None)
                    ),
                    tool_call_id=tool_call_id,
                    function=function,
                )
            )
        if error and not pending_tool_calls:
            reminder = f"Call {SUBMIT_MOVE} exactly once."
            said = "exactly once" in error and not presenter.repeat_reminder
            content = error if said else f"{error} {reminder}"
            messages.append(ChatMessageUser(content=content))
        notice = f"{REPLACED}\n\n" if replaced and not pending_tool_calls else ""
        pending_tool_calls.clear()
        if context.error is None:
            round_observation = observation
            if observation.contract == last_contract:
                shown = observation.model_copy(update={"contract": None})
            else:
                shown = observation
                last_contract = observation.contract
            messages.append(
                ChatMessageUser(content=notice + presenter.render(me, shown, context))
            )
            ask = Ask(model, messages, tools, presenter.answer_tool)
            probes = await presenter.probes(ask, me, observation, context)
            probe_log.extend(_in_round(probes, context))
            if opening is not None:
                note = await opening(ask, me, observation, context)
                if note is not None:
                    messages.append(ChatMessageUser(content=note))

        last_failed = True
        output = await model.generate(messages, tools=tools)
        messages.append(output.message)
        tool_calls = output.message.tool_calls or []
        pending_tool_calls.extend((call.id, call.function) for call in tool_calls)
        moves = tool_calls
        if presenter.ignore_stray_calls:
            moves = [call for call in tool_calls if call.function == SUBMIT_MOVE]
        if len(moves) != 1 or moves[0].function != SUBMIT_MOVE:
            raise InvalidMove(f"You must call {SUBMIT_MOVE} exactly once.")
        arguments = moves[0].arguments
        missing = [
            name for name in tools[0].parameters.required if arguments.get(name) is None
        ]
        if missing and presenter.check_required:
            raise InvalidMove(f"{SUBMIT_MOVE} is missing {', '.join(missing)}.")
        move = presenter.parse(me, arguments, context)
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
            probes = await presenter.after_move(after, me, round_observation, context)
            probe_log.extend(_in_round(probes, context))
        last_failed = False
        return move

    async def debrief(state: EpisodeState) -> str:
        """Ask the presenter's post-episode question on a copy of the conversation,
        with the same tools but none allowed, and return the reply."""
        copy = list(messages)
        for tool_call_id, function in pending_tool_calls:
            if not presenter.ignore_stray_calls:
                function = SUBMIT_MOVE
            copy.append(
                ChatMessageTool(
                    content=_feedback(function, None),
                    tool_call_id=tool_call_id,
                    function=function,
                )
            )
        copy.append(ChatMessageUser(content=presenter.debrief(state, position[0])))
        output = await model.generate(copy, tools=tools, tool_choice="none")
        return output.message.text

    if presenter.debrief is not None:
        play.debrief = debrief  # type: ignore[attr-defined]
    return play


REPLACED = "Your last move was not accepted, so the referee made the default move."


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
    calls = [
        call
        for call in output.message.tool_calls or []
        if call.function == ANSWER_PROBE
    ]
    for call in calls:
        value, choice = call.arguments.get("answer"), call.arguments.get("choice")
        if isinstance(value, int | float):
            return float(value), choice if isinstance(choice, str) else None
    # no usable number, but a choice still counts
    choices = [call.arguments.get("choice") for call in calls]
    return None, next((choice for choice in choices if isinstance(choice, str)), None)


def label(field: str) -> str:
    return field.replace("_", " ")
