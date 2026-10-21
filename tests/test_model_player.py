"""The model player's probes: after the move, in a fresh conversation, with
options, and with the round they belong to. A mock model plays a bare
presenter, without Inspect's task machinery."""

import asyncio

from inspect_ai.model import ModelOutput, get_model
from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util import JSONSchema

from spar_harness.model_player import (
    ANSWER_WITH_CHOICE_TOOL,
    Presenter,
    ProbeResponse,
    model_player,
)
from spar_harness.schema import Move, Observation, Offer, PhaseContext

SUBMIT = ToolInfo(
    name="submit_move",
    description="Make your move.",
    parameters=ToolParams(
        properties={"split": JSONSchema(type="number")}, required=["split"]
    ),
)


def probe(ask_result, probe_id, me):
    return ProbeResponse(
        player=me, probe_id=probe_id, answer=ask_result, ground_truth=None
    )


async def before(ask, me, observation, context):
    return [probe(await ask("Private question: before your move?"), "before", me)]


async def after(ask, me, observation, context):
    choice, confidence = await ask.choice(
        "Private question: strike or accident?", ["strike", "accident"]
    )
    return [
        probe(await ask("Private question: after your move?"), "after", me),
        probe(await ask.fresh("Private question: 3 + 4?"), "fresh", me),
        ProbeResponse(
            player=me,
            probe_id="strike",
            answer=confidence,
            choice=choice,
            ground_truth=None,
        ),
    ]


PRESENTER = Presenter(
    system_prompt="You play a toy game. Call {tool} to move.",
    render=lambda me, observation, context: f"Round {context.round}: your move.",
    tool=lambda observation, context: SUBMIT,
    parse=lambda me, arguments: Move(action=Offer(split=arguments["split"])),
    probes=before,
    after_move=after,
    answer_tool=ANSWER_WITH_CHOICE_TOOL,
)


def play_two_rounds():
    """Returns every model call (input, tools) and the probe log."""
    calls = []

    def mock(input, tools, tool_choice, config):
        calls.append((input, tools))
        prompt = [message for message in input if message.role == "user"][-1].text
        if "strike or accident" in prompt:
            arguments = {"answer": 70, "choice": "strike"}
            return ModelOutput.for_tool_call("mock", "answer_probe", arguments)
        if "Private question" in prompt:
            return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 7})
        return ModelOutput.for_tool_call("mock", "submit_move", {"split": 0.4})

    log = []
    play = model_player(get_model("mockllm/model", custom_outputs=mock), log, PRESENTER)

    async def two_rounds():
        for round_number in (1, 2):
            observation = Observation(recipient="alpha", round=round_number)
            context = PhaseContext(
                round=round_number,
                role="proposer",
                action_kind="offer",
                allowed_actions={"offer"},
            )
            await play(observation, context)

    asyncio.run(two_rounds())
    return calls, log


def asked(calls, question):
    return [(input, tools) for input, tools in calls if question in input[-1].text]


def test_a_probe_after_the_move_sees_the_move_and_nothing_new():
    calls, _ = play_two_rounds()
    input, _ = asked(calls, "after your move")[0]
    *_, move, result, question = input
    assert move.tool_calls[0].function == "submit_move"
    assert result.role == "tool" and result.text == "Move received."
    assert question.role == "user"
    assert not any("Round 2" in message.text for message in input)


def test_probes_never_enter_the_game_conversation():
    calls, _ = play_two_rounds()
    moves = [input for input, _ in calls if "Private question" not in input[-1].text]
    assert len(moves) == 2
    for input in moves:
        assert all("Private question" not in message.text for message in input)


def test_a_fresh_probe_sends_no_game_history_and_only_the_answer_tool():
    calls, _ = play_two_rounds()
    for input, tools in asked(calls, "3 + 4"):
        assert [message.role for message in input] == ["user"]
        assert [tool.name for tool in tools] == ["answer_probe"]


def test_a_probe_with_options_logs_the_choice_and_the_number():
    _, log = play_two_rounds()
    strike = [response for response in log if response.probe_id == "strike"]
    assert [(response.choice, response.answer) for response in strike] == [
        ("strike", 70.0),
        ("strike", 70.0),
    ]


def test_every_probe_answer_is_logged_with_its_round():
    _, log = play_two_rounds()
    assert [(response.probe_id, response.round) for response in log] == [
        ("before", 1),
        ("after", 1),
        ("fresh", 1),
        ("strike", 1),
        ("before", 2),
        ("after", 2),
        ("fresh", 2),
        ("strike", 2),
    ]
