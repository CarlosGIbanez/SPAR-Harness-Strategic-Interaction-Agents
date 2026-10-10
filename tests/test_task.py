"""End to end: a scripted mock model plays through Inspect against a bot."""

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from spar_harness.task import crisis_bargaining

probe_leaks = []


def last_user_text(input):
    """What the mock reads to decide, like a real model: the latest prompt."""
    return [message for message in input if message.role == "user"][-1].text


def is_probe(input):
    return "private question" in last_user_text(input).lower()


def scripted_model(input, tools, tool_choice, config):
    """Answers the probe, offers itself 0.6 when proposing, and accepts otherwise."""
    if is_probe(input):
        return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.4})
    if "private question" in str([message.model_dump() for message in input]):
        probe_leaks.append(input)
    if "You are the proposer" in last_user_text(input):
        return ModelOutput.for_tool_call(
            "mock", "submit_move", {"action": "offer", "own_share": 0.6}
        )
    return ModelOutput.for_tool_call("mock", "submit_move", {"action": "accept"})


def test_model_plays_a_pair_of_episodes_against_the_equilibrium_bot():
    [log] = eval(
        crisis_bargaining(n_episodes=1, a="model", b="equilibrium"),
        model_roles={"a": get_model("mockllm/model", custom_outputs=scripted_model)},
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    assert len(log.samples) == 2  # one pair: seat a plays alpha, then beta
    for sample in log.samples:
        episode = sample.store["episode"]
        assert episode["ending"] in {"agreement", "fight", "horizon", "random_stop"}
        assert all(not record["defaulted"] for record in episode["rounds"])
        scores = sample.scores["outcome_scorer"].value
        assert set(scores) >= {"fight", "agreement", "efficiency"}
        assert sample.scores["seat_scorer"].value["a_comprehension"] in (0, 1)
    # the probe was a side call: the model never saw it again when moving
    assert probe_leaks == []


def test_model_recovers_from_an_invalid_move_with_a_well_formed_conversation():
    malformed_problems = []
    fumbled = []

    def fumbling_model(input, tools, tool_choice, config):
        # every tool call the model made earlier must be answered by a tool result
        for message, following in zip(input, input[1:] + [None]):
            answered = following is not None and following.role == "tool"
            if getattr(message, "tool_calls", None) and not answered:
                malformed_problems.append(message)
        if is_probe(input):
            return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.4})
        if "You are the proposer" in last_user_text(input):
            if not fumbled:
                fumbled.append(True)
                return ModelOutput.for_tool_call(
                    "mock", "submit_move", {"action": "offer"}
                )
            return ModelOutput.for_tool_call(
                "mock", "submit_move", {"action": "offer", "own_share": 0.6}
            )
        return ModelOutput.for_tool_call("mock", "submit_move", {"action": "accept"})

    [log] = eval(
        crisis_bargaining(n_episodes=1, a="model", b="equilibrium"),
        model_roles={"a": get_model("mockllm/model", custom_outputs=fumbling_model)},
        display="none",
        log_dir="logs/tests",
    )
    assert fumbled
    assert malformed_problems == []
    for sample in log.samples:
        assert all(
            not record["defaulted"] for record in sample.store["episode"]["rounds"]
        )


def test_undefined_measures_are_left_out_of_the_averages():
    import math

    [log] = eval(
        crisis_bargaining(n_episodes=3, a="equilibrium", b="equilibrium"),
        model="mockllm/model",
        display="none",
        log_dir="logs/tests",
    )
    metrics = {
        score.name: score.metrics["spar_harness/mean_of_defined"].value
        for score in log.results.scores
    }
    # complete information: every episode agrees in round 1, inside the range
    assert metrics["rounds_to_agreement"] == 1.0
    assert metrics["in_range"] == 1.0
    # bots answer no probes and declare no actions: undefined, not zero
    assert math.isnan(metrics["a_comprehension"])
    assert math.isnan(metrics["b_intention_gap"])


def same_tools_throughout(log):
    """Signed reasoning is bound to the tools list it was created with, so every
    model call in an episode must send exactly the same tools."""
    for sample in log.samples:
        tool_lists = [
            [tool.model_dump() for tool in event.tools]
            for event in sample.events
            if event.event == "model"
        ]
        assert len(tool_lists) > 1
        assert all(tools == tool_lists[0] for tools in tool_lists), sample.id


def test_the_tools_never_change_within_a_conversation():
    [log] = eval(
        crisis_bargaining(n_episodes=1, a="model", b="equilibrium", messages=True),
        model_roles={"a": get_model("mockllm/model", custom_outputs=scripted_model)},
        display="none",
        log_dir="logs/tests",
    )
    same_tools_throughout(log)


def test_a_stray_probe_call_next_to_a_valid_move_is_ignored():
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.tool import ToolCall

    tool_results = []

    def model_with_stray_call(input, tools, tool_choice, config):
        tool_results.extend(m for m in input if m.role == "tool")
        if is_probe(input):
            return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.4})
        arguments = (
            {"action": "offer", "own_share": 0.6}
            if "You are the proposer" in last_user_text(input)
            else {"action": "accept"}
        )
        message = ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(id="move", function="submit_move", arguments=arguments),
                ToolCall(id="stray", function="answer_probe", arguments={}),
            ],
        )
        return ModelOutput.from_message(message)

    [log] = eval(
        crisis_bargaining(n_episodes=1, a="model", b="equilibrium"),
        model_roles={
            "a": get_model("mockllm/model", custom_outputs=model_with_stray_call)
        },
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    for result in tool_results:  # no move was rejected, and results are labeled right
        if result.function == "submit_move":
            assert result.text == "Move received."
        else:
            assert result.function == "answer_probe" and "Ignored" in result.text
