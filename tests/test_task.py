"""End to end: a scripted mock model plays through Inspect against a bot."""

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from spar_harness.task import crisis_bargaining

probe_leaks = []


def scripted_model(input, tools, tool_choice, config):
    """Answers the probe, offers itself 0.6 when proposing, and accepts otherwise."""
    names = [tool.name for tool in tools]
    if "answer_probe" in names:
        return ModelOutput.for_tool_call("mock", "answer_probe", {"fight_payoff": 0.4})
    if "private question" in str([message.model_dump() for message in input]):
        probe_leaks.append(input)
    allowed = tools[0].parameters.properties["action"].enum
    if "offer" in allowed:
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
        names = [tool.name for tool in tools]
        if "answer_probe" in names:
            return ModelOutput.for_tool_call(
                "mock", "answer_probe", {"fight_payoff": 0.4}
            )
        allowed = tools[0].parameters.properties["action"].enum
        if "offer" in allowed:
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
