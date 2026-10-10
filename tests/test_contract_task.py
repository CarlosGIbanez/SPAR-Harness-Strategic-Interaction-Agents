"""End to end: a scripted mock model plays contract dynamics through Inspect."""

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model
from spar_harness.contract_task import contract_dynamics

leaks = []


def scripted_model(input, tools, tool_choice, config):
    """Answers probes, proposes a credible fair contract, signs, pleads, honors."""
    transcript = str([message.model_dump() for message in input])
    tool = tools[0]
    if tool.name == "answer_probe":
        [field] = tool.parameters.required
        return ModelOutput.for_tool_call("mock", "answer_probe", {field: 0.5})
    if "Private question" in transcript:
        leaks.append(transcript)
    properties = tool.parameters.properties
    if "action" not in properties:  # a persuasion message
        return ModelOutput.for_tool_call(
            "mock", "submit_move", {"message": "Please honor."}
        )
    allowed = properties["action"].enum
    if "propose" in allowed:
        arguments = {"action": "propose", "probability_you_bear": 0.5, "transfer": 20}
    elif "accept" in allowed:
        arguments = {"action": "accept"}
    else:
        arguments = {"action": "honor"}
    return ModelOutput.for_tool_call("mock", "submit_move", arguments)


def test_model_plays_a_pair_of_contract_episodes_against_the_rational_bot():
    [log] = eval(
        contract_dynamics(
            n_episodes=2,
            a="model",
            b="rational",
            transfers="free",
            shock_prob=1.0,
            benefit_alpha=40.0,
            benefit_beta=40.0,
            persuasion_rounds=1,
        ),
        model_roles={"a": get_model("mockllm/model", custom_outputs=scripted_model)},
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    assert len(log.samples) == 4
    for sample in log.samples:
        episode = sample.store["episode"]
        assert all(not record["defaulted"] for record in episode["rounds"])
        assert episode["ending"] == "honored"  # credible contract; both bearers honor
        outcome = sample.scores["contract_outcome_scorer"].value
        assert outcome["signed"] is True and outcome["honored"] is True
        seats = sample.scores["contract_seat_scorer"].value
        assert "a_belief" in seats and "a_comprehension" in seats
    assert leaks == []  # probes never enter the conversation
