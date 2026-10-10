"""End to end: a scripted mock model plays contract dynamics through Inspect."""

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from spar_harness.contract_task import contract_dynamics

leaks = []


def scripted_model(input, tools, tool_choice, config):
    """Answers probes, proposes a credible fair contract, signs, pleads, honors."""
    prompt = [message for message in input if message.role == "user"][-1].text
    if "Private question" in prompt:
        return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.5})
    if "Private question" in str([message.model_dump() for message in input]):
        leaks.append(input)
    if "send a message" in prompt:
        arguments = {"message": "Please honor."}
    elif "You are the proposer" in prompt:
        arguments = {"action": "propose", "probability_you_bear": 0.5, "transfer": 20}
    elif "You are the responder" in prompt:
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


def test_the_tools_never_change_within_a_contract_conversation():
    from test_task import same_tools_throughout

    [log] = eval(
        contract_dynamics(
            n_episodes=1,
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
    same_tools_throughout(log)


def test_without_transfers_the_model_is_never_offered_one():
    [log] = eval(
        contract_dynamics(n_episodes=1, a="model", b="always_honor", shock_prob=1.0),
        model_roles={"a": get_model("mockllm/model", custom_outputs=scripted_model)},
        display="none",
        log_dir="logs/tests",
    )
    for sample in log.samples:
        calls = [event for event in sample.events if event.event == "model"]
        submit_move = next(
            tool for tool in calls[0].tools if tool.name == "submit_move"
        )
        assert "transfer" not in submit_move.parameters.properties
        briefing = next(m for m in calls[0].input if m.role == "user").text
        assert "transfer" not in briefing.replace("cannot include a transfer", "")


def test_a_transfer_cap_is_part_of_the_schema():
    [log] = eval(
        contract_dynamics(
            n_episodes=1, a="model", b="always_honor", transfers="capped"
        ),
        model_roles={"a": get_model("mockllm/model", custom_outputs=scripted_model)},
        display="none",
        log_dir="logs/tests",
    )
    calls = [event for event in log.samples[0].events if event.event == "model"]
    submit_move = next(tool for tool in calls[0].tools if tool.name == "submit_move")
    assert submit_move.parameters.properties["transfer"].maximum == 10.0
