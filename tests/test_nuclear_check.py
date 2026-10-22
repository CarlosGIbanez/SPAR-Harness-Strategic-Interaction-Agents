"""The rung-order check (F2, step 0), with mock models."""

import re

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from spar_harness.nuclear_check import nuclear_rung_order, rung_pair
from spar_harness.nuclear_crisis import MAX_MISFIRE, STRATEGIC_WAR
from spar_harness.nuclear_presenter import RUNG_OF


def test_pairs_are_one_to_three_rungs_apart_on_the_ladder():
    pairs = [rung_pair(seed) for seed in range(200)]
    assert all(1 <= abs(first - second) <= MAX_MISFIRE for first, second in pairs)
    assert all(1 <= min(pair) and max(pair) <= STRATEGIC_WAR for pair in pairs)
    assert any(first > second for first, second in pairs)  # either order is asked


def answering(pick):
    """A mock model that picks one of the two options named in the question."""
    prompts = []

    def mock(input, tools, tool_choice, config):
        prompts.append(input)
        question = input[-1].text
        names = re.search(r"(?:escalatory|stronger): (.+) or (.+)\? Pick", question)
        names = names.groups()
        arguments = {"answer": 90, "choice": pick(*names)}
        return ModelOutput.for_tool_call("mock", "answer_probe", arguments)

    return mock, prompts


def run_check(pick, framing="military"):
    mock, prompts = answering(pick)
    [log] = eval(
        nuclear_rung_order(n_questions=10, framing=framing),
        model=get_model("mockllm/model", custom_outputs=mock),
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    return log.results.scores[0].metrics["accuracy"].value, prompts


def test_a_model_that_reads_the_ladder_gets_every_pair_right():
    accuracy, prompts = run_check(lambda a, b: max(a, b, key=RUNG_OF.get))
    assert accuracy == 1.0
    # asked after the game's briefing, as in an episode's first round
    assert all("The options, mildest first:" in input[1].text for input in prompts)


def test_the_neutral_framing_asks_in_its_own_words():
    accuracy, prompts = run_check(lambda a, b: max(a, b, key=RUNG_OF.get), "neutral")
    assert accuracy == 1.0
    for input in prompts:
        assert "strategy game" in input[1].text
        assert "that is stronger:" in input[-1].text
        assert "escalatory" not in input[-1].text


def test_the_milder_pick_is_wrong_and_any_case_is_read():
    milder, _ = run_check(lambda a, b: min(a, b, key=RUNG_OF.get))
    shouting, _ = run_check(lambda a, b: max(a, b, key=RUNG_OF.get).upper())
    assert (milder, shouting) == (0.0, 1.0)
