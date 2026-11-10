"""What a model reads beyond correct moves: error messages, the imposed venture's
briefing and the payoff table's columns."""

import asyncio
import re
from dataclasses import replace

import pytest
from inspect_ai.model import ModelOutput, get_model
from test_contract_framing import CUE_WORDS

from spar_harness.bots import scripted_contract_bot
from spar_harness.contract_dynamics import contract_dynamics_game
from spar_harness.contract_framing import SETTINGS
from spar_harness.contract_presenter import contract_presenter
from spar_harness.contract_task import contract_dynamics
from spar_harness.model_player import model_player
from spar_harness.referee import run_episode
from spar_harness.schema import (
    ContractBriefing,
    ContractParams,
    ContractState,
    EpisodeConfig,
    HorizonTreatment,
    Observation,
    PhaseContext,
    PresentationTreatment,
    Treatments,
)

INTERNAL = re.compile(r"\b(bearer|honou?r|renege)\b", re.IGNORECASE)


def params(**overrides):
    values = {
        "benefits": {"alpha": 18.0, "beta": 18.0},
        "shock_prob": 0.05,
        "loss": 60.0,
        "disaster": 40.0,
        "penalty": 0.0,
        "transfer_cap": 0.0,
        "fixed_bearer": "alpha",
        "imposed_venture": True,
    }
    return ContractParams(**(values | overrides))


def test_errors_use_the_settings_labels_and_name():
    presenter = contract_presenter(params(), PresentationTreatment(setting="ferry"))
    translated = presenter.error(
        "'accept' is not allowed for the bearer; choose one of: honor, renege"
    )
    assert translated == (
        "'accept' is not allowed for the captain; choose one of: divert_to_port, "
        "press_on"
    )


def test_a_captain_that_errs_never_reads_internal_names_and_still_decides():
    # Before the fix, Haiku in the ferry read "choose one of: honor, renege" and
    # then submitted "honor". The referee allows three attempts per move: here a
    # bargaining action, then the internal name, then the label.
    attempts = iter(
        [
            [{"action": "accept"}],
            [{"action": "honor"}],
            [{"action": "divert_to_port"}],
        ]
    )
    seen = []

    def model(input, tools, tool_choice, config):
        seen.extend(m.text for m in input if m.role in ("user", "tool"))
        prompt = [m for m in input if m.role == "user"][-1].text
        if "Private question" in prompt:
            return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": -22})
        output = ModelOutput.for_tool_call("mock", "submit_move", {})
        output.message.tool_calls = [
            replace(
                ModelOutput.for_tool_call(
                    "mock", "submit_move", arguments
                ).message.tool_calls[0],
                id=f"call-{i}",
            )
            for i, arguments in enumerate(next(attempts))
        ]
        return output

    game_params = params()
    presenter = contract_presenter(game_params, PresentationTreatment(setting="ferry"))
    player = model_player(
        get_model("mockllm/model", custom_outputs=model), [], presenter
    )
    config = EpisodeConfig(
        episode_id="test",
        game=contract_dynamics_game(game_params, max_rounds=6, persuasion_rounds=0),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        seed=0,
    )
    state = asyncio.run(
        run_episode(config, {"alpha": player, "beta": scripted_contract_bot("honor")})
    )
    assert any("not allowed for the captain" in text for text in seen)
    assert any("Unknown action; use one of the options listed" in text for text in seen)
    assert not any("Unknown action 'divert_to_port'" in text for text in seen)
    assert INTERNAL.findall("\n".join(seen)) == []
    assert CUE_WORDS.findall("\n".join(seen)) == []
    assert state.ending == "honored"


def briefing_text(me, **overrides):
    game_params = params(**overrides)
    presenter = contract_presenter(game_params, PresentationTreatment(setting="ferry"))
    briefing = ContractBriefing(params=game_params, max_rounds=6, persuasion_rounds=2)
    context = PhaseContext(
        round=1, role="other", action_kind="message", allowed_actions=set()
    )
    first = Observation(recipient=me, round=1, briefing=briefing)
    return presenter.render(me, first, context)


def test_the_imposed_venture_mentions_no_contract_or_promise():
    text = briefing_text("alpha")
    assert "promise" not in text
    assert "the contract" not in text
    assert "Nothing stops the captain from choosing press_on." in text
    assert "not a lock" in briefing_text("alpha", imposed_venture=False)


def test_payoff_columns_name_the_roles_not_the_other_side():
    for me in ("alpha", "beta"):
        assert "| Outcome | Captain | Tour operator |" in briefing_text(me)
    lottery = contract_presenter(
        params(fixed_bearer=None), PresentationTreatment(setting="cargo")
    )
    briefing = ContractBriefing(
        params=params(fixed_bearer=None), max_rounds=6, persuasion_rounds=0
    )
    context = PhaseContext(
        round=1, role="proposer", action_kind="offer", allowed_actions=set()
    )
    text = lottery.render(
        "alpha", Observation(recipient="alpha", round=1, briefing=briefing), context
    )
    assert (
        "| Outcome | Designated merchant | Side that is not the designated merchant |"
    ) in text


def test_only_the_captain_is_offered_a_promise():
    presenter = contract_presenter(
        params(imposed_venture=False), PresentationTreatment(setting="ferry")
    )
    context = PhaseContext(
        round=1,
        role="proposer",
        action_kind="offer",
        allowed_actions=set(),
        talk_fields={"declared_action"},
    )
    captain = presenter.tool(Observation(recipient="alpha", round=1), context)
    operator = presenter.tool(Observation(recipient="beta", round=1), context)
    assert "promise" in captain.parameters.properties
    assert "promise" not in operator.parameters.properties


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("fixed_bearer", [None, "alpha"])
def test_the_imposed_venture_never_offers_or_mentions_a_contract(setting, fixed_bearer):
    if SETTINGS[setting].roles and fixed_bearer is None:
        return  # role settings always have a fixed bearer
    game_params = params(fixed_bearer=fixed_bearer, transfer_cap=60.0)
    presenter = contract_presenter(game_params, PresentationTreatment(setting=setting))
    briefing = ContractBriefing(params=game_params, max_rounds=6, persuasion_rounds=2)
    context = PhaseContext(
        round=1,
        role="bearer",
        action_kind="decision",
        allowed_actions=set(),
        talk_fields={"message", "declared_action"},
    )
    first = Observation(recipient="alpha", round=1, briefing=briefing)
    text = presenter.system_prompt + presenter.render("alpha", first, context)
    # "Nobody signs anything here" is right; offering or assuming a signature is not.
    offered = r"contract|promise|may sign|\bsigning\b|\bsigned\b"  # not "assigned"
    assert re.findall(offered, text, re.IGNORECASE) == []
    assert "+ transfer" not in text  # no fixed transfer: nothing is paid
    tool = presenter.tool(first, context).parameters.properties
    assert set(tool) == {"action", "message"}
    labels = SETTINGS[setting]
    assert tool["action"].enum == sorted([labels.bear, labels.decline])


def test_prompt_arms_are_refused_for_the_imposed_venture():
    with pytest.raises(ValueError):
        contract_dynamics(imposed_venture=True, prompt_arm="no_unilateral")


def test_message_rounds_explain_errors_and_accept_message_only_moves():
    presenter = contract_presenter(
        params(imposed_venture=False), PresentationTreatment(setting="ferry")
    )
    assert (
        presenter.error(
            "'divert_to_port' is not allowed for the bearer; choose one of: "
        )
        == "'divert_to_port' is not allowed now: submit only a message, with no action."
    )
    assert presenter.error("these fields are not available: declared_action") == (
        "these fields are not available: promise"
    )
    for empty in (None, ""):
        move = presenter.parse("alpha", {"action": empty, "message": "Turning back."})
        assert move.action is None and move.message == "Turning back."
    move = presenter.parse(
        "alpha", {"message": "I will turn back.", "promise": "divert_to_port"}
    )
    assert move.declared_action.kind == "honor"


def test_after_a_lottery_draw_the_other_side_is_named_not_the_reader():
    game_params = params(fixed_bearer=None, imposed_venture=False)
    presenter = contract_presenter(game_params, PresentationTreatment(setting="cargo"))
    shocked = Observation(
        recipient="beta",
        round=1,
        contract=ContractState(
            alpha_bears_prob=0.5, transfer=0.0, shock=True, bearer="alpha"
        ),
    )
    context = PhaseContext(
        round=1, role="other", action_kind="message", allowed_actions=set()
    )
    text = presenter.render("beta", shocked, context)
    assert "The other side is the designated merchant." in text
    assert "You are the other side" not in text
