"""Story settings, neutral labels, forced shocks and the persuader's belief."""

import asyncio
import re

import pytest
from inspect_ai import eval
from inspect_ai.model import get_model
from test_contract_task import scripted_model

from spar_harness.bots import scripted_contract_bot
from spar_harness.contract_dynamics import contract_dynamics_game
from spar_harness.contract_framing import SETTINGS
from spar_harness.contract_presenter import contract_presenter
from spar_harness.contract_task import contract_dynamics
from spar_harness.referee import InvalidMove, run_episode
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

CUE_WORDS = re.compile(
    r"\b(honou?r\w*|reneg\w*|sacrific\w*|betray\w*|duty)\b", re.IGNORECASE
)


def params(**overrides):
    values = {
        "benefits": {"alpha": 18.0, "beta": 18.0},
        "shock_prob": 0.5,
        "loss": 60.0,
        "disaster": 40.0,
        "penalty": 0.0,
        "transfer_cap": 60.0,
    }
    return ContractParams(**(values | overrides))


def model_facing_text(setting, imposed_venture=False, penalty=0.0):
    """Everything a model can read: system prompt, briefing, tool schema, tasks."""
    game_params = params(imposed_venture=imposed_venture, penalty=penalty)
    presenter = contract_presenter(game_params, PresentationTreatment(setting=setting))
    context = PhaseContext(
        round=1,
        role="proposer",
        action_kind="offer",
        allowed_actions={"propose", "walk_away"},
        talk_fields={"message", "declared_action"},
    )
    briefing = ContractBriefing(params=game_params, max_rounds=6, persuasion_rounds=2)
    first = Observation(recipient="alpha", round=1, briefing=briefing)
    shocked = Observation(
        recipient="alpha",
        round=1,
        contract=ContractState(
            alpha_bears_prob=0.5, transfer=0.0, shock=True, bearer="alpha"
        ),
    )
    decision = PhaseContext(
        round=1, role="bearer", action_kind="decision", allowed_actions=set()
    )
    tool = presenter.tool(first, context)
    return "\n".join(
        [
            presenter.system_prompt,
            presenter.render("alpha", first, context),
            presenter.render("alpha", shocked, decision),
            tool.model_dump_json(),
        ]
    )


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("imposed_venture", [False, True])
def test_no_setting_shows_a_cue_word(setting, imposed_venture):
    text = model_facing_text(setting, imposed_venture, penalty=5.0)
    assert CUE_WORDS.findall(text) == []
    labels = SETTINGS[setting]
    assert labels.bear in text and labels.decline in text


@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("imposed_venture", [False, True])
def test_the_side_that_takes_the_cost_has_its_settings_name_not_bearer(
    setting, imposed_venture
):
    text = model_facing_text(setting, imposed_venture, penalty=5.0)
    assert "bearer" not in text.lower().replace("probability_you_bear", "")
    assert SETTINGS[setting].taker in text


def test_settings_change_the_story_but_not_the_numbers():
    numbers = {
        setting: re.findall(r"\d+(?:\.\d+)?", model_facing_text(setting))
        for setting in SETTINGS
    }
    assert len({tuple(found) for found in numbers.values()}) == 1


def test_labels_map_to_internal_actions_and_internal_names_are_rejected():
    presenter = contract_presenter(params(), PresentationTreatment(setting="cargo"))
    assert presenter.parse("alpha", {"action": "jettison"}).action.kind == "honor"
    assert presenter.parse("alpha", {"action": "keep_cargo"}).action.kind == "renege"
    with pytest.raises(InvalidMove):
        presenter.parse("alpha", {"action": "honor"})
    move = presenter.parse("alpha", {"action": "accept", "promise": "jettison"})
    assert move.declared_action.kind == "honor"


def test_imposed_venture_briefing_states_the_shock_is_certain():
    text = model_facing_text("compute", imposed_venture=True)
    assert "will occur" in text
    assert "occurs with probability" not in text
    assert "No contract" not in text


def test_prompt_arms_change_only_the_framing_sentence():
    base = contract_presenter(params()).system_prompt
    future = contract_presenter(
        params(), PresentationTreatment(prompt_arm="future_dealings")
    ).system_prompt
    assert "one-shot" in base and "one-shot" not in future
    assert "deal with the other side again" in future


def run_bots(seed, **overrides):
    config = EpisodeConfig(
        episode_id="test",
        game=contract_dynamics_game(
            params(**overrides), max_rounds=2, persuasion_rounds=0
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        seed=seed,
    )
    players = {
        "alpha": scripted_contract_bot("honor"),
        "beta": scripted_contract_bot("honor"),
    }
    return asyncio.run(run_episode(config, players))


def test_forced_shock_hits_every_signed_contract_with_the_same_bearer_draw():
    for seed in range(20):
        forced = run_bots(seed, shock_prob=0.0, force_shock=True)
        natural = run_bots(seed, shock_prob=1.0)
        assert forced.contract.shock is True
        assert forced.contract.bearer == natural.contract.bearer
    never = [run_bots(seed, shock_prob=0.0).ending for seed in range(20)]
    assert set(never) == {"no_shock"}


def test_a_framed_episode_runs_end_to_end_and_records_the_persuaders_belief():
    [log] = eval(
        contract_dynamics(
            n_episodes=2,
            a="model",
            b="always_honor",
            transfers="free",
            shock_prob=1.0,
            persuasion_rounds=2,
            setting="cargo",
            promises=True,
            messages=True,
        ),
        model_roles={"a": get_model("mockllm/model", custom_outputs=scripted_model)},
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    shifts = []
    for sample in log.samples:
        episode = sample.store["episode"]
        assert episode["ending"] == "honored"
        assert all(not record["defaulted"] for record in episode["rounds"])
        seats = sample.scores["contract_seat_scorer"].value
        if episode["contract"]["bearer"] != _model_position(sample):
            assert seats["a_persuader_belief_first"] == 0.5
            shifts.append(seats["a_persuader_belief_shift"])
    assert shifts and all(shift == 0.0 for shift in shifts)


def _model_position(sample):
    return next(p for p, seat in sample.metadata["seats"].items() if seat == "a")
