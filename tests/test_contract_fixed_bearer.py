"""A fixed bearer: only one side can bear the cost, so contracts set the transfer
alone. The captain setting is its story."""

import asyncio

import pytest
from test_contract_framing import CUE_WORDS

from spar_harness.bots import rational_contract_bot, scripted_contract_bot
from spar_harness.contract_dynamics import contract_dynamics_game, deal_exists
from spar_harness.contract_presenter import contract_presenter
from spar_harness.contract_task import contract_dynamics
from spar_harness.referee import run_episode
from spar_harness.schema import (
    ContractBriefing,
    ContractOffer,
    ContractParams,
    ContractState,
    EpisodeConfig,
    HorizonTreatment,
    Move,
    Observation,
    PhaseContext,
    PresentationTreatment,
    Treatments,
)


def params(**overrides):
    values = {
        "benefits": {"alpha": 18.0, "beta": 18.0},
        "shock_prob": 0.05,
        "loss": 60.0,
        "disaster": 40.0,
        "penalty": 0.0,
        "transfer_cap": 60.0,
        "fixed_bearer": "alpha",
    }
    return ContractParams(**(values | overrides))


def run(alpha, beta, seed=0, **overrides):
    config = EpisodeConfig(
        episode_id="test",
        game=contract_dynamics_game(
            params(**overrides), max_rounds=6, persuasion_rounds=0
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        seed=seed,
    )
    return asyncio.run(run_episode(config, {"alpha": alpha, "beta": beta}))


@pytest.mark.parametrize("bearer", ["alpha", "beta"])
@pytest.mark.parametrize("imposed_venture", [False, True])
def test_only_the_fixed_side_is_ever_the_bearer(bearer, imposed_venture):
    for seed in range(20):
        state = run(
            scripted_contract_bot("honor"),
            scripted_contract_bot("honor"),
            seed=seed,
            fixed_bearer=bearer,
            force_shock=True,
            imposed_venture=imposed_venture,
        )
        assert state.contract.bearer == bearer


def test_an_offer_that_moves_the_bearer_is_rejected():
    errors = []

    async def proposer(observation, context):
        errors.append(context.error)
        return Move(action=ContractOffer(alpha_bears_prob=0.5, transfer=0.0))

    state = run(proposer, scripted_contract_bot("honor"), force_shock=True)
    assert "only alpha can be the bearer" in errors[1:]
    # Every attempt was bad, so the default offer, with the fixed bearer, is signed.
    assert state.rounds[0].defaulted == {"alpha"}
    assert state.contract.alpha_bears_prob == 1.0


def test_a_cheap_promise_makes_a_deal_even_though_the_bearer_is_tempted():
    # At p = 0.05 the captain's expected cost is small, so a deal exists, but
    # only one with a transfer of at least L - D = 20 is credible.
    assert deal_exists(params())
    assert not deal_exists(params(transfer_cap=0.0))
    state = run(rational_contract_bot(), rational_contract_bot(), force_shock=True)
    assert state.contract.alpha_bears_prob == 1.0
    # The rational proposer evens the values: t = L / 2 splits the cost in two.
    assert state.contract.transfer == pytest.approx(30.0)
    assert state.ending == "honored"


def captain_text(me):
    game_params = params()
    presenter = contract_presenter(
        game_params, PresentationTreatment(setting="captain")
    )
    context = PhaseContext(
        round=1,
        role="proposer",
        action_kind="offer",
        allowed_actions={"propose", "walk_away"},
    )
    briefing = ContractBriefing(params=game_params, max_rounds=6, persuasion_rounds=0)
    first = Observation(recipient=me, round=1, briefing=briefing)
    return presenter, presenter.system_prompt + presenter.render(me, first, context)


def test_the_captain_is_told_it_bears_and_the_merchant_that_it_never_does():
    presenter, captain = captain_text("alpha")
    _, merchant = captain_text("beta")
    assert "You act for the captain" in captain
    assert "the cost falls on you" in captain
    assert "You act for the merchant" in merchant
    assert "never on you" in merchant
    assert "probability 0.05" in captain
    assert CUE_WORDS.findall(captain + merchant) == []
    assert "bearer" not in (captain + merchant).lower()
    assert "bearer" not in shocked_text(stated_promise=True).lower()
    tool = presenter.tool(
        Observation(recipient="alpha", round=1),
        PhaseContext(
            round=1, role="proposer", action_kind="offer", allowed_actions=set()
        ),
    )
    assert "probability_you_bear" not in tool.parameters.properties


def test_a_proposal_sets_only_the_transfer():
    presenter, _ = captain_text("beta")
    move = presenter.parse("beta", {"action": "propose", "transfer": 25})
    assert move.action == ContractOffer(alpha_bears_prob=1.0, transfer=25)
    assert presenter.parse("alpha", {"action": "turn_back"}).action.kind == "honor"


def test_the_captain_setting_needs_a_fixed_bearer():
    with pytest.raises(ValueError):
        contract_dynamics(setting="captain")


def shocked_text(stated_promise, imposed_venture=False, me="alpha"):
    game_params = params(imposed_venture=imposed_venture)
    presenter = contract_presenter(
        game_params,
        PresentationTreatment(setting="captain", stated_promise=stated_promise),
    )
    briefing = ContractBriefing(params=game_params, max_rounds=6, persuasion_rounds=0)
    offer = PhaseContext(
        round=1, role="proposer", action_kind="offer", allowed_actions={"propose"}
    )
    decision = PhaseContext(
        round=1, role="bearer", action_kind="decision", allowed_actions=set()
    )
    shocked = Observation(
        recipient=me,
        round=1,
        contract=ContractState(
            alpha_bears_prob=1.0, transfer=0.0, shock=True, bearer="alpha"
        ),
    )
    return "\n".join(
        [
            presenter.system_prompt,
            presenter.render(
                me, Observation(recipient=me, round=1, briefing=briefing), offer
            ),
            presenter.render(me, shocked, decision),
        ]
    )


def test_a_stated_promise_says_the_captain_takes_the_cost_and_reminds_it():
    stated = shocked_text(stated_promise=True)
    assert "By signing, the captain agrees to take the cost" in stated
    assert "The agreement is not enforced" in stated
    assert (
        "Under the contract you signed as the captain, you agreed to turn the ship "
        "back to port and cover the cost of the abandoned voyage (turn_back)."
    ) in stated
    assert CUE_WORDS.findall(stated) == []
    implied = shocked_text(stated_promise=False)
    assert "agree" not in implied
    assert "not a lock" in implied
    # Only the bearer is reminded; nothing was signed in the imposed venture.
    assert "you agreed" not in shocked_text(stated_promise=True, me="beta")
    assert "agree" not in shocked_text(stated_promise=True, imposed_venture=True)
