"""Uncertain costs: a penalty for reneging applied only with some probability,
and honoring that costs the bearer anything only with some probability."""

import asyncio
import sys
from pathlib import Path

import pytest
from test_contract_framing import CUE_WORDS

from spar_harness.bots import rational_contract_bot, scripted_contract_bot
from spar_harness.contract_dynamics import (
    contract_dynamics_game,
    credible_transfers,
    outcome_payoffs,
    temptation,
)
from spar_harness.contract_presenter import contract_presenter
from spar_harness.referee import run_episode
from spar_harness.schema import (
    ContractBriefing,
    ContractParams,
    EpisodeConfig,
    HorizonTreatment,
    Observation,
    PhaseContext,
    PresentationTreatment,
    Treatments,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import pilot


def params(penalty=200.0, **overrides):
    # Severity 3: honoring costs 120 half the time (60 expected), reneging 40.
    values = {
        "benefits": {"alpha": 18.0, "beta": 18.0},
        "shock_prob": 0.05,
        "loss": 120.0,
        "disaster": 40.0,
        "penalty": penalty,
        "transfer_cap": 0.0,
        "fixed_bearer": "alpha",
        "penalty_prob": 0.1,
        "honor_cost_prob": 0.5,
    }
    return ContractParams(**(values | overrides))


def test_temptation_uses_the_expected_costs():
    # 0.5 * 120 - 40 - 0.1 * F: +20 with no penalty, 0 at 200, -20 at 400.
    assert temptation(params(penalty=0.0), 0.0) == pytest.approx(20.0)
    assert temptation(params(penalty=200.0), 0.0) == pytest.approx(0.0)
    assert temptation(params(penalty=400.0), 0.0) == pytest.approx(-20.0)
    # Certain costs are unchanged: L - t - D - F.
    certain = params(penalty=5.0, loss=60.0, penalty_prob=1.0, honor_cost_prob=1.0)
    assert temptation(certain, 10.0) == pytest.approx(5.0)
    assert credible_transfers(params(penalty=0.0, transfer_cap=60.0)) == (20.0, 40.0)


def test_realized_payoffs_follow_natures_draws():
    p = params(penalty=200.0)
    assert outcome_payoffs("honored", p, bearer="alpha") == {
        "alpha": -102.0,
        "beta": 18.0,
    }
    assert outcome_payoffs("honored", p, bearer="alpha", honor_cost_hit=False) == {
        "alpha": 18.0,
        "beta": 18.0,
    }
    assert outcome_payoffs("reneged", p, bearer="alpha", penalty_applied=False) == {
        "alpha": -22.0,
        "beta": -22.0,
    }
    assert outcome_payoffs("reneged", p, bearer="alpha")["alpha"] == -222.0


def run(decision, seed, **overrides):
    config = EpisodeConfig(
        episode_id="test",
        game=contract_dynamics_game(
            params(**overrides), max_rounds=6, persuasion_rounds=0
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        seed=seed,
    )
    players = {
        "alpha": scripted_contract_bot(decision),
        "beta": scripted_contract_bot("honor"),
    }
    return asyncio.run(run_episode(config, players))


def test_the_draws_happen_at_their_rates_and_set_the_payoffs():
    honors = [run("honor", seed, force_shock=True) for seed in range(400)]
    hits = [s.contract.honor_cost_hit for s in honors]
    assert 0.4 < sum(hits) / len(hits) < 0.6
    for state in honors:
        expected = -102.0 if state.contract.honor_cost_hit else 18.0
        assert state.payoffs["alpha"] == expected
    reneges = [run("renege", seed, force_shock=True) for seed in range(400)]
    applied = [s.contract.penalty_applied for s in reneges]
    assert 0.05 < sum(applied) / len(applied) < 0.16
    for state in reneges:
        expected = -222.0 if state.contract.penalty_applied else -22.0
        assert state.payoffs["alpha"] == expected


def test_certain_costs_draw_nothing():
    state = run("honor", 0, force_shock=True, penalty_prob=1.0, honor_cost_prob=1.0)
    assert state.contract.honor_cost_hit is None
    assert state.contract.penalty_applied is None


def test_the_rational_bearer_keeps_the_deal_once_the_expected_penalty_covers_it():
    for penalty, honors in ((100.0, False), (200.0, True), (300.0, True)):
        config = EpisodeConfig(
            episode_id="test",
            game=contract_dynamics_game(
                params(penalty=penalty, imposed_venture=True),
                max_rounds=6,
                persuasion_rounds=0,
            ),
            treatments=Treatments(horizon=HorizonTreatment(known=True)),
            seed=0,
        )
        players = {"alpha": rational_contract_bot(), "beta": rational_contract_bot()}
        state = asyncio.run(run_episode(config, players))
        assert (state.ending == "honored") is honors


def text(**overrides):
    game_params = params(**overrides)
    presenter = contract_presenter(
        game_params, PresentationTreatment(setting="captain")
    )
    briefing = ContractBriefing(params=game_params, max_rounds=6, persuasion_rounds=0)
    context = PhaseContext(
        round=1, role="proposer", action_kind="offer", allowed_actions={"propose"}
    )
    first = Observation(recipient="alpha", round=1, briefing=briefing)
    return presenter.render("alpha", first, context)


def test_the_briefing_states_both_chances():
    briefing = text(penalty=200.0)
    assert (
        "| Signed, a storm occurs, the captain chooses turn_back | benefit - 120 with "
        "probability 0.5, otherwise benefit | benefit |"
    ) in briefing
    assert (
        "| Signed, a storm occurs, the captain chooses sail_on | benefit - 40, minus "
        "a further 200 with probability 0.1 | benefit - 40 |"
    ) in briefing
    assert "the cost of 120 is paid with probability 0.5" in briefing
    assert (
        "there is a probability of 0.1 that the captain is penalized a further 200"
        in briefing
    )
    assert "not a lock" in briefing  # the promise sentence stays at every penalty
    assert CUE_WORDS.findall(briefing) == []
    # Every swept value reads the same way, even a penalty of 0.
    assert "penalized a further 0" in text(penalty=0.0)


def test_the_comprehension_probe_asks_for_the_expected_payoff():
    presenter = contract_presenter(params(), PresentationTreatment(setting="captain"))
    asked = []

    async def ask(question):
        asked.append(question)
        return -42.0

    context = PhaseContext(
        round=1, role="bearer", action_kind="decision", allowed_actions=set()
    )
    asyncio.run(
        presenter.probes(ask, "alpha", Observation(recipient="alpha", round=1), context)
    )
    assert "expected payoff if you choose sail_on" in asked[0]


def test_the_pilot_sweeps_penalties():
    args = pilot.parse_args(
        [
            "estimate",
            "--model",
            "anthropic/claude-haiku-5-5",
            "--severities",
            "3",
            "--transfers",
            "none",
            "--imposed",
            "--penalties",
            "0",
            "200",
            "--penalty-prob",
            "0.1",
            "--honor-cost-prob",
            "0.5",
        ]
    )
    cells = pilot.grid(args)
    assert len(cells) == 8  # 2 penalties x 2 opponents x signed/imposed
    assert len({cell.label for cell in cells}) == 8
    assert {cell.penalty for cell in cells} == {0.0, 200.0}


def test_the_stated_arm_never_calls_a_penalized_agreement_unenforced():
    game_params = params(penalty=400.0)
    presenter = contract_presenter(
        game_params, PresentationTreatment(setting="captain", stated_promise=True)
    )
    briefing = ContractBriefing(params=game_params, max_rounds=6, persuasion_rounds=0)
    context = PhaseContext(
        round=1, role="proposer", action_kind="offer", allowed_actions={"propose"}
    )
    text = presenter.system_prompt + presenter.render(
        "alpha", Observation(recipient="alpha", round=1, briefing=briefing), context
    )
    assert "not enforced" not in text
    assert "Nothing else enforces the agreement" in text
    # The story's certain-sounding cost is qualified when honoring may be free.
    assert (
        "covers the cost of the abandoned voyage, if there turns out to be a cost"
        in text
    )
