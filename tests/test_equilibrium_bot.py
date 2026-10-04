import asyncio

from pytest import approx

from spar_harness.bots import equilibrium_bot
from spar_harness.crisis_bargaining import (
    bargaining_range,
    crisis_bargaining_game,
    win_probability,
)
from spar_harness.referee import run_episode
from spar_harness.schema import (
    EpisodeConfig,
    HorizonTreatment,
    PartialType,
    PrivateInfoTreatment,
    TalkTreatment,
    Treatments,
    TypePrior,
    UniformPrior,
)


def fixed(value):
    return UniformPrior(low=value, high=value)


def complete_info_config(alpha, beta, status_quo=0.5):
    """alpha and beta are (capability, cost_of_conflict) pairs."""
    priors = {
        player: TypePrior(capability=fixed(k), cost_of_conflict=fixed(c))
        for player, (k, c) in {"alpha": alpha, "beta": beta}.items()
    }
    return EpisodeConfig(
        episode_id="test",
        game=crisis_bargaining_game(
            max_rounds=20, stop_prob=0.1, status_quo=status_quo
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        priors=priors,
        seed=0,
    )


def play(config):
    players = {"alpha": equilibrium_bot(), "beta": equilibrium_bot()}
    return asyncio.run(run_episode(config, players))


def test_equilibrium_bots_agree_in_the_first_round_under_complete_information():
    cases = [
        ((3.0, 0.1), (1.0, 0.2)),
        ((1.0, 0.05), (1.0, 0.05)),
        ((1.0, 0.3), (9.0, 0.01)),
        ((5.0, 0.0), (2.0, 0.0)),  # costless conflict: the range is a single point
    ]
    for alpha, beta in cases:
        state = play(complete_info_config(alpha, beta))
        assert state.ending == "agreement", (alpha, beta)
        assert len(state.rounds) == 1
        split = state.rounds[0].moves["alpha"].action.split
        low, high = bargaining_range(
            win_probability(alpha[0], beta[0]), alpha[1], beta[1]
        )
        assert low - 1e-9 <= split <= high + 1e-9, (alpha, beta, split)


def test_proposer_trades_off_a_better_deal_against_the_risk_of_war():
    # Equal capabilities, so p = 0.5. Beta's cost is private, c_beta ~ U[0, 0.4].
    # Beta accepts share s iff s >= 0.5 - c_beta, so P(accept) = (s - 0.1) / 0.4.
    # Alpha's expected payoff, with u = s - 0.1, is 0.4 + 1.25u - 2.5u^2,
    # maximized at u = 0.25: offer beta 0.35 and keep 0.65.
    config = EpisodeConfig(
        episode_id="test",
        game=crisis_bargaining_game(max_rounds=20, stop_prob=0.1, status_quo=0.5),
        treatments=Treatments(
            horizon=HorizonTreatment(known=True),
            private_info=PrivateInfoTreatment(private_fields={"cost_of_conflict"}),
        ),
        priors={
            "alpha": TypePrior(capability=fixed(1.0), cost_of_conflict=fixed(0.1)),
            "beta": TypePrior(
                capability=fixed(1.0), cost_of_conflict=UniformPrior(low=0.0, high=0.4)
            ),
        },
        seed=0,
    )
    state = play(config)
    assert state.rounds[0].moves["alpha"].action.split == approx(0.65, abs=0.005)


def test_bot_truthfully_declares_its_private_fields_and_sends_no_message():
    config = complete_info_config((3.0, 0.1), (1.0, 0.2)).model_copy(
        update={
            "treatments": Treatments(
                horizon=HorizonTreatment(known=True),
                private_info=PrivateInfoTreatment(private_fields={"capability"}),
                talk=TalkTreatment(
                    messages=True, declared_action=False, declared_type=True
                ),
            )
        }
    )
    state = play(config)
    alpha_move = state.rounds[0].moves["alpha"]
    assert alpha_move.declared_type == PartialType(capability=3.0)
    assert alpha_move.message is None
