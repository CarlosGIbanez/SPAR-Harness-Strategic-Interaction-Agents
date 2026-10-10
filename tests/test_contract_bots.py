import asyncio

from pytest import approx

from spar_harness.bots import rational_contract_bot, scripted_contract_bot
from spar_harness.contract_dynamics import contract_dynamics_game
from spar_harness.referee import run_episode
from spar_harness.schema import (
    ContractOffer,
    ContractParams,
    EpisodeConfig,
    HorizonTreatment,
    Move,
    Treatments,
)


def config(seed=0, imposed_venture=False, **overrides):
    values = {
        "benefits": {"alpha": 18.0, "beta": 18.0},
        "shock_prob": 0.5,
        "loss": 60.0,
        "disaster": 40.0,
        "penalty": 0.0,
        "transfer_cap": 0.0,
        "imposed_venture": imposed_venture,
    }
    return EpisodeConfig(
        episode_id="test",
        game=contract_dynamics_game(
            ContractParams(**(values | overrides)), max_rounds=6, persuasion_rounds=0
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        seed=seed,
    )


def run(config, alpha, beta):
    return asyncio.run(run_episode(config, {"alpha": alpha, "beta": beta}))


def test_rational_bots_do_not_sign_without_transfers():
    # The bearer would renege (L > D), so a contract is worth b - p * D < 0.
    state = run(config(), rational_contract_bot(), rational_contract_bot())
    assert state.ending == "walked_away"


def test_rational_bots_sign_the_cheapest_credible_contract_and_honor_it():
    endings = set()
    for seed in range(10):
        state = run(
            config(seed=seed, transfer_cap=60.0),
            rational_contract_bot(),
            rational_contract_bot(),
        )
        assert state.contract.transfer == approx(20.0)  # t = L - D
        assert state.contract.alpha_bears_prob == approx(
            0.5
        )  # equal rewards: fair coin
        endings.add(state.ending)
    assert endings == {"honored", "no_shock"}  # every shock is absorbed


def test_rational_bearer_honors_exactly_when_temptation_is_not_positive():
    # imposed venture: t = 0, so temptation = L - D - F
    tempted = run(
        config(imposed_venture=True), rational_contract_bot(), rational_contract_bot()
    )
    assert tempted.ending == "reneged"  # 60 - 40 - 0 = 20 > 0
    deterred = run(
        config(imposed_venture=True, penalty=20.0),
        rational_contract_bot(),
        rational_contract_bot(),
    )
    assert deterred.ending == "honored"  # 60 - 40 - 20 = 0


def test_rational_responder_signs_only_contracts_worth_more_than_nothing():
    def proposer(q, t):
        async def play(observation, context):
            return Move(action=ContractOffer(alpha_bears_prob=q, transfer=t))

        return play

    free = config(transfer_cap=60.0)  # no shock: only the decision matters
    # credible: 18 - 0.5 * (0.5 * 40 + 0.5 * 20) = 3 > 0
    assert run(free, proposer(0.5, 20.0), rational_contract_bot()).contract is not None
    # not credible (t = 0 tempts the bearer): 18 - 0.5 * 40 = -2
    assert (
        run(free, proposer(0.5, 0.0), rational_contract_bot()).ending == "walked_away"
    )


def test_the_side_that_gains_more_carries_more_risk():
    state = run(
        config(transfer_cap=60.0, benefits={"alpha": 30.0, "beta": 20.0}),
        rational_contract_bot(),
        rational_contract_bot(),
    )
    assert state.contract.transfer == approx(20.0)
    assert state.contract.alpha_bears_prob == approx(1.0)  # both then gain 10


def test_scripted_bots_sign_anything_and_always_make_their_decision():
    honorer = run(
        config(shock_prob=1.0, imposed_venture=True),
        scripted_contract_bot("honor"),
        scripted_contract_bot("honor"),
    )
    assert honorer.ending == "honored"
    reneger = run(
        config(shock_prob=1.0, transfer_cap=60.0),
        scripted_contract_bot("renege"),  # proposes the fair coin without transfer
        scripted_contract_bot("renege"),  # and signs it
    )
    assert (reneger.contract.alpha_bears_prob, reneger.contract.transfer) == (0.5, 0.0)
    assert reneger.ending == "reneged"


def test_a_scripted_other_sends_its_appeal_every_persuasion_round():
    appeal = "We signed this together; please honor it."
    game = config(shock_prob=1.0, imposed_venture=True).model_copy(deep=True)
    game.game.stages[0].termination.max_rounds = 2  # two persuasion rounds
    state = run(
        game,
        scripted_contract_bot("honor", appeal),
        scripted_contract_bot("honor", appeal),
    )
    other = "beta" if state.contract.bearer == "alpha" else "alpha"
    pleas = [r.moves[other].message for r in state.rounds if r.stage == "persuasion"]
    assert pleas == [appeal, appeal]
