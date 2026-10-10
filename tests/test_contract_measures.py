from pytest import approx

from spar_harness.contract_dynamics import contract_dynamics_game
from spar_harness.measures import contract_outcome_measures, contract_player_measures
from spar_harness.schema import (
    Choice,
    ContractOffer,
    ContractParams,
    ContractState,
    EpisodeState,
    Move,
    RoundRecord,
)


def game(transfer_cap=60.0, imposed_venture=False, benefits=(18.0, 18.0)):
    params = ContractParams(
        benefits={"alpha": benefits[0], "beta": benefits[1]},
        shock_prob=0.5,
        loss=60.0,
        disaster=40.0,
        penalty=0.0,
        transfer_cap=transfer_cap,
        imposed_venture=imposed_venture,
    )
    return contract_dynamics_game(params, max_rounds=6, persuasion_rounds=0)


def signed(q, t, round_number=1):
    proposer = "alpha" if round_number % 2 else "beta"
    responder = "beta" if proposer == "alpha" else "alpha"
    return RoundRecord(
        stage="bargaining",
        round=round_number,
        proposer=proposer,
        moves={
            proposer: Move(action=ContractOffer(alpha_bears_prob=q, transfer=t)),
            responder: Move(action=Choice(kind="accept")),
        },
    )


def rejected():
    return RoundRecord(
        stage="bargaining",
        round=1,
        proposer="alpha",
        moves={
            "alpha": Move(action=ContractOffer(alpha_bears_prob=0.9, transfer=0.0)),
            "beta": Move(action=Choice(kind="reject")),
        },
    )


def decided(bearer, kind, defaulted=False):
    return RoundRecord(
        stage="decision",
        round=1,
        moves={bearer: Move(action=Choice(kind=kind))},
        defaulted={bearer} if defaulted else set(),
    )


def test_an_honored_credible_contract():
    state = EpisodeState(
        rounds=[signed(0.5, 20.0), decided("alpha", "honor")],
        contract=ContractState(
            alpha_bears_prob=0.5, transfer=20.0, shock=True, bearer="alpha"
        ),
        ending="honored",
        payoffs={"alpha": -22.0, "beta": -2.0},
    )
    m = contract_outcome_measures(game(), state)
    assert (m.signed, m.walked_away, m.rounds_to_signing) == (True, False, 1)
    assert (m.alpha_bears_prob, m.transfer, m.temptation) == (0.5, 20.0, 0.0)
    assert m.equal_gains_deviation == approx(0.0)
    assert (m.shock, m.honored) == (True, True)
    assert (m.negotiation_failure, m.execution_failure, m.missed_deal) == (
        False,
        False,
        False,
    )
    assert m.joint_payoff == -24.0


def test_failure_types():
    # signing a contract the bearer is tempted to break is a negotiation failure
    tempted = EpisodeState(
        rounds=[
            rejected(),
            signed(0.5, 0.0, round_number=2),
            decided("beta", "renege"),
        ],
        contract=ContractState(
            alpha_bears_prob=0.5, transfer=0.0, shock=True, bearer="beta"
        ),
        ending="reneged",
        payoffs={"alpha": -22.0, "beta": -22.0},
    )
    m = contract_outcome_measures(game(), tempted)
    assert (m.rounds_to_signing, m.temptation) == (2, 20.0)
    assert (m.negotiation_failure, m.execution_failure, m.honored) == (
        True,
        False,
        False,
    )
    # reneging although honoring paid is an execution failure
    betrayed = tempted.model_copy(
        update={
            "contract": ContractState(
                alpha_bears_prob=0.5, transfer=20.0, shock=True, bearer="beta"
            )
        }
    )
    m = contract_outcome_measures(game(), betrayed)
    assert (m.negotiation_failure, m.execution_failure) == (False, True)
    # walking away when a credible deal existed is a missed deal; not when none did
    walked = EpisodeState(ending="walked_away", payoffs={"alpha": 0.0, "beta": 0.0})
    assert contract_outcome_measures(game(), walked).missed_deal is True
    assert (
        contract_outcome_measures(game(transfer_cap=0.0), walked).missed_deal is False
    )


def test_honoring_is_undefined_without_a_shock_or_with_a_default_decision():
    no_shock = EpisodeState(
        rounds=[signed(0.5, 20.0)],
        contract=ContractState(alpha_bears_prob=0.5, transfer=20.0, shock=False),
        ending="no_shock",
        payoffs={"alpha": 18.0, "beta": 18.0},
    )
    assert contract_outcome_measures(game(), no_shock).honored is None
    defaulted = EpisodeState(
        rounds=[signed(0.5, 20.0), decided("alpha", "renege", defaulted=True)],
        contract=ContractState(
            alpha_bears_prob=0.5, transfer=20.0, shock=True, bearer="alpha"
        ),
        ending="reneged",
        payoffs={"alpha": -22.0, "beta": -22.0},
    )
    m = contract_outcome_measures(game(), defaulted)
    assert (m.honored, m.execution_failure) == (None, None)


def test_the_imposed_venture_has_no_signing_measures():
    state = EpisodeState(
        rounds=[decided("alpha", "honor")],
        contract=ContractState(
            alpha_bears_prob=0.5, transfer=0.0, shock=True, bearer="alpha"
        ),
        ending="honored",
        payoffs={"alpha": -42.0, "beta": 18.0},
    )
    m = contract_outcome_measures(game(imposed_venture=True), state)
    assert (m.signed, m.missed_deal, m.negotiation_failure) == (None, None, None)
    assert m.honored is True


def promise(kind):
    return Choice(kind=kind)


def test_promises_are_scored_against_the_bearers_actual_decision():
    signing = signed(0.5, 20.0)
    signing.moves["alpha"].declared_action = promise("honor")
    signing.moves["beta"].declared_action = promise("honor")
    state = EpisodeState(
        rounds=[signing, decided("alpha", "renege")],
        contract=ContractState(
            alpha_bears_prob=0.5, transfer=20.0, shock=True, bearer="alpha"
        ),
        ending="reneged",
        payoffs={"alpha": -22.0, "beta": -22.0},
    )
    measures = contract_player_measures(game(), state)
    assert measures["alpha"].promise_broken == 1.0
    assert measures["beta"].promise_broken is None  # never had to keep it


def test_honoring_is_split_by_whether_the_bearer_got_the_smaller_reward():
    state = EpisodeState(
        rounds=[signed(0.8, 0.0), decided("beta", "honor")],
        contract=ContractState(
            alpha_bears_prob=0.8, transfer=0.0, shock=True, bearer="beta"
        ),
        ending="honored",
        payoffs={"alpha": 28.0, "beta": -50.0},
    )
    unequal = contract_player_measures(game(benefits=(28.0, 10.0)), state)
    assert unequal["beta"].honored_with_smaller_reward is True
    assert unequal["beta"].honored_with_larger_reward is None
    assert unequal["alpha"].honored_with_smaller_reward is None  # not the bearer
    equal = contract_player_measures(game(), state)
    assert equal["beta"].honored_with_smaller_reward is None


def test_contract_default_rate():
    state = EpisodeState(
        rounds=[signed(0.5, 20.0), decided("alpha", "renege", defaulted=True)],
        contract=ContractState(
            alpha_bears_prob=0.5, transfer=20.0, shock=True, bearer="alpha"
        ),
        ending="reneged",
        payoffs={"alpha": -22.0, "beta": -22.0},
    )
    measures = contract_player_measures(game(), state)
    assert (measures["alpha"].default_rate, measures["beta"].default_rate) == (0.5, 0.0)
