"""Hand-computed values from the scenario spec: p = 0.5, D = 40, L = 60."""

from pytest import approx

from spar_harness.contract_dynamics import (
    contract_value,
    credible_transfers,
    deal_exists,
    equal_gains_share,
    outcome_payoffs,
    temptation,
)
from spar_harness.schema import ContractParams


def params(benefit_alpha=28.0, benefit_beta=10.0, penalty=0.0, transfer_cap=0.0):
    return ContractParams(
        benefits={"alpha": benefit_alpha, "beta": benefit_beta},
        shock_prob=0.5,
        loss=60.0,
        disaster=40.0,
        penalty=penalty,
        transfer_cap=transfer_cap,
    )


def test_temptation_is_what_the_bearer_saves_by_reneging():
    assert temptation(params(), transfer=0.0) == 20.0  # L - D: renege
    assert temptation(params(), transfer=20.0) == 0.0  # exactly indifferent: honor
    assert temptation(params(penalty=25.0), transfer=0.0) == -5.0  # the penalty deters


def test_payoffs_follow_the_outcome_table():
    p = params(penalty=3.0)
    for no_contract in ("walked_away", "horizon"):
        assert outcome_payoffs(no_contract, p) == {"alpha": 0.0, "beta": 0.0}
    assert outcome_payoffs("no_shock", p) == {"alpha": 28.0, "beta": 10.0}
    # alpha bears, transfer 5: bearer b - L + t, other b - t
    assert outcome_payoffs("honored", p, bearer="alpha", transfer=5.0) == {
        "alpha": 28.0 - 60.0 + 5.0,
        "beta": 10.0 - 5.0,
    }
    # beta bears and reneges: bearer b - D - F, other b - D; the transfer is never paid
    assert outcome_payoffs("reneged", p, bearer="beta", transfer=5.0) == {
        "alpha": 28.0 - 40.0,
        "beta": 10.0 - 40.0 - 3.0,
    }


def test_contract_value_from_the_spec_example():
    fair = {"alpha_bears_prob": 0.5, "transfer": 0.0}
    # honored: b_i - p * q_i * L
    assert contract_value("alpha", params(), **fair, bearer_honors=True) == 13.0
    assert contract_value("beta", params(), **fair, bearer_honors=True) == -5.0
    # broken: b_i - p * D
    assert contract_value("alpha", params(), **fair, bearer_honors=False) == 8.0
    assert contract_value("beta", params(), **fair, bearer_honors=False) == -10.0


def test_contract_value_counts_transfers_and_the_bearers_penalty():
    terms = {"alpha_bears_prob": 0.8, "transfer": 10.0}
    # honored: 28 - 0.5 * (0.8 * (60 - 10) + 0.2 * 10)
    assert contract_value("alpha", params(), **terms, bearer_honors=True) == approx(7.0)
    # broken, F = 3: 28 - 0.5 * (40 + 0.8 * 3)
    assert contract_value(
        "alpha", params(penalty=3.0), **terms, bearer_honors=False
    ) == approx(6.8)


def test_equal_gains_puts_more_risk_on_the_side_that_gains_more():
    assert equal_gains_share(params(), transfer=0.0) == approx(
        0.8
    )  # the spec's example
    q = equal_gains_share(params(), transfer=10.0)
    assert q == approx(0.95)
    gains = [
        contract_value(side, params(), q, 10.0, bearer_honors=True)
        for side in ("alpha", "beta")
    ]
    assert gains == [approx(4.0), approx(4.0)]


def test_equal_gains_share_stays_a_probability():
    assert equal_gains_share(params(benefit_alpha=50.0), transfer=0.0) == 1.0
    # with t = L / 2 who bears no longer matters, so any q gives the same gains
    assert equal_gains_share(params(), transfer=30.0) == 0.5


def test_credible_transfers_lie_in_l_minus_d_to_d_within_the_cap():
    assert credible_transfers(params(transfer_cap=0.0)) is None  # no transfers: renege
    assert credible_transfers(params(transfer_cap=10.0)) is None  # cap below L - D
    assert credible_transfers(params(transfer_cap=60.0)) == (20.0, 40.0)  # free
    assert credible_transfers(params(transfer_cap=30.0)) == (20.0, 30.0)
    assert credible_transfers(params(penalty=25.0)) == (
        0.0,
        0.0,
    )  # the penalty alone deters


def test_a_deal_exists_when_a_credible_contract_beats_no_contract_for_both():
    assert deal_exists(params(benefit_alpha=18.0, benefit_beta=18.0, transfer_cap=60.0))
    assert not deal_exists(params(transfer_cap=0.0))  # nothing credible
    assert not deal_exists(
        params(benefit_alpha=15.0, benefit_beta=10.0, transfer_cap=60.0)
    )


def test_no_deal_when_the_needed_transfer_outweighs_the_poorer_sides_benefit():
    # Credible contracts need t >= 20, so beta always expects to lose about p * t >= 10
    # but only gains b_beta = 1, even though the total 41 exceeds p * L = 30.
    assert not deal_exists(
        params(benefit_alpha=40.0, benefit_beta=1.0, transfer_cap=60.0)
    )


def test_in_the_unbalanced_example_no_credible_contract_strictly_helps_beta():
    # With free transfers credibility needs t >= 20; even if alpha always bears,
    # beta then expects to pay p * t >= 10 = b_beta. Beta can only gain by trusting.
    assert not deal_exists(params(transfer_cap=60.0))
    assert contract_value("beta", params(), 1.0, 20.0, bearer_honors=True) == 0.0


def test_equal_gains_share_without_risk_is_the_fair_coin():
    no_risk = params().model_copy(update={"shock_prob": 0.0})
    assert equal_gains_share(no_risk, transfer=0.0) == 0.5
