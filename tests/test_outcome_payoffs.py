from pytest import approx

from spar_harness.crisis_bargaining import outcome_payoffs
from spar_harness.schema import PrivateType

TYPES = {
    "alpha": PrivateType(capability=3.0, cost_of_conflict=0.1),
    "beta": PrivateType(capability=1.0, cost_of_conflict=0.2),
}


def test_agreement_gives_each_player_their_share_of_the_split():
    payoffs = outcome_payoffs("agreement", TYPES, status_quo=0.5, split=0.7)
    assert payoffs == {"alpha": approx(0.7), "beta": approx(0.3)}


def test_fight_gives_each_player_their_reservation_value():
    payoffs = outcome_payoffs("fight", TYPES, status_quo=0.5)
    assert payoffs == {"alpha": approx(0.65), "beta": approx(0.05)}


def test_ending_without_a_deal_pays_the_status_quo():
    for ending in ("horizon", "random_stop"):
        payoffs = outcome_payoffs(ending, TYPES, status_quo=0.4)
        assert payoffs == {"alpha": approx(0.4), "beta": approx(0.6)}
