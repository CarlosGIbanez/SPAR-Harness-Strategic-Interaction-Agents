from pytest import approx

from spar_harness.crisis_bargaining import (
    bargaining_range,
    reservation_value,
    win_probability,
)


def test_win_probability_is_alphas_share_of_total_capability():
    assert win_probability(capability_alpha=3.0, capability_beta=1.0) == 0.75


def test_reservation_value_is_own_expected_share_from_fighting():
    assert reservation_value(
        "alpha", win_probability=0.75, cost_of_conflict=0.1
    ) == approx(0.65)
    assert reservation_value(
        "beta", win_probability=0.75, cost_of_conflict=0.2
    ) == approx(0.05)


def test_bargaining_range_is_alpha_shares_both_prefer_to_fighting():
    low, high = bargaining_range(win_probability=0.75, cost_alpha=0.1, cost_beta=0.2)
    assert (low, high) == (approx(0.65), approx(0.95))


def test_bargaining_range_never_extends_beyond_the_stake():
    low, high = bargaining_range(win_probability=0.95, cost_alpha=0.1, cost_beta=0.2)
    assert (low, high) == (approx(0.85), 1.0)
