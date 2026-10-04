from spar_harness.priors import draw_types
from spar_harness.schema import TypePrior, UniformPrior


def symmetric_priors():
    prior = TypePrior(
        capability=UniformPrior(low=1.0, high=10.0),
        cost_of_conflict=UniformPrior(low=0.05, high=0.3),
    )
    return {"alpha": prior, "beta": prior}


def test_same_seed_draws_the_same_types():
    assert draw_types(symmetric_priors(), seed=7) == draw_types(
        symmetric_priors(), seed=7
    )


def test_draws_do_not_depend_on_the_order_players_are_listed():
    alpha_first = symmetric_priors()
    beta_first = {"beta": alpha_first["beta"], "alpha": alpha_first["alpha"]}
    assert draw_types(alpha_first, seed=7) == draw_types(beta_first, seed=7)


def test_equal_bounds_fix_the_value():
    priors = {
        "alpha": TypePrior(
            capability=UniformPrior(low=3.0, high=3.0),
            cost_of_conflict=UniformPrior(low=0.1, high=0.1),
        ),
        "beta": TypePrior(
            capability=UniformPrior(low=1.0, high=1.0),
            cost_of_conflict=UniformPrior(low=0.2, high=0.2),
        ),
    }
    types = draw_types(priors, seed=7)
    assert (types["alpha"].capability, types["alpha"].cost_of_conflict) == (3.0, 0.1)
    assert (types["beta"].capability, types["beta"].cost_of_conflict) == (1.0, 0.2)
