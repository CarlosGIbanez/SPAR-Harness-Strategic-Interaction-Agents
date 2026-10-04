import asyncio

from spar_harness.bots import random_bot
from spar_harness.crisis_bargaining import crisis_bargaining_game
from spar_harness.referee import run_episode
from spar_harness.schema import (
    EpisodeConfig,
    HorizonTreatment,
    Treatments,
    TypePrior,
    UniformPrior,
)


def config():
    prior = TypePrior(
        capability=UniformPrior(low=1.0, high=10.0),
        cost_of_conflict=UniformPrior(low=0.05, high=0.3),
    )
    return EpisodeConfig(
        episode_id="test",
        game=crisis_bargaining_game(max_rounds=20, stop_prob=0.0, status_quo=0.5),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        priors={"alpha": prior, "beta": prior},
        seed=0,
    )


def play(seed):
    players = {"alpha": random_bot(seed), "beta": random_bot(seed + 1)}
    return asyncio.run(run_episode(config(), players))


def test_random_bot_only_makes_valid_moves():
    for seed in range(50):
        assert all(not record.defaulted for record in play(seed).rounds)


def test_random_bot_is_reproducible_from_its_seed():
    assert play(3) == play(3)
