"""Engine features that no real game uses yet, played with a toy game.

Each round both players of the toy game offer a split at once. It is
registered in the referee's plugin dicts only for the tests that ask for the
`toy` fixture.
"""

import asyncio

import pytest

from spar_harness import referee
from spar_harness.referee import InvalidMove, run_episode
from spar_harness.schema import (
    EpisodeConfig,
    GameSpec,
    HorizonTreatment,
    Move,
    Offer,
    PhaseSpec,
    StageSpec,
    TerminationSpec,
    Treatments,
)

TOY = "toy"
BOTH_OFFER = PhaseSpec(kind="act", actors="both", action_kind="offer")
DEFAULT = Move(action=Offer(split=0.5))


@pytest.fixture
def toy(monkeypatch):
    """Registers the toy game for one test and returns a config factory."""
    plugins = {
        "ALLOWED_ACTIONS": {"offer": {"offer"}},
        "BRIEFINGS": lambda config, state, player: None,
        "ENDING_RULES": lambda move: None,
        "PAYOFF_MODELS": lambda game, state: {"alpha": 0.0, "beta": 0.0},
        "DEFAULT_MOVES": lambda game, state, player, context: DEFAULT,
    }
    for table, plugin in plugins.items():
        monkeypatch.setitem(getattr(referee, table), TOY, plugin)

    def config(phases=(BOTH_OFFER,), max_rounds=3, seed=0, treatments=None):
        stage = StageSpec(
            name="toy",
            phases=list(phases),
            termination=TerminationSpec(max_rounds=max_rounds, stop_prob=0.0),
            when_exhausted="horizon",
        )
        # GameId lists only the real games, so the toy game skips validation.
        game = GameSpec.model_construct(game_id=TOY, stages=[stage], status_quo=0.0)
        return EpisodeConfig(
            episode_id="toy",
            game=game,
            treatments=treatments or Treatments(horizon=HorizonTreatment(known=True)),
            seed=seed,
        )

    return config


def run(config, alpha, beta):
    return asyncio.run(run_episode(config, {"alpha": alpha, "beta": beta}))


def offering(split, log=None):
    """Offers the same split every round and records what it was shown."""

    async def play(observation, context):
        if log is not None:
            log.append((observation, context))
        return Move(action=Offer(split=split))

    return play


def broken(calls):
    """A player that can never produce a valid move."""

    async def play(observation, context):
        calls.append(context)
        raise InvalidMove("no tool call")

    return play


def test_simultaneous_movers_see_each_others_move_only_in_the_next_round(toy):
    alpha_log, beta_log = [], []
    state = run(
        toy(phases=[BOTH_OFFER], max_rounds=2),
        alpha=offering(0.3, alpha_log),
        beta=offering(0.8, beta_log),
    )
    assert alpha_log[0][0].opponent_moves == []
    assert beta_log[0][0].opponent_moves == []
    assert alpha_log[1][0].opponent_moves == [Move(action=Offer(split=0.8))]
    assert beta_log[1][0].opponent_moves == [Move(action=Offer(split=0.3))]
    assert state.rounds[0].moves == {
        "alpha": Move(action=Offer(split=0.3)),
        "beta": Move(action=Offer(split=0.8)),
    }
    assert alpha_log[0][1].role == beta_log[0][1].role == "both"


def test_simultaneous_movers_are_asked_at_the_same_time(toy):
    beta_asked = asyncio.Event()

    async def alpha(observation, context):
        # asked one after the other, alpha would wait for beta forever
        await asyncio.wait_for(beta_asked.wait(), timeout=1)
        return Move(action=Offer(split=0.3))

    async def beta(observation, context):
        beta_asked.set()
        return Move(action=Offer(split=0.8))

    state = run(toy(phases=[BOTH_OFFER], max_rounds=1), alpha=alpha, beta=beta)
    assert set(state.rounds[0].moves) == {"alpha", "beta"}


def test_a_simultaneous_mover_that_fails_gets_the_default_without_blocking_the_other(
    toy,
):
    calls = []
    state = run(
        toy(phases=[BOTH_OFFER], max_rounds=1), alpha=broken(calls), beta=offering(0.8)
    )
    assert len(calls) == 3
    assert state.rounds[0].moves == {
        "alpha": DEFAULT,
        "beta": Move(action=Offer(split=0.8)),
    }
    assert state.rounds[0].defaulted == {"alpha"}


def test_the_ending_rule_runs_once_both_simultaneous_moves_are_in(toy, monkeypatch):
    def full_split_ends(move):
        return "fight" if move.action.split == 1.0 else None

    monkeypatch.setitem(referee.ENDING_RULES, TOY, full_split_ends)
    state = run(toy(phases=[BOTH_OFFER]), alpha=offering(1.0), beta=offering(0.2))
    assert state.ending == "fight"
    assert len(state.rounds) == 1
    assert state.rounds[0].moves["beta"] == Move(action=Offer(split=0.2))
