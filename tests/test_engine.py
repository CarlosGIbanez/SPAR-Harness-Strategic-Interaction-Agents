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
    EpisodeState,
    GameSpec,
    HorizonTreatment,
    Move,
    NoiseTreatment,
    Offer,
    PhaseSpec,
    RoundEnd,
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


def secret_reports(game, state, record):
    """Tells each player the round and its own secret, and nothing else."""
    return RoundEnd(
        reports={
            player: {"round": record.round, "secret": player.upper()}
            for player in ("alpha", "beta")
        }
    )


def test_each_report_reaches_only_its_player_before_its_next_move(toy, monkeypatch):
    monkeypatch.setitem(referee.AFTER_ROUND, TOY, secret_reports)
    alpha_log, beta_log = [], []
    state = run(
        toy(max_rounds=2), alpha=offering(0.3, alpha_log), beta=offering(0.8, beta_log)
    )
    assert alpha_log[0][0].reports == [] and beta_log[0][0].reports == []
    assert alpha_log[1][0].reports == [{"round": 1, "secret": "ALPHA"}]
    assert beta_log[1][0].reports == [{"round": 1, "secret": "BETA"}]
    assert state.rounds[0].reports == {
        "alpha": {"round": 1, "secret": "ALPHA"},
        "beta": {"round": 1, "secret": "BETA"},
    }


def test_reports_wait_for_a_player_who_does_not_move_every_round(toy, monkeypatch):
    monkeypatch.setitem(referee.AFTER_ROUND, TOY, secret_reports)
    proposer = PhaseSpec(kind="act", actors="proposer", action_kind="offer")
    alpha_log = []
    run(
        toy(phases=[proposer], max_rounds=3),
        alpha=offering(0.3, alpha_log),
        beta=offering(0.8),
    )
    # the proposer alternates: alpha moves in rounds 1 and 3, so in round 3 it
    # hears about rounds 1 and 2
    assert [observation.reports for observation, _ in alpha_log] == [
        [],
        [{"round": 1, "secret": "ALPHA"}, {"round": 2, "secret": "ALPHA"}],
    ]


def test_without_a_round_end_rule_players_get_no_reports(toy):
    alpha_log = []
    state = run(toy(max_rounds=2), alpha=offering(0.3, alpha_log), beta=offering(0.8))
    assert [observation.reports for observation, _ in alpha_log] == [[], []]
    assert state.rounds[0].reports == {}


def test_a_round_end_rule_can_end_the_episode_on_the_game_state(toy, monkeypatch):
    def ends_when_offers_add_up(game, state, record):
        total = sum(
            move.action.split for done in state.rounds for move in done.moves.values()
        )
        return RoundEnd(outcome="fight" if total >= 1.5 else None)

    monkeypatch.setitem(referee.AFTER_ROUND, TOY, ends_when_offers_add_up)
    state = run(toy(max_rounds=5), alpha=offering(0.4), beta=offering(0.4))
    assert state.ending == "fight"
    assert len(state.rounds) == 2  # 0.8 after round 1, 1.6 after round 2


def test_reports_survive_the_episode_record_round_trip(toy, monkeypatch):
    monkeypatch.setitem(referee.AFTER_ROUND, TOY, secret_reports)
    state = run(toy(max_rounds=1), alpha=offering(0.3), beta=offering(0.8))
    stored = EpisodeState.model_validate(state.model_dump(mode="json"))
    assert stored.rounds[0].reports == state.rounds[0].reports


FULL = Move(action=Offer(split=1.0))


def misfire_to_full(game, state, player, move, rng):
    """In the toy game an order that misfires becomes the full split."""
    return FULL


def noisy(toy, rate, **options):
    return toy(
        treatments=Treatments(
            horizon=HorizonTreatment(known=True), noise=NoiseTreatment(rate=rate)
        ),
        **options,
    )


def test_without_noise_every_move_is_executed_as_chosen(toy, monkeypatch):
    monkeypatch.setitem(referee.NOISE, TOY, misfire_to_full)
    for config in (toy(), noisy(toy, 0.0)):
        state = run(config, alpha=offering(0.3), beta=offering(0.8))
        assert all(record.executed == {} for record in state.rounds)


def test_at_rate_one_every_order_misfires_and_the_opponent_sees_the_result(
    toy, monkeypatch
):
    monkeypatch.setitem(referee.NOISE, TOY, misfire_to_full)
    beta_log = []
    state = run(
        noisy(toy, 1.0, max_rounds=2),
        alpha=offering(0.3),
        beta=offering(0.8, beta_log),
    )
    first = state.rounds[0]
    assert first.moves["alpha"] == Move(action=Offer(split=0.3))  # chosen, for intent
    assert first.executed == {"alpha": FULL, "beta": FULL}
    assert beta_log[1][0].opponent_moves == [FULL]
    stored = EpisodeState.model_validate(state.model_dump(mode="json"))
    assert stored.rounds[0].executed == first.executed


def test_both_episodes_of_a_role_swapped_pair_get_the_same_accidents(toy, monkeypatch):
    monkeypatch.setitem(referee.NOISE, TOY, misfire_to_full)
    config = noisy(toy, 0.5, max_rounds=10, seed=7)
    first = run(config, alpha=offering(0.3), beta=offering(0.8))
    swapped = run(config, alpha=offering(0.8), beta=offering(0.3))
    accidents = [set(record.executed) for record in first.rounds]
    assert accidents == [set(record.executed) for record in swapped.rounds]
    assert set() in accidents and any(accidents)  # some rounds misfire, some don't


def test_only_the_player_whose_order_misfired_is_told(toy, monkeypatch):
    def alpha_only(game, state, player, move, rng):
        return FULL if player == "alpha" else move  # an unchanged move is no accident

    def accident_reports(game, state, record):
        return RoundEnd(
            reports={p: {"misfired": p in record.executed} for p in ("alpha", "beta")}
        )

    monkeypatch.setitem(referee.NOISE, TOY, alpha_only)
    monkeypatch.setitem(referee.AFTER_ROUND, TOY, accident_reports)
    state = run(noisy(toy, 1.0, max_rounds=1), alpha=offering(0.3), beta=offering(0.8))
    assert state.rounds[0].executed == {"alpha": FULL}
    assert state.rounds[0].reports == {
        "alpha": {"misfired": True},
        "beta": {"misfired": False},
    }


def test_noise_never_touches_a_default_move(toy, monkeypatch):
    monkeypatch.setitem(referee.NOISE, TOY, misfire_to_full)
    state = run(noisy(toy, 1.0, max_rounds=1), alpha=broken([]), beta=offering(0.8))
    assert state.rounds[0].defaulted == {"alpha"}
    assert state.rounds[0].executed == {"beta": FULL}


def test_one_players_accidents_do_not_depend_on_the_other_players_moves(
    toy, monkeypatch
):
    monkeypatch.setitem(referee.NOISE, TOY, misfire_to_full)
    config = noisy(toy, 0.5, max_rounds=10, seed=3)
    steady = run(config, alpha=offering(0.3), beta=offering(0.8))
    flaky = run(config, alpha=broken([]), beta=offering(0.8))  # alpha only defaults

    def beta_accidents(state):
        return ["beta" in record.executed for record in state.rounds]

    assert beta_accidents(steady) == beta_accidents(flaky)
    assert any(beta_accidents(steady)) and not all(beta_accidents(steady))
