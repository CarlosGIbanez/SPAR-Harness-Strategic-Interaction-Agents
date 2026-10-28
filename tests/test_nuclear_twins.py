"""F2 twins: situations from the reports, the fit, and twins playing the game."""

import asyncio
import random

from spar_harness.nuclear_bots import level_start, nuclear_bot
from spar_harness.nuclear_crisis import draw_world, nuclear_crisis_game
from spar_harness.nuclear_twins import (
    COPY,
    Situation,
    decisions,
    first_move,
    fit,
    move_class,
    situation_after,
    twin,
)
from spar_harness.referee import run_episode
from spar_harness.schema import (
    EpisodeConfig,
    HorizonTreatment,
    Move,
    NuclearParams,
    Rung,
    TalkTreatment,
    Treatments,
)

TALK = TalkTreatment(
    messages=False, declared_action=True, declared_type=False, declared_value=True
)


def config(seed=1, max_rounds=8, start=None, **params):
    world = draw_world(seed)
    return EpisodeConfig(
        episode_id="test",
        game=nuclear_crisis_game(
            NuclearParams(world=world, start=start, **params), max_rounds
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=False), talk=TALK),
        seed=seed,
    )


def taking(*moves):
    """(announced, taken) pairs or plain rungs, in order, then the last again."""
    remaining = [m if isinstance(m, tuple) else (m, m) for m in moves]

    async def play(observation, context):
        announced, taken = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return Move(action=Rung(rung=taken), declared_action=Rung(rung=announced))

    return play


def test_move_classes_cap_at_minus_two_and_plus_three():
    assert [move_class(d) for d in (-7, -2, -1, 0, 1, 2, 3, 9)] == [
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        5,
    ]


def test_a_situation_is_what_the_side_knew_before_its_move():
    cfg = config(max_rounds=3)
    state = asyncio.run(
        run_episode(
            cfg, {"alpha": taking(12, 12, 14), "beta": taking(12, (12, 15), 15)}
        )
    )
    reports = [record.reports["alpha"] for record in state.rounds]
    assert situation_after(reports[:2]) == Situation(
        own=12,
        opponent=15,
        opponent_move=3,
        jump=True,
        explained=False,
        own_move=0,
        exceeded=1,
        below_peak=3,
    )
    rows = decisions(state, cfg, "alpha")
    assert [delta for _, delta in rows] == [0, 2]  # 12 -> 12, then 12 -> 14
    assert first_move(state, cfg, "alpha") == 12


def test_a_game_with_a_history_starts_from_it():
    start = level_start(4, draw_world(4))
    cfg = config(seed=4, start=start)
    rung = start.history[-1]["alpha"]
    state = asyncio.run(
        run_episode(cfg, {"alpha": taking(rung + 1), "beta": taking(rung)})
    )
    situation, delta = decisions(state, cfg, "alpha")[0]
    assert (situation.own, situation.opponent, delta) == (rung, rung, 1)
    assert first_move(state, cfg, "alpha") is None


def test_a_twin_fitted_on_a_copier_copies():
    rows = [
        (Situation(own, opp, opp - own, False, False), opp - own)
        for own in range(10, 25)
        for opp in range(10, 25)
        if abs(opp - own) <= 6
    ] * 3
    copier = fit(rows, "T0", first_moves=[10], copy=True)
    assert copier.copy
    situation = Situation(
        own=12, opponent=19, opponent_move=7, jump=False, explained=False
    )
    assert copier.probabilities(situation).argmax() == COPY
    plain = fit(rows, "T0", first_moves=[10])
    near = Situation(own=12, opponent=13, opponent_move=1, jump=False, explained=False)
    assert plain.probabilities(near).argmax() == move_class(1)


def test_twins_play_legal_honest_moves_against_a_bot_and_each_other():
    rows = [
        (Situation(own, own + gap, gap, False, False), gap)
        for own in range(10, 20)
        for gap in (-1, 0, 1)
    ] * 5
    policy = fit(rows, "T3", first_moves=[10, 11], copy=True)
    cfg = config(seed=2, max_rounds=10)
    for opponent in (nuclear_bot("strike", seed=2), twin(policy, random.Random(1))):
        state = asyncio.run(
            run_episode(
                cfg, {"alpha": twin(policy, random.Random(0)), "beta": opponent}
            )
        )
        for record in state.rounds:
            move = record.moves["alpha"]
            assert 1 <= move.action.rung <= 29
            assert move.declared_action == move.action  # honest announcements
            assert move.declared_value is not None
