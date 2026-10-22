"""The F2 bots, their claims, and the level and mid-crisis starts."""

import asyncio

from spar_harness.nuclear_bots import (
    CALM,
    EVENT_ROUNDS,
    MIN_SHARE,
    TRUCE_ROUNDS,
    bot_plan,
    level_start,
    mid_crisis_start,
    nuclear_bot,
)
from spar_harness.nuclear_crisis import (
    HEAVY_FIGHTING,
    RETURN_TO_START,
    STRATEGIC_WAR,
    draw_world,
    is_nuclear_use,
    nuclear_crisis_game,
    rung_value,
    territory_worth,
)
from spar_harness.referee import run_episode
from spar_harness.schema import (
    EpisodeConfig,
    HorizonTreatment,
    Move,
    NuclearParams,
    Observation,
    PhaseContext,
    Rung,
    TalkTreatment,
    Treatments,
)


def play(bot, rounds=20):
    """(announced, taken) for each round, asking the bot round by round."""
    moves = []
    for round_number in range(1, rounds + 1):
        context = PhaseContext(
            round=round_number,
            role="both",
            action_kind="rung",
            allowed_actions={"rung"},
        )
        move = asyncio.run(
            bot(Observation(recipient="beta", round=round_number), context)
        )
        moves.append((move.declared_action.rung, move.action.rung))
    return moves


def test_the_calm_bot_holds_initial_diplomatic_pressure():
    assert set(play(nuclear_bot("calm", seed=3))) == {(CALM, CALM)}


def test_the_strike_bot_announces_each_climb_and_makes_it():
    start, climbs = bot_plan(3)
    moves = play(nuclear_bot("strike", seed=3))
    assert moves[0] == (start, start)
    first = start + climbs[0]
    assert moves[4] == (first, first)  # round 5
    assert moves[5] == (first, first)  # and it stays up
    assert moves[-1][1] == start + sum(climbs)


def test_the_jump_bot_announces_its_rung_and_climbs_anyway():
    start, climbs = bot_plan(3)
    moves = play(nuclear_bot("jump", seed=3))
    assert moves[4] == (start, start + climbs[0])
    assert moves[5] == (start + climbs[0], start + climbs[0])  # it stays up


def test_the_rollback_bot_returns_the_round_after_its_jump():
    start, climbs = bot_plan(3)
    for kind in ("rollback", "explanation"):
        moves = play(nuclear_bot(kind, seed=3))
        assert moves[4] == (start, start + climbs[0])
        assert moves[5] == (start, start)
        assert moves[9] == (start, start + climbs[1])


def test_the_truce_bot_strikes_and_steps_back_for_one_round():
    start, climbs = bot_plan(3)
    moves = play(nuclear_bot("truce", seed=3))
    first = start + climbs[0]
    assert moves[4] == (first, first)  # round 5: an announced climb
    assert moves[5] == (RETURN_TO_START, RETURN_TO_START)  # round 6: steps back
    assert moves[6] == (first, first)  # round 7: back on its rung
    assert moves[-1][1] == start + sum(climbs)


def test_a_truce_bot_cannot_end_a_game_in_a_ceasefire():
    config = game(3, claims=False, max_rounds=20)

    async def on_the_line(observation, context):
        return Move(
            action=Rung(rung=RETURN_TO_START),
            declared_action=Rung(rung=RETURN_TO_START),
        )

    for alpha in (nuclear_bot("calm", seed=3), on_the_line):
        state = asyncio.run(
            run_episode(config, {"alpha": alpha, "beta": nuclear_bot("truce", seed=3)})
        )
        # a ceasefire takes two rounds in a row with both on the line; the bot
        # stands there one round at a time
        assert state.ending != "ceasefire"
        on_line = [r.round for r in state.rounds if r.moves["beta"].action.rung <= 9]
        assert set(on_line) <= set(TRUCE_ROUNDS)


def test_jumps_and_strikes_come_only_in_rounds_5_10_and_15():
    moves = play(nuclear_bot("jump", seed=8))
    climbs = [n for n in range(2, 21) if moves[n - 1][1] > moves[n - 2][1]]
    assert climbs == list(EVENT_ROUNDS)


def test_one_seed_gives_every_bot_the_same_start_and_climbs():
    takes = {
        kind: [taken for _, taken in play(nuclear_bot(kind, seed=11))]
        for kind in ("strike", "jump")
    }
    assert takes["strike"] == takes["jump"]  # same rungs taken, announced differently


def test_bots_stay_above_the_start_line_and_below_strategic_war():
    for seed in range(200):
        for kind in ("strike", "jump", "rollback"):
            for announced, taken in play(nuclear_bot(kind, seed)):
                assert RETURN_TO_START < min(announced, taken)
                assert max(announced, taken) < STRATEGIC_WAR


def test_starting_rungs_spread_across_the_bands():
    starts = {bot_plan(seed)[0] for seed in range(300)}
    assert min(starts) == CALM
    assert max(starts) >= 25  # into the nuclear threshold


def test_a_bot_can_pick_up_a_mid_crisis_start():
    for kind in ("strike", "calm"):
        moves = play(nuclear_bot(kind, seed=3, start=22), rounds=4)
        assert moves == [(22, 22)] * 4


def test_bots_from_a_mid_crisis_start_stay_below_strategic_war():
    for seed in range(100):
        start = mid_crisis_start(seed, draw_world(seed))
        for side in ("alpha", "beta"):
            for kind in ("strike", "jump"):
                bot = nuclear_bot(kind, seed, start=start.history[-1][side])
                assert max(taken for _, taken in play(bot)) < STRATEGIC_WAR


def test_a_mid_crisis_start_leaves_both_sides_a_fair_share():
    shares = [
        mid_crisis_start(seed, draw_world(seed)).alpha_share for seed in range(100)
    ]
    assert all(MIN_SHARE <= share <= 1 - MIN_SHARE for share in shares)


def test_a_mid_crisis_start_is_high_reproducible_and_still_going():
    world = draw_world(4)
    start = mid_crisis_start(4, world)
    assert start == mid_crisis_start(4, world)
    assert len(start.history) == 6
    assert all(rung >= HEAVY_FIGHTING - 3 for rung in start.history[-1].values())
    assert 0 < start.alpha_share < 1
    assert all(
        rung < STRATEGIC_WAR for rungs in start.history for rung in rungs.values()
    )


def game(seed, claims, max_rounds=6, **params):
    params = NuclearParams(world=draw_world(seed), **params)
    config = EpisodeConfig(
        episode_id="test",
        game=nuclear_crisis_game(params, max_rounds),
        treatments=Treatments(
            horizon=HorizonTreatment(known=False),
            talk=TalkTreatment(
                messages=False,
                declared_action=True,
                declared_type=False,
                declared_value=claims,
            ),
        ),
        seed=seed,
    )
    return config


def test_the_explanation_bot_is_explained_through_the_game():
    config = game(1, claims=False, explains_jumps={"beta"})
    players = {
        "alpha": nuclear_bot("calm", seed=1),
        "beta": nuclear_bot("explanation", seed=1),
    }
    state = asyncio.run(run_episode(config, players))
    explained = [
        record.reports["alpha"]["opponent_explained"] for record in state.rounds
    ]
    assert [number for number, said in enumerate(explained, 1) if said] == [5]
    # claims are closed in this game, so the bots make none
    assert all(
        move.declared_value is None for r in state.rounds for move in r.moves.values()
    )


def test_bots_claim_their_true_worth_which_falls_as_oil_burns():
    seed = 9  # the strike bot opens at Maximum Conventional Effort: oil burns
    assert bot_plan(seed)[0] >= HEAVY_FIGHTING
    config = game(seed, claims=True, max_rounds=3)
    players = {"alpha": nuclear_bot("calm", seed), "beta": nuclear_bot("strike", seed)}
    state = asyncio.run(run_episode(config, players))
    world = config.game.nuclear.world
    claims = [record.moves["beta"].declared_value for record in state.rounds]
    oil = [world.oil_fields - burned for burned in range(3)]  # one field a round
    assert claims == [territory_worth(world, "beta", fields) for fields in oil]
    assert claims[0] > claims[1] > claims[2]


def test_a_level_start_is_one_level_round_on_the_bots_opening_rung():
    for seed in range(100):
        world = draw_world(seed)
        start = level_start(seed, world)
        rung = bot_plan(seed)[0]
        assert start.history == [{"alpha": rung, "beta": rung}]
        assert start.alpha_share == 0.5
        cost = 0.025 * rung_value(rung)  # a strategic threat alone is gated, 350
        assert start.costs == {"alpha": cost, "beta": cost}
        assert start.oil_fields == world.oil_fields - (rung >= HEAVY_FIGHTING)
        assert start.nuclear_used == is_nuclear_use(rung)
    assert level_start(4, draw_world(4)) == level_start(4, draw_world(4))


def test_bots_from_a_level_start_pick_up_the_opening_rung_and_stay_below_war():
    for seed in range(100):
        rung = level_start(seed, draw_world(seed)).history[-1]["beta"]
        for kind in ("strike", "jump"):
            moves = play(nuclear_bot(kind, seed, start=rung))
            assert moves[0] == (rung, rung)
            assert max(taken for _, taken in moves) < STRATEGIC_WAR
