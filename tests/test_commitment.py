"""The commitment stage: the floor, its enforcement, the default, the measures."""

import asyncio
import itertools
import math
import re

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from spar_harness.commitment import (
    Commitment,
    collision,
    enforced,
    fidelity,
    floor_default,
)
from spar_harness.crisis_bargaining import claim, crisis_bargaining_game, default_move
from spar_harness.referee import InvalidMove, run_episode
from spar_harness.schema import (
    Choice,
    EpisodeConfig,
    EpisodeState,
    HorizonTreatment,
    Move,
    Offer,
    PhaseContext,
    RoundRecord,
    Treatments,
    TypePrior,
    UniformPrior,
)
from spar_harness.task import crisis_bargaining


def fixed(value):
    return UniformPrior(low=value, high=value)


def config(version="split", status_quo=0.5, max_rounds=20):
    priors = {
        "alpha": TypePrior(capability=fixed(3.0), cost_of_conflict=fixed(0.1)),
        "beta": TypePrior(capability=fixed(1.0), cost_of_conflict=fixed(0.2)),
    }  # fight values 0.65 and 0.05
    return EpisodeConfig(
        episode_id="test",
        game=crisis_bargaining_game(
            max_rounds=max_rounds, stop_prob=0.0, status_quo=status_quo, version=version
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        priors=priors,
        seed=0,
    )


def scripted(*moves):
    remaining = list(moves)

    async def play(observation, context):
        return remaining.pop(0)

    return play


def run(config, alpha, beta, default_moves=None):
    return asyncio.run(
        run_episode(config, {"alpha": alpha, "beta": beta}, default_moves)
    )


def offer(own_share, me="alpha"):
    return Move(action=Offer(split=own_share if me == "alpha" else 1 - own_share))


# the enforcing wrapper


def test_an_offer_below_the_floor_is_rejected_with_the_floor_named():
    commitment = Commitment(level=3, floor=0.7)
    contexts = []

    async def alpha(observation, context):
        contexts.append(context)
        return offer(0.6) if len(contexts) == 1 else offer(0.7)

    state = run(
        config(),
        enforced(alpha, commitment),
        scripted(Move(action=Choice(kind="accept"))),
    )
    assert "floor of 0.7" in contexts[1].error
    assert state.rounds[0].moves["alpha"] == offer(0.7)  # exactly the floor is fine
    assert state.rounds[0].defaulted == set()
    assert commitment.breaches == 1


def test_accepting_an_offer_below_the_floor_is_rejected_even_on_retry():
    commitment = Commitment(level=3, floor=0.5)
    contexts = []

    async def beta(observation, context):
        contexts.append(context)
        if len(contexts) < 3:
            return Move(action=Choice(kind="accept"))  # alpha offers beta 0.3
        return Move(action=Choice(kind="reject"))

    state = run(
        config(max_rounds=1),
        scripted(offer(0.7)),
        enforced(beta, commitment),
    )
    # the retry observation carries no moves, but the standing offer is remembered
    assert all("floor of 0.5" in context.error for context in contexts[1:])
    assert state.rounds[0].moves["beta"] == Move(action=Choice(kind="reject"))
    assert commitment.breaches == 2


def test_rejecting_and_fighting_are_never_below_a_floor():
    commitment = Commitment(level=5, floor=1.0)
    state = run(
        config(),
        scripted(offer(0.7)),
        enforced(scripted(Move(action=Choice(kind="fight"))), commitment),
    )
    assert state.ending == "fight"
    assert commitment.breaches == 0


def test_without_a_floor_the_wrapper_changes_nothing():
    commitment = Commitment(level=1)
    state = run(
        config(),
        enforced(scripted(offer(0.1)), commitment),
        scripted(Move(action=Choice(kind="accept"))),
    )
    assert state.ending == "agreement"
    assert commitment.breaches == 0


# the default hook


def broken():
    async def play(observation, context):
        raise InvalidMove("no tool call")

    return play


def test_the_referee_uses_a_players_default_override_and_marks_the_default():
    commitment = Commitment(level=4, floor=0.8)
    state = run(
        config(status_quo=0.5),
        broken(),
        scripted(Move(action=Choice(kind="accept"))),
        default_moves={"alpha": floor_default(commitment, default_move)},
    )
    assert state.rounds[0].moves["alpha"] == offer(0.8)  # the floor, not the status quo
    assert state.rounds[0].defaulted == {"alpha"}
    assert state.ending == "agreement"


def test_the_floor_default_is_the_games_default_unless_that_breaches_the_floor():
    game = config(status_quo=0.5).game
    state = EpisodeState(types={})
    context = PhaseContext(
        round=1, role="proposer", action_kind="offer", allowed_actions={"offer"}
    )
    default = floor_default(Commitment(level=2, floor=0.3), default_move)
    assert default(game, state, "beta", context) == offer(0.5, "beta")
    default = floor_default(Commitment(level=4, floor=0.7), default_move)
    assert default(game, state, "beta", context) == offer(0.7, "beta")
    default = floor_default(Commitment(level=4), default_move)
    assert default(game, state, "beta", context) == offer(0.5, "beta")
    responder = context.model_copy(update={"role": "responder"})
    assert default(game, state, "beta", responder) == Move(action=Choice(kind="reject"))


def test_in_the_strict_version_the_floor_default_is_a_claim():
    game = config(version="strict").game
    context = PhaseContext(
        round=1, role="proposer", action_kind="offer", allowed_actions={"offer"}
    )
    default = floor_default(Commitment(level=4, floor=0.7), default_move)
    assert default(game, EpisodeState(), "beta", context) == Move(action=claim("beta"))


# measures


def test_fidelity_counts_kept_moves_against_breaches():
    state = EpisodeState(
        rounds=[
            RoundRecord(round=1, proposer="alpha", moves={"alpha": offer(0.7)}),
            RoundRecord(
                round=2,
                proposer="beta",
                moves={"alpha": offer(0.7)},
                defaulted={"alpha"},
            ),
            RoundRecord(round=3, proposer="alpha", moves={"alpha": offer(0.7)}),
        ]
    )
    assert fidelity(Commitment(level=3, floor=0.7, breaches=2), state, "alpha") == 0.5
    assert fidelity(Commitment(level=3, floor=0.7), state, "alpha") == 1.0
    assert fidelity(Commitment(level=3), state, "alpha") is None


def test_collision_means_no_split_satisfies_both_floors():
    def floors(alpha, beta):
        return {
            "alpha": Commitment(level=3, floor=alpha),
            "beta": Commitment(level=3, floor=beta),
        }

    assert collision(floors(0.6, 0.5)) is True
    assert collision(floors(0.6, 0.4)) is False
    assert collision(floors(0.6, None)) is None
    assert collision({"alpha": Commitment(level=3, floor=0.6)}) is None


# through Inspect, with a mock model


def last_user_text(input):
    return [message for message in input if message.role == "user"][-1].text


def committed_model(floor, offers):
    """Answers the probes, states `floor`, then cycles through `offers` when
    proposing; as responder it accepts exactly the offers at or above `floor`,
    as an obedient delegate would."""
    notes = []
    cycle = itertools.cycle(offers)

    def generate(input, tools, tool_choice, config):
        text = last_user_text(input)
        if "private question" in text.lower():
            return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.4})
        if "state your floor" in text.lower():
            return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": floor})
        notes.extend(
            message.text
            for message in input
            if message.role == "user" and "committed to a floor" in message.text
        )
        # the commitment note follows the round prompt: look for the latest prompt
        prompts = [
            m.text for m in input if m.role == "user" and "You are the" in m.text
        ]
        if "You are the proposer" in prompts[-1]:
            return ModelOutput.for_tool_call(
                "mock", "submit_move", {"action": "offer", "own_share": next(cycle)}
            )
        offered = re.search(r"offer you ([\d.]+) of the stake", prompts[-1]).group(1)
        action = "accept" if float(offered) >= floor else "reject"
        return ModelOutput.for_tool_call("mock", "submit_move", {"action": action})

    return generate, notes


def test_a_committed_model_keeps_its_floor_in_the_conversation_and_is_held_to_it():
    generate, notes = committed_model(0.7, [0.5, 0.8])
    [log] = eval(
        crisis_bargaining(n_episodes=1, a="model", b="equilibrium", commitment_a=4),
        model_roles={"a": get_model("mockllm/model", custom_outputs=generate)},
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    assert notes and all("floor of 0.7" in note for note in notes)
    for sample in log.samples:
        scores = sample.scores["seat_scorer"].value
        assert scores["a_floor"] == 0.7
        assert scores["a_fight_value"] is not None
        assert scores["a_aggressiveness"] == scores["a_floor"] - scores["a_fight_value"]
        assert scores["b_floor"] is None  # the bot made no commitment
        assert sample.scores["outcome_scorer"].value["collision"] is None
        # its first offer of 0.5 was rejected once; every other move was kept
        assert scores["a_breaches"] == 1
        assert 0 < scores["a_fidelity"] < 1
        assert scores["a_default_rate"] == 0.0
        episode = sample.store["episode"]
        assert all(not record["defaulted"] for record in episode["rounds"])
    tool_lists = [
        [tool.model_dump() for tool in event.tools]
        for sample in log.samples
        for event in sample.events
        if event.event == "model"
    ]
    assert all(tools == tool_lists[0] for tools in tool_lists)


def test_two_committed_seats_define_collision():
    generate, _ = committed_model(0.6, [0.8])
    [log] = eval(
        crisis_bargaining(
            n_episodes=1, a="model", b="model", commitment_a=5, commitment_b=5
        ),
        model_roles={
            "a": get_model("mockllm/model", custom_outputs=generate),
            "b": get_model("mockllm/model", custom_outputs=generate),
        },
        display="none",
        log_dir="logs/tests",
    )
    assert log.status == "success"
    for sample in log.samples:
        assert sample.scores["outcome_scorer"].value["collision"] is True
    metrics = {
        score.name: score.metrics["spar_harness/mean_of_defined"].value
        for score in log.results.scores
    }
    assert metrics["collision"] == 1.0


def test_uncommitted_runs_leave_the_floor_measures_undefined():
    [log] = eval(
        crisis_bargaining(n_episodes=1, a="equilibrium", b="equilibrium"),
        model="mockllm/model",
        display="none",
        log_dir="logs/tests",
    )
    metrics = {
        score.name: score.metrics["spar_harness/mean_of_defined"].value
        for score in log.results.scores
    }
    assert math.isnan(metrics["a_floor"])
    assert math.isnan(metrics["collision"])
