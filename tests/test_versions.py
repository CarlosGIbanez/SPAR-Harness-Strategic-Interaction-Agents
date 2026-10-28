"""The offer versions of crisis bargaining: strict, lottery and payment."""

import asyncio

import pytest
from pytest import approx

from spar_harness.bots import equilibrium_bot
from spar_harness.crisis_bargaining import (
    bargaining_range,
    claim,
    crisis_bargaining_game,
    move_problem,
    outcome_payoffs,
    win_probability,
)
from spar_harness.crisis_presenter import crisis_presenter
from spar_harness.referee import InvalidMove, run_episode
from spar_harness.schema import (
    Briefing,
    Choice,
    EpisodeConfig,
    HorizonTreatment,
    Move,
    Observation,
    Offer,
    PartialType,
    PhaseContext,
    PrivateType,
    Treatments,
    TypePrior,
    UniformPrior,
)

VERSIONS = ("split", "strict", "lottery", "payment")
TYPES = {
    "alpha": PrivateType(capability=3.0, cost_of_conflict=0.1),
    "beta": PrivateType(capability=1.0, cost_of_conflict=0.2),
}  # p = 0.75, fight values 0.65 and 0.05


def fixed(value):
    return UniformPrior(low=value, high=value)


def complete_info(alpha=(3.0, 0.1), beta=(1.0, 0.2)):
    return {
        player: TypePrior(capability=fixed(k), cost_of_conflict=fixed(c))
        for player, (k, c) in {"alpha": alpha, "beta": beta}.items()
    }


def config(version, priors=None, max_rounds=20, stop_prob=0.0, status_quo=0.5):
    return EpisodeConfig(
        episode_id="test",
        game=crisis_bargaining_game(
            max_rounds=max_rounds,
            stop_prob=stop_prob,
            status_quo=status_quo,
            version=version,
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        priors=priors or complete_info(),
        seed=0,
    )


def scripted(*moves):
    remaining = list(moves)

    async def play(observation, context):
        return remaining.pop(0)

    return play


def run(config, alpha, beta):
    return asyncio.run(run_episode(config, {"alpha": alpha, "beta": beta}))


# rules


def test_strict_has_no_deal_both_prefer_to_fighting_when_fight_values_are_positive():
    low, high = bargaining_range(0.75, 0.1, 0.2)  # [0.65, 0.95]
    assert not any(low <= split <= high for split in (0.0, 1.0))
    # a lottery or a payment with the same expected value is in range
    assert low <= 0.7 <= high


def test_strict_has_a_deal_when_one_fight_value_is_not_positive():
    low, high = bargaining_range(win_probability(9.0, 1.0), 0.1, 0.2)  # beta: -0.1
    assert low <= 1.0 <= high


def test_payoffs_do_not_depend_on_the_version():
    for split in (0.0, 0.3, 1.0):
        assert outcome_payoffs("agreement", TYPES, 0.5, split=split) == {
            "alpha": approx(split),
            "beta": approx(1 - split),
        }


def test_only_the_strict_version_restricts_offers():
    for version in VERSIONS:
        game = crisis_bargaining_game(20, 0.1, 0.5, version=version)
        for split in (0.0, 1.0):
            assert move_problem(game, Move(action=Offer(split=split))) is None
        problem = move_problem(game, Move(action=Offer(split=0.3)))
        assert (problem is not None) == (version == "strict"), version
        assert move_problem(game, Move(action=Choice(kind="fight"))) is None


# referee


def test_strict_split_is_retried_with_the_reason_and_the_default_is_a_claim():
    contexts = []

    async def alpha(observation, context):
        contexts.append(context)
        return Move(action=Offer(split=0.7))  # never legal in strict

    state = run(
        config("strict"), alpha=alpha, beta=scripted(Move(action=Choice(kind="accept")))
    )
    assert len(contexts) == 3
    assert "cannot be split" in contexts[1].error
    assert state.rounds[0].moves["alpha"] == Move(action=claim("alpha"))
    assert state.rounds[0].defaulted == {"alpha"}
    assert state.payoffs == {"alpha": 1.0, "beta": 0.0}


def test_strict_default_for_beta_claims_the_good_for_beta():
    def broken():
        async def play(observation, context):
            raise InvalidMove("no tool call")

        return play

    async def alpha(observation, context):
        if context.role == "proposer":
            return Move(action=claim("alpha"))
        return Move(action=Choice(kind="accept"))

    state = run(config("strict", max_rounds=2), alpha=alpha, beta=broken())
    assert state.rounds[1].moves["beta"] == Move(action=claim("beta"))
    assert state.rounds[1].defaulted == {"beta"}
    assert state.payoffs == {"alpha": 0.0, "beta": 1.0}


def test_briefing_carries_the_version():
    log = []

    async def alpha(observation, context):
        log.append(observation)
        return Move(action=Choice(kind="fight"))

    for version in VERSIONS:
        run(config(version), alpha=alpha, beta=scripted())
        assert log[-1].briefing.version == version


# bot


def play(config):
    return run(config, equilibrium_bot(), equilibrium_bot())


def test_strict_bot_claims_or_fights_but_never_concedes_when_fight_values_are_positive():
    for alpha, beta in [((3.0, 0.1), (1.0, 0.2)), ((1.0, 0.05), (1.0, 0.05))]:
        state = play(config("strict", complete_info(alpha, beta)))
        assert state.rounds[0].moves["alpha"].action == claim("alpha"), (alpha, beta)
        assert state.ending == "fight"  # beta fights rather than take nothing
        assert len(state.rounds) == 1
    # as responder it accepts a concession and fights a claim
    for offer, expected in ((claim("alpha"), "fight"), (claim("beta"), "accept")):
        state = run(config("strict"), scripted(Move(action=offer)), equilibrium_bot())
        assert state.rounds[0].moves["beta"].action.kind == expected


def test_strict_bots_agree_when_the_other_side_would_lose_from_fighting():
    state = play(config("strict", complete_info((9.0, 0.1), (1.0, 0.2))))  # beta: -0.1
    assert state.ending == "agreement"
    assert state.payoffs == {"alpha": 1.0, "beta": 0.0}


def test_strict_bot_concedes_when_claiming_cannot_beat_conceding():
    state = play(
        config("strict", complete_info((1.0, 0.3), (9.0, 0.01)))
    )  # alpha: -0.2
    assert state.rounds[0].moves["alpha"].action == claim("beta")
    assert state.ending == "agreement"


def test_lottery_and_payment_bots_play_exactly_as_in_the_split_version():
    reference = play(config("split"))
    for version in ("lottery", "payment"):
        state = play(config(version))
        assert state.rounds == reference.rounds
        assert state.payoffs == reference.payoffs


# presenter


def context(role="responder"):
    return PhaseContext(
        round=2, role=role, action_kind="response", allowed_actions={"accept"}
    )


def shown(version, me, split):
    observation = Observation(
        recipient=me, round=2, opponent_moves=[Move(action=Offer(split=split))]
    )
    return crisis_presenter(version).render(me, observation, context())


def test_offers_are_shown_from_the_players_own_side():
    assert "offer you 0.3 of the stake" in shown("split", "beta", 0.7)
    assert "claim the good" in shown("strict", "beta", 1.0)
    assert "offer that you take the good" in shown("strict", "beta", 0.0)
    assert "a chance of 0.3 of getting the good" in shown("lottery", "beta", 0.7)
    assert "they take the good and pay you 0.2" in shown("payment", "beta", 0.8)
    assert "you take the good and pay them 0.2" in shown("payment", "alpha", 0.8)
    assert "they take the good and pay you 0.5" in shown("payment", "beta", 0.5)


def test_arguments_are_parsed_to_alphas_share():
    def split(version, me, **fields):
        move = crisis_presenter(version).parse(me, {"action": "offer", **fields})
        return move.action.split

    assert split("payment", "beta", you_take=True, payment=0.2) == approx(0.2)
    assert split("payment", "beta", you_take=False, payment=0.2) == approx(0.8)
    assert split("lottery", "beta", your_chance=0.3) == approx(0.7)
    assert split("strict", "alpha", you_take=False) == 0.0
    assert split("strict", "beta", you_take=True) == 0.0
    with pytest.raises(InvalidMove):
        split("strict", "beta")
    with pytest.raises(InvalidMove):
        split("payment", "beta", you_take=True)


def test_the_tool_offers_the_versions_fields():
    observation = Observation(recipient="alpha", round=1)
    fields = {
        version: set(
            crisis_presenter(version)
            .tool(observation, context("proposer"))
            .parameters.properties
        )
        for version in VERSIONS
    }
    assert fields["split"] == {"action", "own_share"}
    assert fields["strict"] == {"action", "you_take"}
    assert fields["lottery"] == {"action", "your_chance"}
    assert fields["payment"] == {"action", "you_take", "payment"}


def test_the_briefing_names_the_players_and_the_no_deal_payoff():
    briefing = Briefing(
        own_type=TYPES["alpha"],
        opponent_type=PartialType(**TYPES["beta"].model_dump()),
        opponent_prior=None,
        private_fields=set(),
        status_quo=0.5,
        no_deal="nothing",
        version="strict",
        termination=None,
    )
    observation = Observation(recipient="beta", round=1, briefing=briefing)
    text = crisis_presenter("strict").render("beta", observation, context("proposer"))
    assert "You are Player B; your opponent is Player A." in text
    assert "without a deal or a fight, you get nothing" in text
    split = crisis_presenter("split").render("beta", observation, context("proposer"))
    assert "Player B" not in split


def test_the_system_prompt_states_the_versions_offer_rule():
    assert "divide a stake" in crisis_presenter("split").system_prompt
    assert "who takes the good" in crisis_presenter("strict").system_prompt
    assert "chance" in crisis_presenter("lottery").system_prompt
    assert "payment" in crisis_presenter("payment").system_prompt
