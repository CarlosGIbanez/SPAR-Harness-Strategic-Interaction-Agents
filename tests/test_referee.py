import asyncio

from pytest import approx

from spar_harness.crisis_bargaining import crisis_bargaining_game
from spar_harness.referee import InvalidMove, run_episode
from spar_harness.schema import (
    Choice,
    EpisodeConfig,
    HorizonTreatment,
    Move,
    Offer,
    PartialType,
    PrivateInfoTreatment,
    PrivateType,
    TalkTreatment,
    TerminationSpec,
    Treatments,
    TypePrior,
    UniformPrior,
)


def fixed_priors():
    def prior(capability, cost):
        return TypePrior(
            capability=UniformPrior(low=capability, high=capability),
            cost_of_conflict=UniformPrior(low=cost, high=cost),
        )

    return {"alpha": prior(3.0, 0.1), "beta": prior(1.0, 0.2)}


def config(max_rounds=20, stop_prob=0.0, status_quo=0.5, seed=0):
    return EpisodeConfig(
        episode_id="test",
        game=crisis_bargaining_game(
            max_rounds=max_rounds, stop_prob=stop_prob, status_quo=status_quo
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        priors=fixed_priors(),
        seed=seed,
    )


def scripted(*moves):
    """A player that plays the given moves in order."""
    remaining = list(moves)

    async def play(observation, context):
        return remaining.pop(0)

    return play


def run(config, alpha, beta):
    return asyncio.run(run_episode(config, {"alpha": alpha, "beta": beta}))


def test_accepted_offer_ends_the_episode_in_agreement():
    state = run(
        config(),
        alpha=scripted(Move(action=Offer(split=0.7))),
        beta=scripted(Move(action=Choice(kind="accept"))),
    )
    assert state.ending == "agreement"
    assert len(state.rounds) == 1
    assert state.payoffs == {"alpha": approx(0.7), "beta": approx(0.3)}


def test_rejection_moves_play_to_the_next_round_with_the_other_proposer():
    state = run(
        config(),
        alpha=scripted(
            Move(action=Offer(split=0.9)), Move(action=Choice(kind="accept"))
        ),
        beta=scripted(
            Move(action=Choice(kind="reject")), Move(action=Offer(split=0.6))
        ),
    )
    assert [r.proposer for r in state.rounds] == ["alpha", "beta"]
    assert state.ending == "agreement"
    assert state.payoffs == {"alpha": approx(0.6), "beta": approx(0.4)}


def test_proposer_can_fight_instead_of_offering():
    beta_moves = []

    async def beta(observation, context):
        beta_moves.append(context)
        return Move(action=Choice(kind="accept"))

    state = run(config(), alpha=scripted(Move(action=Choice(kind="fight"))), beta=beta)
    assert state.ending == "fight"
    assert beta_moves == []
    assert state.payoffs == {"alpha": approx(0.65), "beta": approx(0.05)}


def test_episode_ends_at_the_horizon_with_the_status_quo():
    state = run(config(max_rounds=3, status_quo=0.4), alpha=stubborn(), beta=stubborn())
    assert len(state.rounds) == 3
    assert state.ending == "horizon"
    assert state.payoffs == {"alpha": approx(0.4), "beta": approx(0.6)}


def stubborn():
    """Offers an even split and rejects every offer."""

    async def play(observation, context):
        if context.role == "proposer":
            return Move(action=Offer(split=0.5))
        return Move(action=Choice(kind="reject"))

    return play


def test_certain_random_stop_ends_the_episode_after_the_first_round():
    state = run(
        config(stop_prob=1.0, status_quo=0.4), alpha=stubborn(), beta=stubborn()
    )
    assert len(state.rounds) == 1
    assert state.ending == "random_stop"
    assert state.payoffs == {"alpha": approx(0.4), "beta": approx(0.6)}


def test_random_stop_is_reproducible_from_the_seed():
    lengths = [
        len(
            run(
                config(stop_prob=0.5, seed=seed), alpha=stubborn(), beta=stubborn()
            ).rounds
        )
        for seed in (11, 11)
    ]
    assert lengths[0] == lengths[1]


def test_last_round_ends_at_the_horizon_even_with_a_random_stop():
    state = run(config(max_rounds=1, stop_prob=1.0), alpha=stubborn(), beta=stubborn())
    assert state.ending == "horizon"


def recording(player, log):
    """Wraps a player and records every observation it receives."""

    async def play(observation, context):
        log.append(observation)
        return await player(observation, context)

    return play


def test_players_observe_every_opponent_move_since_their_last_turn():
    alpha_log, beta_log = [], []
    alpha_offer = Move(action=Offer(split=0.9))
    beta_reject = Move(action=Choice(kind="reject"))
    beta_offer = Move(action=Offer(split=0.6))
    run(
        config(),
        alpha=recording(
            scripted(alpha_offer, Move(action=Choice(kind="accept"))), alpha_log
        ),
        beta=recording(scripted(beta_reject, beta_offer), beta_log),
    )
    assert alpha_log[0].opponent_moves == []
    assert beta_log[0].opponent_moves == [alpha_offer]
    assert beta_log[1].opponent_moves == []
    assert alpha_log[1].opponent_moves == [beta_reject, beta_offer]


def test_first_observation_briefs_the_player_on_the_game():
    alpha_log = []
    run(
        config(max_rounds=5, stop_prob=0.1, status_quo=0.4),
        alpha=recording(
            scripted(Move(action=Offer(split=0.9)), Move(action=Choice(kind="accept"))),
            alpha_log,
        ),
        beta=scripted(
            Move(action=Choice(kind="reject")), Move(action=Offer(split=0.6))
        ),
    )
    briefing = alpha_log[0].briefing
    assert briefing.own_type == PrivateType(capability=3.0, cost_of_conflict=0.1)
    assert briefing.opponent_type == PartialType(capability=1.0, cost_of_conflict=0.2)
    assert briefing.status_quo == 0.4
    assert briefing.termination == TerminationSpec(max_rounds=5, stop_prob=0.1)
    assert alpha_log[1].briefing is None


def first_briefing(config, player="alpha"):
    log = []
    players = {"alpha": stubborn(), "beta": stubborn()}
    players[player] = recording(players[player], log)
    run(config, **players)
    return log[0].briefing


def test_unknown_horizon_is_not_disclosed():
    unknown = config().model_copy(
        update={"treatments": Treatments(horizon=HorizonTreatment(known=False))}
    )
    assert first_briefing(unknown).termination is None


def test_private_fields_of_the_opponent_are_hidden_and_their_prior_disclosed():
    private = config().model_copy(
        update={
            "treatments": Treatments(
                horizon=HorizonTreatment(known=True),
                private_info=PrivateInfoTreatment(private_fields={"capability"}),
            )
        }
    )
    briefing = first_briefing(private)
    assert briefing.own_type == PrivateType(capability=3.0, cost_of_conflict=0.1)
    assert briefing.opponent_type == PartialType(capability=None, cost_of_conflict=0.2)
    assert briefing.opponent_prior == fixed_priors()["beta"]
    assert briefing.private_fields == {"capability"}


def test_complete_information_discloses_no_prior():
    briefing = first_briefing(config())
    assert briefing.opponent_prior is None
    assert briefing.private_fields == set()


def test_invalid_move_is_retried_with_the_reason():
    contexts = []

    async def confused_alpha(observation, context):
        contexts.append(context)
        if len(contexts) == 1:
            return Move(action=Choice(kind="accept"))  # a proposer cannot accept
        return Move(action=Offer(split=0.7))

    state = run(
        config(),
        alpha=confused_alpha,
        beta=scripted(Move(action=Choice(kind="accept"))),
    )
    assert contexts[0].error is None
    assert "accept" in contexts[1].error
    assert state.rounds[0].moves["alpha"] == Move(action=Offer(split=0.7))
    assert state.ending == "agreement"


def broken(calls):
    """A player that can never produce a valid move."""

    async def play(observation, context):
        calls.append(context)
        raise InvalidMove("no tool call")

    return play


def test_broken_proposer_offers_the_status_quo_after_two_retries():
    calls = []
    state = run(
        config(status_quo=0.4),
        alpha=broken(calls),
        beta=scripted(Move(action=Choice(kind="accept"))),
    )
    assert len(calls) == 3
    assert state.rounds[0].moves["alpha"] == Move(action=Offer(split=0.4))
    assert state.rounds[0].defaulted == {"alpha"}
    assert state.ending == "agreement"


def test_broken_responder_rejects():
    state = run(config(max_rounds=1), alpha=stubborn(), beta=broken([]))
    assert state.rounds[0].moves["beta"] == Move(action=Choice(kind="reject"))
    assert state.ending == "horizon"


def test_proposer_that_breaks_later_repeats_its_last_offer():
    attempts = []

    async def alpha(observation, context):
        attempts.append(context)
        if context.round == 1:
            return Move(action=Offer(split=0.8))
        if context.role == "responder":
            return Move(action=Choice(kind="reject"))
        raise InvalidMove("no tool call")

    state = run(config(max_rounds=3), alpha=alpha, beta=stubborn())
    assert state.rounds[2].moves["alpha"] == Move(action=Offer(split=0.8))
    assert state.rounds[2].defaulted == {"alpha"}


def with_treatments(**treatments):
    return config().model_copy(
        update={
            "treatments": Treatments(horizon=HorizonTreatment(known=True), **treatments)
        }
    )


def first_attempt_errors(config, alpha_move):
    """Alpha tries alpha_move first, then a plain offer; returns the referee's error."""
    contexts = []

    async def alpha(observation, context):
        contexts.append(context)
        return alpha_move if len(contexts) == 1 else Move(action=Offer(split=0.7))

    run(config, alpha=alpha, beta=scripted(Move(action=Choice(kind="accept"))))
    return contexts[1].error if len(contexts) > 1 else None


def test_talk_is_rejected_when_the_treatment_is_off():
    move = Move(action=Offer(split=0.7), message="be reasonable")
    assert "message" in first_attempt_errors(config(), move)


def test_each_talk_field_needs_its_own_switch():
    talk = TalkTreatment(messages=True, declared_action=False, declared_type=False)
    message = Move(action=Offer(split=0.7), message="be reasonable")
    threat = Move(action=Offer(split=0.7), declared_action=Choice(kind="fight"))
    assert first_attempt_errors(with_treatments(talk=talk), message) is None
    assert "declared_action" in first_attempt_errors(with_treatments(talk=talk), threat)


def test_declared_types_may_only_claim_private_fields():
    talk = TalkTreatment(messages=False, declared_action=False, declared_type=True)
    private = PrivateInfoTreatment(private_fields={"capability"})
    config = with_treatments(talk=talk, private_info=private)
    bluff = Move(action=Offer(split=0.7), declared_type=PartialType(capability=9.0))
    public_claim = Move(
        action=Offer(split=0.7), declared_type=PartialType(cost_of_conflict=0.5)
    )
    assert first_attempt_errors(config, bluff) is None
    assert "cost_of_conflict" in first_attempt_errors(config, public_claim)


def test_context_tells_the_player_which_talk_fields_are_open():
    talk = TalkTreatment(messages=True, declared_action=True, declared_type=False)
    contexts = []

    async def alpha(observation, context):
        contexts.append(context)
        return Move(action=Offer(split=0.7))

    run(
        with_treatments(talk=talk),
        alpha=alpha,
        beta=scripted(Move(action=Choice(kind="accept"))),
    )
    assert contexts[0].talk_fields == {"message", "declared_action"}
