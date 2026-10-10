import asyncio

from spar_harness.contract_dynamics import contract_dynamics_game
from spar_harness.referee import InvalidMove, run_episode
from spar_harness.schema import (
    Choice,
    ContractOffer,
    ContractParams,
    ContractState,
    EpisodeConfig,
    HorizonTreatment,
    Move,
    TalkTreatment,
    Treatments,
)


def params(**overrides):
    values = {
        "benefits": {"alpha": 18.0, "beta": 18.0},
        "shock_prob": 0.5,
        "loss": 60.0,
        "disaster": 40.0,
        "penalty": 0.0,
        "transfer_cap": 60.0,
    }
    return ContractParams(**(values | overrides))


def config(
    seed=0, max_rounds=6, persuasion_rounds=0, imposed_venture=False, **overrides
):
    return EpisodeConfig(
        episode_id="test",
        game=contract_dynamics_game(
            params(imposed_venture=imposed_venture, **overrides),
            max_rounds=max_rounds,
            persuasion_rounds=persuasion_rounds,
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        seed=seed,
    )


def scripted(*moves):
    remaining = list(moves)

    async def play(observation, context):
        return remaining.pop(0)

    return play


def run(config, alpha, beta):
    return asyncio.run(run_episode(config, {"alpha": alpha, "beta": beta}))


def propose(q, t):
    return Move(action=ContractOffer(alpha_bears_prob=q, transfer=t))


def choose(kind, **talk):
    return Move(action=Choice(kind=kind), **talk)


def test_walking_away_leaves_both_at_the_status_quo():
    state = run(
        config(), alpha=scripted(propose(0.5, 20.0)), beta=scripted(choose("walk_away"))
    )
    assert state.ending == "walked_away"
    assert state.contract is None
    assert state.payoffs == {"alpha": 0.0, "beta": 0.0}


def test_signed_contract_without_a_shock_pays_both_benefits():
    state = run(
        config(shock_prob=0.0),
        alpha=scripted(propose(0.5, 20.0)),
        beta=scripted(choose("accept")),
    )
    assert state.ending == "no_shock"
    assert state.contract == ContractState(
        alpha_bears_prob=0.5, transfer=20.0, shock=False
    )
    assert state.payoffs == {"alpha": 18.0, "beta": 18.0}


def test_after_a_shock_the_drawn_bearer_decides():
    # q = 1: alpha always bears. Honoring with t = 20: alpha 18 - 60 + 20, beta 18 - 20.
    state = run(
        config(shock_prob=1.0),
        alpha=scripted(propose(1.0, 20.0), choose("honor")),
        beta=scripted(choose("accept")),
    )
    assert state.contract.bearer == "alpha"
    assert state.ending == "honored"
    assert state.payoffs == {"alpha": -22.0, "beta": -2.0}
    # q = 0: beta bears and reneges: beta 18 - 40 - F, alpha 18 - 40
    state = run(
        config(shock_prob=1.0, penalty=5.0),
        alpha=scripted(propose(0.0, 20.0)),
        beta=scripted(choose("accept"), choose("renege")),
    )
    assert (state.contract.bearer, state.ending) == ("beta", "reneged")
    assert state.payoffs == {"alpha": -22.0, "beta": -27.0}


def recording(moves, log):
    """A scripted player that records every observation and context it receives."""
    play = scripted(*moves)

    async def wrapped(observation, context):
        log.append((observation, context))
        return await play(observation, context)

    return wrapped


def test_the_other_persuades_the_bearer_who_then_decides_alone():
    alpha_log, beta_log = [], []
    plea = Move(message="Please honor it.")
    reply = Move(message="I will think about it.")
    run(
        config(shock_prob=1.0, persuasion_rounds=2),
        # alpha proposes that it always bears, then hears two pleas and decides
        alpha=recording(
            [propose(1.0, 20.0), reply, reply, choose("renege")], alpha_log
        ),
        beta=recording([choose("accept"), plea, plea], beta_log),
    )
    beta_first_plea, beta_context = beta_log[1]
    assert beta_context.role == "other"
    assert beta_first_plea.contract == ContractState(
        alpha_bears_prob=1.0, transfer=20.0, shock=True, bearer="alpha"
    )
    alpha_first_reply, alpha_context = alpha_log[1]
    assert alpha_context.role == "bearer"
    assert alpha_first_reply.opponent_moves == [choose("accept"), plea]
    assert alpha_log[2][0].opponent_moves == [plea]  # the second plea
    alpha_decision, decision_context = alpha_log[3]
    assert decision_context.allowed_actions == {"honor", "renege"}
    assert alpha_decision.opponent_moves == []  # alpha spoke last
    # beta only ever sees alpha's replies, never its decision
    seen_by_beta = [
        move for observation, _ in beta_log for move in observation.opponent_moves
    ]
    assert choose("renege") not in seen_by_beta


def test_imposed_venture_skips_signing_and_the_shock_is_certain():
    bearers = set()
    for seed in range(20):
        state = run(
            config(seed=seed, imposed_venture=True, shock_prob=0.5),
            alpha=scripted(choose("honor")),
            beta=scripted(choose("honor")),
        )
        assert [record.stage for record in state.rounds] == ["decision"]
        assert state.contract.shock is True
        assert (state.contract.alpha_bears_prob, state.contract.transfer) == (0.5, 0.0)
        bearers.add(state.contract.bearer)
    assert bearers == {"alpha", "beta"}  # a fair coin picks the bearer


def test_transfers_above_the_cap_are_rejected():
    contexts = []

    async def alpha(observation, context):
        contexts.append(context)
        return propose(0.5, 15.0) if len(contexts) == 1 else propose(0.5, 10.0)

    state = run(
        config(transfer_cap=10.0, shock_prob=0.0),
        alpha=alpha,
        beta=scripted(choose("accept")),
    )
    assert "10" in contexts[1].error
    assert state.contract.transfer == 10.0


def test_bargaining_without_agreement_ends_at_the_horizon_with_no_contract():
    def stubborn():
        async def play(observation, context):
            if context.role == "proposer":
                return propose(0.5, 0.0)
            return choose("reject")

        return play

    state = run(config(max_rounds=4), alpha=stubborn(), beta=stubborn())
    assert len(state.rounds) == 4
    assert state.ending == "horizon"
    assert state.payoffs == {"alpha": 0.0, "beta": 0.0}


def broken():
    async def play(observation, context):
        raise InvalidMove("no tool call")

    return play


def test_default_moves_for_a_player_that_keeps_failing():
    # proposer with no earlier offer: the fair coin without transfer
    state = run(config(shock_prob=0.0), alpha=broken(), beta=scripted(choose("accept")))
    assert state.rounds[0].moves["alpha"] == propose(0.5, 0.0)
    assert state.rounds[0].defaulted == {"alpha"}
    # responder rejects
    state = run(config(max_rounds=1), alpha=scripted(propose(0.5, 0.0)), beta=broken())
    assert state.rounds[0].moves["beta"] == choose("reject")


def test_default_bearer_reneges_and_default_messages_are_empty():
    async def alpha(observation, context):  # the bearer: replies, then cannot decide
        if context.role == "proposer":
            return propose(1.0, 20.0)
        if context.action_kind == "message":
            return Move(message="Hello")
        raise InvalidMove("no tool call")

    async def beta(observation, context):  # the other: signs, then cannot speak
        if context.role == "responder":
            return choose("accept")
        raise InvalidMove("no tool call")

    state = run(config(shock_prob=1.0, persuasion_rounds=1), alpha=alpha, beta=beta)
    persuasion, decision = state.rounds[1], state.rounds[2]
    assert persuasion.moves["beta"] == Move(message="")
    assert decision.moves["alpha"] == choose("renege")
    assert state.ending == "reneged"


def test_proposer_that_breaks_later_repeats_its_last_offer():
    async def alpha(observation, context):
        if context.round == 1:
            return propose(0.7, 5.0)
        if context.role == "responder":
            return choose("reject")
        raise InvalidMove("no tool call")

    async def beta(observation, context):
        return propose(0.5, 0.0) if context.role == "proposer" else choose("reject")

    state = run(config(max_rounds=3), alpha=alpha, beta=beta)
    assert state.rounds[2].moves["alpha"] == propose(0.7, 5.0)


def test_the_proposer_may_walk_away_too():
    state = run(config(), alpha=scripted(choose("walk_away")), beta=scripted())
    assert state.ending == "walked_away"


def test_a_player_can_promise_to_honor_while_signing():
    talk = config(shock_prob=0.0).model_copy(
        update={
            "treatments": Treatments(
                horizon=HorizonTreatment(known=True),
                talk=TalkTreatment(
                    messages=False, declared_action=True, declared_type=False
                ),
            )
        }
    )
    promise = choose("accept", declared_action=Choice(kind="honor"))
    state = run(talk, alpha=scripted(propose(0.5, 20.0)), beta=scripted(promise))
    assert state.rounds[0].moves["beta"] == promise
