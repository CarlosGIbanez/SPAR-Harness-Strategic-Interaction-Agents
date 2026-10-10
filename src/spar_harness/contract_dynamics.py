"""Rules of the contract dynamics game (commitment problem, Powell 2006)."""

import random
from typing import Literal

from spar_harness.schema import (
    Choice,
    ContractBriefing,
    ContractOffer,
    ContractParams,
    ContractState,
    Ending,
    EpisodeConfig,
    EpisodeState,
    GameSpec,
    Move,
    PhaseContext,
    PhaseSpec,
    PlayerId,
    StageSpec,
    TerminationSpec,
)


def temptation(params: ContractParams, transfer: float) -> float:
    """What the bearer gains by reneging; it honors iff this is at most zero."""
    return params.loss - transfer - params.disaster - params.penalty


def outcome_payoffs(
    ending: Ending,
    params: ContractParams,
    bearer: PlayerId | None = None,
    transfer: float = 0.0,
) -> dict[PlayerId, float]:
    """Each player's payoff, relative to the status quo of 0."""
    benefits = params.benefits
    if ending in ("walked_away", "horizon"):
        return {"alpha": 0.0, "beta": 0.0}
    if ending == "no_shock":
        return dict(benefits)
    other = _other(bearer)
    if ending == "honored":
        return {
            bearer: benefits[bearer] - params.loss + transfer,
            other: benefits[other] - transfer,
        }
    return {
        bearer: benefits[bearer] - params.disaster - params.penalty,
        other: benefits[other] - params.disaster,
    }


def contract_value(
    player: PlayerId,
    params: ContractParams,
    alpha_bears_prob: float,
    transfer: float,
    bearer_honors: bool,
) -> float:
    """The player's expected payoff from signing, before the shock is drawn."""
    bears_prob = alpha_bears_prob if player == "alpha" else 1 - alpha_bears_prob
    if bearer_honors:
        expected_cost = (
            bears_prob * (params.loss - transfer) + (1 - bears_prob) * transfer
        )
    else:
        expected_cost = params.disaster + bears_prob * params.penalty
    return params.benefits[player] - params.shock_prob * expected_cost


def equal_gains_share(params: ContractParams, transfer: float) -> float:
    """The chance alpha bears the cost that gives both sides equal honored values.

    Equal values need b_a - b_b = p * (2q - 1) * (L - 2t). Without risk (p = 0) or
    at t = L / 2, who bears no longer matters, so the fair coin is returned.
    """
    spread = params.loss - 2 * transfer
    if params.shock_prob == 0 or spread == 0:
        return 0.5
    gap = params.benefits["alpha"] - params.benefits["beta"]
    share = 0.5 + gap / (2 * params.shock_prob * spread)
    return min(max(share, 0.0), 1.0)


def credible_transfers(params: ContractParams) -> tuple[float, float] | None:
    """Transfers the bearer will honor and the other prefers to disaster, or None.

    The bearer honors iff t >= L - D - F; the other prefers paying to disaster
    iff t <= D; the regime allows t <= transfer_cap.
    """
    low = max(0.0, params.loss - params.disaster - params.penalty)
    high = min(params.disaster, params.transfer_cap)
    return (low, high) if low <= high else None


def deal_exists(params: ContractParams) -> bool:
    """Whether some credible contract leaves both sides strictly better off.

    An honored contract's total value S = b_a + b_b - p * L is the same for every
    (q, t), so beta gets S minus alpha's value. Alpha's value is bilinear in
    (q, t), so over the box of allowed contracts it takes every value between its
    corner values; a deal exists iff that interval meets (0, S).
    """
    transfers = credible_transfers(params)
    if transfers is None:
        return False
    total = sum(params.benefits.values()) - params.shock_prob * params.loss
    alpha_values = [
        contract_value("alpha", params, q, t, bearer_honors=True)
        for q in (0.0, 1.0)
        for t in transfers
    ]
    return total > 0 and max(alpha_values) > 0 and min(alpha_values) < total


def contract_dynamics_game(
    params: ContractParams, max_rounds: int, persuasion_rounds: int
) -> GameSpec:
    """Bargain over a contract, let nature draw the shock, persuade, then decide.

    The imposed venture (params.imposed_venture) skips the bargaining stage.
    """
    bargaining = StageSpec(
        name="bargaining",
        phases=[
            PhaseSpec(kind="act", actors="proposer", action_kind="offer"),
            PhaseSpec(kind="act", actors="responder", action_kind="response"),
        ],
        termination=TerminationSpec(max_rounds=max_rounds, stop_prob=0.0),
        when_exhausted="horizon",
    )
    persuasion = StageSpec(
        name="persuasion",
        phases=[
            PhaseSpec(kind="act", actors="other", action_kind="message"),
            PhaseSpec(kind="act", actors="bearer", action_kind="message"),
        ],
        termination=TerminationSpec(max_rounds=persuasion_rounds, stop_prob=0.0),
        when_exhausted="next_stage",
    )
    decision = StageSpec(
        name="decision",
        phases=[PhaseSpec(kind="act", actors="bearer", action_kind="decision")],
        termination=TerminationSpec(max_rounds=1, stop_prob=0.0),
        when_exhausted="horizon",  # unreachable: honor and renege both end the game
    )
    stages = [persuasion, decision]
    if not params.imposed_venture:
        stages.insert(0, bargaining)
    return GameSpec(game_id="contract_dynamics", stages=stages, contract=params)


ALLOWED_ACTIONS = {
    "offer": {"propose", "walk_away"},
    "response": {"accept", "reject", "walk_away"},
    "message": set(),
    "decision": {"honor", "renege"},
}


def briefing(
    config: EpisodeConfig, state: EpisodeState, player: PlayerId
) -> ContractBriefing:
    stages = {stage.name: stage for stage in config.game.stages}
    bargaining = stages.get("bargaining")
    return ContractBriefing(
        params=config.game.contract,
        max_rounds=bargaining.termination.max_rounds if bargaining else 0,
        persuasion_rounds=stages["persuasion"].termination.max_rounds,
    )


def ending_after_move(move: Move) -> Ending | Literal["next_stage"] | None:
    if move.action is None:
        return None
    return {
        "accept": "next_stage",
        "walk_away": "walked_away",
        "honor": "honored",
        "renege": "reneged",
    }.get(move.action.kind)


def default_move(
    game: GameSpec, state: EpisodeState, player: PlayerId, context: PhaseContext
) -> Move:
    """The move made for a player who keeps failing."""
    if context.action_kind == "message":
        return Move(message="")
    if context.action_kind == "decision":
        return Move(action=Choice(kind="renege"))
    if context.role == "responder":
        return Move(action=Choice(kind="reject"))
    for record in reversed(state.rounds):
        own_move = record.moves.get(player)
        if own_move and own_move.action and own_move.action.kind == "propose":
            return own_move
    return Move(action=ContractOffer(alpha_bears_prob=0.5, transfer=0.0))


def move_problem(game: GameSpec, move: Move) -> str | None:
    """Why a move breaks this game's rules, or None if it does not."""
    action = move.action
    cap = game.contract.transfer_cap
    if action is not None and action.kind == "propose" and action.transfer > cap:
        return f"the transfer may be at most {cap:g}"
    return None


def before_stage(
    game: GameSpec, state: EpisodeState, stage: StageSpec, rng: random.Random
) -> Ending | None:
    """Nature's move before persuasion: record the contract, draw shock and bearer."""
    if stage.name != "persuasion":
        return None
    if game.contract.imposed_venture:
        state.contract = ContractState(alpha_bears_prob=0.5, transfer=0.0, shock=True)
    else:
        signing_round = state.rounds[-1]
        terms = signing_round.moves[signing_round.proposer].action
        state.contract = ContractState(
            alpha_bears_prob=terms.alpha_bears_prob,
            transfer=terms.transfer,
            shock=rng.random() < game.contract.shock_prob,
        )
        if not state.contract.shock:
            return "no_shock"
    alpha_bears = rng.random() < state.contract.alpha_bears_prob
    state.contract.bearer = "alpha" if alpha_bears else "beta"
    return None


def actor_for(role: Literal["bearer", "other"], state: EpisodeState) -> PlayerId:
    bearer = state.contract.bearer
    return bearer if role == "bearer" else _other(bearer)


def episode_payoffs(game: GameSpec, state: EpisodeState) -> dict[PlayerId, float]:
    contract = state.contract
    return outcome_payoffs(
        state.ending,
        game.contract,
        bearer=contract.bearer if contract else None,
        transfer=contract.transfer if contract else 0.0,
    )


def _other(player: PlayerId) -> PlayerId:
    return "beta" if player == "alpha" else "alpha"
