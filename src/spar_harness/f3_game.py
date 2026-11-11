"""Rules of the AI race game (F3: commitment and verification), as plugins of the
shared referee: type draws, negotiation, rounds, risk, payoffs, matched draws.

An episode has two stages. "negotiation": a lab chosen at random proposes an
agreement of N rounds (or none), the other accepts, rejects or counters once,
and after a counter the first accepts or rejects. "race": up to max_rounds
rounds. With params.renegotiate, a round in which no agreement is in force
opens with a new negotiation, the proposer alternating. Then both labs declare
Slow down or Accelerate, then both act; AFTER_ROUND applies enforcement, the
update, the risk, the payoffs, the verification signals and the catastrophe
and stopping draws.

The labs are alpha and beta inside the harness and Lab A and Lab B in the text.
Every random draw comes from a stream named by (purpose, seed, keys), so all
conditions with the same seed see the same draws whatever the labs do.
"""

import math
import random
from typing import Literal

from spar_harness.schema import (
    Act,
    AgreementOffer,
    Choice,
    EpisodeConfig,
    EpisodeState,
    GameSpec,
    LabType,
    Move,
    PhaseContext,
    PhaseSpec,
    PlayerId,
    RaceBriefing,
    RaceNegotiation,
    RaceParams,
    RaceRound,
    RaceSpec,
    RaceState,
    Report,
    RoundEnd,
    RoundRecord,
    StageSpec,
    TerminationSpec,
)

PLAYERS: tuple[PlayerId, PlayerId] = ("alpha", "beta")
LABEL: dict[PlayerId, str] = {"alpha": "A", "beta": "B"}  # the labs' names in text
NEGOTIATION_KINDS = ("offer", "response", "final")
# What a lab may do in each negotiation phase, in the order it is told. The
# model sees "propose"; the move it makes is an AgreementOffer.
OPTIONS: dict[str, list[str]] = {
    "offer": ["propose"],
    "response": ["accept", "reject", "counter"],
    "final": ["accept", "reject"],
}
KIND = {"propose": "propose_duration"}  # the move kind of an option, where it differs
ALLOWED_ACTIONS = {
    "offer": {"propose_duration"},
    "response": {"accept", "reject", "counter"},
    "final": {"accept", "reject"},
    "declaration": {"slow", "accelerate"},
    "action": {"slow", "accelerate"},
}


def rival_of(lab: PlayerId) -> PlayerId:
    return "beta" if lab == "alpha" else "alpha"


def rng(purpose: str, seed: int, *keys: object) -> random.Random:
    """A random stream fixed by (purpose, seed, keys), so that every condition
    sees the same draws (matched seeds) whatever the labs do."""
    return random.Random("-".join([purpose, str(seed), *map(str, keys)]))


def draw_types(seed: int, spec: RaceSpec) -> dict[PlayerId, LabType]:
    """Private k and c, uniform on the ranges, rounded to 2 decimals so that the
    numbers in the prompts are exactly the true values."""
    if spec.fixed_types is not None:
        return spec.fixed_types
    p, draw = spec.params, rng("types", seed)
    return {
        lab: LabType(
            k=round(draw.uniform(p.k_low, p.k_high), 2),
            c=round(draw.uniform(p.c_low, p.c_high), 2),
        )
        for lab in PLAYERS
    }


def draw_proposer(seed: int) -> PlayerId:
    return rng("proposer", seed).choice(list(PLAYERS))


def risk(
    capability: dict[PlayerId, float],
    threshold: float,
    kappa: float,
    background: float = 0.0,
) -> tuple[float, float]:
    """Frontier distance D and catastrophe probability p = 1 - exp(-(kappa D + b)),
    where b is the accumulated background hazard (zero unless switched on)."""
    distance = max(0.0, max(capability.values()) - threshold)
    return distance, 1.0 - math.exp(-(kappa * distance + background))


def background_step(seed: int, params: RaceParams, round_number: int) -> float:
    """This round's random increase of the background hazard, the same in every condition."""
    return rng("background", seed, round_number).uniform(0.0, params.background_max)


def round_payoff(
    lab: PlayerId,
    capability: dict[PlayerId, float],
    action: Act,
    types: dict[PlayerId, LabType],
    a: float,
) -> float:
    """u = 1 + a (own - rival capability) - c (1 if slowed down), after the update."""
    lead = capability[lab] - capability[rival_of(lab)]
    return 1.0 + a * lead - (types[lab].c if action == "slow" else 0.0)


def report_about(
    verification: str,
    actual: Act,
    q: float,
    seed: int,
    round_number: int,
    receiver: PlayerId,
) -> Act | None:
    """The signal the receiver gets about the rival's actual action."""
    if verification == "none":
        return None
    draw = rng("signal", seed, round_number, LABEL[receiver])
    if verification == "full" or draw.random() < q:
        return actual
    return "slow" if actual == "accelerate" else "accelerate"


def mutual_slowdown_benchmark(
    types: dict[PlayerId, LabType], params: RaceParams
) -> float:
    """Joint payoff if both labs slow down in every round, expected over the
    stopping rule (no catastrophe, since mutual slowdown keeps the frontier at
    the threshold)."""
    weight = sum(params.continue_prob**t for t in range(params.max_rounds))
    return weight * sum(1.0 - types[lab].c for lab in PLAYERS)


def in_window(race: RaceState, round_number: int) -> bool:
    """Whether an agreement is in force in this round."""
    return round_number <= race.window_end


# the game


def ai_race_game(spec: RaceSpec) -> GameSpec:
    negotiation = [
        PhaseSpec(kind="act", actors="proposer", action_kind="offer"),
        PhaseSpec(kind="act", actors="responder", action_kind="response"),
        PhaseSpec(kind="act", actors="proposer", action_kind="final"),
    ]
    return GameSpec(
        game_id="ai_race",
        stages=[
            StageSpec(
                name="negotiation",
                phases=negotiation,
                termination=TerminationSpec(max_rounds=1, stop_prob=0.0),
                proposer_rule="game",
                when_exhausted="next_stage",
            ),
            StageSpec(
                name="race",
                phases=[
                    *negotiation,  # played only to renegotiate (OPEN_PHASES)
                    PhaseSpec(kind="act", actors="both", action_kind="declaration"),
                    PhaseSpec(kind="act", actors="both", action_kind="action"),
                ],
                # the stopping draw is the game's own stream, made in after_round
                termination=TerminationSpec(
                    max_rounds=spec.params.max_rounds, stop_prob=0.0
                ),
                proposer_rule="game",
                when_exhausted="horizon",
            ),
        ],
        race=spec,
    )


def start(config: EpisodeConfig, state: EpisodeState) -> None:
    state.race = RaceState(
        seed=config.seed,
        types=draw_types(config.seed, config.game.race),
        proposer=draw_proposer(config.seed),
    )


def briefing(
    config: EpisodeConfig, state: EpisodeState, player: PlayerId
) -> RaceBriefing:
    spec, race = config.game.race, state.race
    return RaceBriefing(
        me=player,
        k=race.types[player].k,
        c=race.types[player].c,
        params=spec.params,
        condition=spec.condition,
        wording=spec.wording,
        proposes_first=race.proposer == player,
    )


def proposer(state: EpisodeState) -> PlayerId:
    return state.race.proposer


def phase_open(
    game: GameSpec, state: EpisodeState, record: RoundRecord, phase: PhaseSpec
) -> bool:
    """The opening offer always, a renegotiation only outside an agreement, and
    the replies only while a negotiation is under way."""
    race = state.race
    if phase.action_kind == "offer":
        if record.stage == "negotiation":
            return True
        return (
            game.race.params.renegotiate
            and record.round > 1
            and not in_window(race, record.round)
        )
    if phase.action_kind in ("response", "final"):
        return race.negotiation is not None
    return True


def after_phase(
    game: GameSpec,
    state: EpisodeState,
    record: RoundRecord,
    phase: PhaseSpec,
    moves: dict[PlayerId, Move],
    defaulted: list[PlayerId],
) -> None:
    """Keep every move: negotiations move by move, and the round's declarations."""
    race = state.race
    if phase.action_kind == "declaration":
        race.declared = {lab: move.action.kind for lab, move in moves.items()}
        race.round_defaulted = list(defaulted)
        return
    if phase.action_kind == "action":
        race.round_defaulted.extend(defaulted)
        return
    [(lab, move)] = moves.items()
    if phase.action_kind == "offer":
        round_number = 0 if record.stage == "negotiation" else record.round
        race.negotiation = RaceNegotiation(proposer=lab, round=round_number)
    negotiation = race.negotiation
    negotiation.moves.append((lab, move))
    negotiation.defaulted.extend(defaulted)
    kind = move.action.kind
    if phase.action_kind == "offer" and move.action.duration is None:
        _conclude(race, None)
    elif kind == "accept":
        _conclude(race, negotiation.moves[-2][1].action.duration)
    elif kind == "reject":
        _conclude(race, None)
    # a counter leaves the negotiation open for the proposer's final answer


def _conclude(race: RaceState, duration: int | None) -> None:
    """End the negotiation under way; an agreement starts with the next round."""
    negotiation = race.negotiation
    negotiation.duration = duration
    race.negotiations.append(negotiation)
    race.negotiation = None
    race.proposer = rival_of(negotiation.proposer)
    if duration is not None:
        first = 1 if negotiation.round == 0 else negotiation.round
        race.duration = duration
        race.window_end = first + duration - 1


def situation(
    game: GameSpec,
    state: EpisodeState,
    record: RoundRecord,
    phase: PhaseSpec,
    player: PlayerId,
) -> Report:
    """Where the game stands for a lab about to move."""
    race = state.race
    pending = None  # the duration on the table
    if race.negotiation is not None and phase.action_kind in ("response", "final"):
        pending = race.negotiation.moves[-1][1].action.duration
    if record.stage == "negotiation":
        return {"round": 0, "pending_offer": pending}
    t = record.round
    agreement = in_window(race, t)
    return {
        "round": t,
        "own_capability": race.capability[player],
        "threshold": race.threshold,
        "agreement_duration": race.duration if agreement else None,
        "rounds_left": max(0, race.window_end - t + 1) if agreement else None,
        "agreement_expired": race.duration is not None and not agreement,
        "last_report": race.last_report[player],
        "own_overridden": race.overridden[player],
        "pending_offer": pending,
    }


def move_problem(
    game: GameSpec, state: EpisodeState, player: PlayerId, move: Move
) -> str | None:
    """The negotiation rules, in the order they are checked."""
    action, max_n = move.action, game.race.params.max_duration
    if isinstance(action, AgreementOffer):
        needs_n = action.kind == "counter" or action.duration is not None
        if needs_n and not (
            action.duration is not None and 1 <= action.duration <= max_n
        ):
            return f"The duration N must be a whole number from 1 to {max_n}."
        negotiation = state.race.negotiation
        pending = negotiation.moves[-1][1].action.duration if negotiation else None
        if action.kind == "counter" and action.duration == pending:
            return "A counter-proposal must differ from the offer; accept it instead."
    stated = move.declared_type
    if stated is not None:
        for value in (stated.k, stated.c):
            if value is not None and value <= 0:
                return "Stated k and c must be positive."
    return None


def ending_after_move(move: Move) -> Literal["next_stage"] | None:
    return None  # rounds end in after_round


def default_move(
    game: GameSpec, state: EpisodeState, player: PlayerId, context: PhaseContext
) -> Move:
    """No agreement for a proposer, a rejection for a reply, Slow down in a round."""
    if context.action_kind == "offer":
        return Move(action=AgreementOffer(kind="propose_duration"))
    if context.action_kind in ("response", "final"):
        return Move(action=Choice(kind="reject"))
    return Move(action=Choice(kind="slow"))


def after_round(game: GameSpec, state: EpisodeState, record: RoundRecord) -> RoundEnd:
    """Enforcement, the update, risk, payoffs, verification and the draws."""
    if record.stage != "race":
        return RoundEnd()
    race, spec, t = state.race, game.race, record.round
    p, seed = spec.params, race.seed
    attempted: dict[PlayerId, Act] = {
        lab: record.moves[lab].action.kind for lab in PLAYERS
    }
    enforced = spec.condition.commitment == "enforced" and in_window(race, t)
    overridden = {lab: enforced and attempted[lab] == "accelerate" for lab in PLAYERS}
    actual: dict[PlayerId, Act] = {
        lab: "slow" if overridden[lab] else attempted[lab] for lab in PLAYERS
    }
    for lab in PLAYERS:
        if actual[lab] == "accelerate":
            race.capability[lab] += race.types[lab].k
    race.threshold += p.sigma
    race.background += background_step(seed, p, t)
    distance, probability = risk(
        race.capability, race.threshold, p.kappa, race.background
    )
    payoff = {
        lab: round_payoff(lab, race.capability, actual[lab], race.types, p.a)
        for lab in PLAYERS
    }
    catastrophe = rng("catastrophe", seed, t).random() < probability
    report = {
        lab: report_about(
            spec.condition.verification, actual[rival_of(lab)], p.q, seed, t, lab
        )
        for lab in PLAYERS
    }
    race.last_report = report
    race.overridden = overridden
    race.rounds.append(
        RaceRound(
            round=t,
            in_window=in_window(race, t),
            declared=race.declared,
            attempted=attempted,
            actual=actual,
            overridden=overridden,
            report=report,
            capability=dict(race.capability),
            threshold=race.threshold,
            distance=distance,
            background=race.background,
            risk=probability,
            payoff=payoff,
            catastrophe=catastrophe,
            defaulted=race.round_defaulted,
        )
    )
    if catastrophe:
        return RoundEnd(outcome="catastrophe")
    if t == p.max_rounds:
        return RoundEnd()  # the stage runs out: horizon
    if rng("stop", seed, t).random() >= p.continue_prob:
        return RoundEnd(outcome="random_stop")
    return RoundEnd()


def episode_payoffs(game: GameSpec, state: EpisodeState) -> dict[PlayerId, float]:
    return {lab: sum(r.payoff[lab] for r in state.race.rounds) for lab in PLAYERS}
