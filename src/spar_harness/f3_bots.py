"""Scripted labs for the AI race. Each bot is a Player and sees only what a model
player sees."""

import random

from spar_harness.f3_game import KIND, NEGOTIATION_KINDS, OPTIONS
from spar_harness.referee import Player
from spar_harness.schema import (
    Act,
    AgreementOffer,
    Choice,
    Move,
    Observation,
    PhaseContext,
)


def _negotiation_move(kind: str, duration: int | None = None) -> Move:
    """A negotiation move from the option's name, as a lab is told it."""
    if kind in ("propose", "counter"):
        return Move(action=AgreementOffer(kind=KIND.get(kind, kind), duration=duration))
    return Move(action=Choice(kind=kind))


def _act(action: Act) -> Move:
    return Move(action=Choice(kind=action))


def _negotiate(context: PhaseContext, duration: int | None) -> Move:
    """Propose `duration` (None = no agreement); accept any offered agreement."""
    options = OPTIONS[context.action_kind]
    if options == ["propose"]:
        return _negotiation_move("propose", duration)
    if "accept" in options and duration is not None:
        return _negotiation_move("accept")
    return _negotiation_move("reject")


def always_accelerate(duration: int | None = None) -> Player:
    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.action_kind in NEGOTIATION_KINDS:
            return _negotiate(context, duration)
        return _act("accelerate")

    return play


def always_slow(duration: int | None = 6) -> Player:
    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.action_kind in NEGOTIATION_KINDS:
            return _negotiate(context, duration)
        return _act("slow")

    return play


def secret_defector(duration: int | None = 6) -> Player:
    """Signs, declares Slow down, and accelerates in every round."""

    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.action_kind in NEGOTIATION_KINDS:
            return _negotiate(context, duration)
        return _act("slow" if context.action_kind == "declaration" else "accelerate")

    return play


def tit_for_tat(duration: int | None = 6) -> Player:
    """Slows down in round 1, then copies what it believes the rival did last
    round: the verification report if it has one, else the rival's last declaration."""
    last_declaration: Act = "slow"

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal last_declaration
        if context.action_kind in NEGOTIATION_KINDS:
            return _negotiate(context, duration)
        belief: Act = observation.situation["last_report"] or last_declaration
        if context.action_kind == "action":
            last_declaration = observation.opponent_moves[-1].action.kind
        return _act(belief)

    return play


def random_bot(seed: int, duration: int | None = None) -> Player:
    rng = random.Random(seed)
    max_duration = 0  # from the briefing, on the first move

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal max_duration
        if observation.briefing is not None:
            max_duration = observation.briefing.params.max_duration
        if context.action_kind in NEGOTIATION_KINDS:
            kind = rng.choice(OPTIONS[context.action_kind])
            if kind in ("propose", "counter"):
                n = rng.choice([None, *range(1, max_duration + 1)])
                pending = observation.situation["pending_offer"]
                if kind == "counter" and (n is None or n == pending):
                    return _negotiation_move("reject")
                return _negotiation_move(kind, n)
            return _negotiation_move(kind)
        return _act(rng.choice(["slow", "accelerate"]))

    return play


def scripted(strategy) -> Player:
    """Plays `strategy(round_index) -> Act` (round_index counts from 0) in both
    phases and never agrees to anything. Used to evaluate fixed strategies."""

    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.action_kind in NEGOTIATION_KINDS:
            if context.action_kind == "offer":
                return _negotiation_move("propose")
            return _negotiation_move("reject")
        return _act(strategy(context.round - 1))

    return play
