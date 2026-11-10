"""Scripted labs. Each bot is a Player and sees only what a model player sees."""

import random

from spar_harness.f3_referee import Player
from spar_harness.f3_schema import (
    Act,
    ActMove,
    Move,
    NegotiationMove,
    Observation,
    PhaseContext,
)


def _negotiate(
    observation: Observation, context: PhaseContext, duration: int | None
) -> Move:
    """Propose `duration` (None = no agreement); accept any offered agreement."""
    options = context.options
    if options == ["propose"]:
        return NegotiationMove(kind="propose", duration=duration)
    if "accept" in options and duration is not None:
        return NegotiationMove(kind="accept")
    return NegotiationMove(kind="reject")


def always_accelerate(duration: int | None = None) -> Player:
    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.phase == "negotiation":
            return _negotiate(observation, context, duration)
        return ActMove(action="accelerate")

    return play


def always_slow(duration: int | None = 6) -> Player:
    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.phase == "negotiation":
            return _negotiate(observation, context, duration)
        return ActMove(action="slow")

    return play


def secret_defector(duration: int | None = 6) -> Player:
    """Signs, declares Slow down, and accelerates in every round."""

    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.phase == "negotiation":
            return _negotiate(observation, context, duration)
        return ActMove(
            action="slow" if context.phase == "declaration" else "accelerate"
        )

    return play


def tit_for_tat(duration: int | None = 6) -> Player:
    """Slows down in round 1, then copies what it believes the rival did last
    round: the verification report if it has one, else the rival's last declaration."""
    last_declaration: Act = "slow"

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal last_declaration
        if context.phase == "negotiation":
            return _negotiate(observation, context, duration)
        belief: Act = observation.last_report or last_declaration
        if context.phase == "action":
            last_declaration = observation.rival_declaration or "slow"
        return ActMove(action=belief)

    return play


def random_bot(seed: int, duration: int | None = None) -> Player:
    rng = random.Random(seed)

    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.phase == "negotiation":
            kind = rng.choice(context.options)
            if kind in ("propose", "counter"):
                n = rng.choice([None, *range(1, context.max_duration + 1)])
                if kind == "counter" and (n is None or n == context.pending_offer):
                    return NegotiationMove(kind="reject")
                return NegotiationMove(kind=kind, duration=n)
            return NegotiationMove(kind=kind)
        return ActMove(action=rng.choice(["slow", "accelerate"]))

    return play


def scripted(strategy) -> Player:
    """Plays `strategy(round_index) -> Act` (round_index counts from 0) in both
    phases and never agrees to anything. Used to evaluate fixed strategies."""

    async def play(observation: Observation, context: PhaseContext) -> Move:
        if context.phase == "negotiation":
            if context.options == ["propose"]:
                return NegotiationMove(kind="propose")
            return NegotiationMove(kind="reject")
        return ActMove(action=strategy(observation.round - 1))

    return play
