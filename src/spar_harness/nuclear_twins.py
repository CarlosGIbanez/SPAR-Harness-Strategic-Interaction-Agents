"""Twins for F2 (steps 2 and 3): programs without AI that move the way a model
moved against the bots, and play each other.

A twin's situation is what the model knew when it chose its move: the rung it
stood on, the rung the opponent stood on, the opponent's last move, and whether
that move went above its announcement (and was explained). Versions add memory
one feature at a time: T1 its own last move, T2 how many times the opponent
exceeded its announcement, T3 how far below the game's highest rung it stands.
A multinomial logistic regression on the model's step-1 decisions gives the
probability of each move class (-2 or less, -1, 0, +1, +2, +3 or more); in the
end classes the exact size is drawn from the model's own moves. A twin announces
honestly and claims its true worth, so its announced and executed rungs differ
only through accidents. It doesn't know about its own accidents: in step 1 the
model had none.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from spar_harness.nuclear_crisis import RETURN_TO_START, STRATEGIC_WAR
from spar_harness.referee import Player
from spar_harness.schema import (
    EpisodeConfig,
    EpisodeState,
    Move,
    Observation,
    PhaseContext,
    PlayerId,
    Report,
    Rung,
)

CLASSES = (-2, -1, 0, 1, 2, 3)  # the end classes stand for -2 or less, +3 or more
VERSIONS = ("T0", "T1", "T2", "T3")
COPY = len(CLASSES)  # the extra class of a copying twin: the opponent's last rung
HIGHEST = STRATEGIC_WAR - 1  # no model chose Strategic Nuclear War in the pilot
BAND_TOPS = (RETURN_TO_START, 13, 17, 20, 24, 27, STRATEGIC_WAR)


@dataclass(frozen=True)
class Situation:
    own: int  # the rung the side stood on: its executed rung last round
    opponent: int  # the opponent's executed rung last round
    opponent_move: int  # how far the opponent moved last round
    jump: bool  # the opponent executed above its announcement last round
    explained: bool  # ... and said it was an accident
    own_move: int = 0  # T1: how far the side itself chose to move last round
    exceeded: int = 0  # T2: rounds in which the opponent went above its announcement
    below_peak: int = 0  # T3: the game's highest executed rung minus own


def move_class(delta: int) -> int:
    """The index in CLASSES of a move of `delta` rungs."""
    return min(max(delta, CLASSES[0]), CLASSES[-1]) - CLASSES[0]


def situation_after(reports: Sequence[Report], history: Sequence[dict] = (), me=None):
    """The situation before the next move, from the side's reports so far and,
    for a game that started with a history, that history."""
    them = None if me is None else ("beta" if me == "alpha" else "alpha")
    if not reports:
        last, before = history[-1], (history[-2] if len(history) > 1 else history[-1])
        peak = max(rung for rungs in history for rung in rungs.values())
        return Situation(
            own=last[me],
            opponent=last[them],
            opponent_move=last[them] - before[them],
            jump=False,
            explained=False,
            own_move=last[me] - before[me],
            exceeded=0,
            below_peak=peak - last[me],
        )
    last = reports[-1]
    opponent_before = last["opponent_previous"]
    own_before = last["previous"]
    exceeded = sum(
        1
        for report in reports
        if report["opponent_announced"] is not None
        and report["opponent_executed"] > report["opponent_announced"]
    )
    peak = max(
        [max(r["executed"], r["opponent_executed"]) for r in reports]
        + [rung for rungs in history for rung in rungs.values()]
    )
    return Situation(
        own=last["executed"],
        opponent=last["opponent_executed"],
        opponent_move=last["opponent_executed"]
        - (
            opponent_before
            if opponent_before is not None
            else last["opponent_executed"]
        ),
        jump=last["opponent_announced"] is not None
        and last["opponent_executed"] > last["opponent_announced"],
        explained=bool(last["opponent_explained"]),
        own_move=last["ordered"]
        - (own_before if own_before is not None else last["ordered"]),
        exceeded=exceeded,
        below_peak=peak - last["executed"],
    )


def decisions(
    state: EpisodeState, config: EpisodeConfig, player: PlayerId
) -> list[tuple[Situation, int]]:
    """The side's decisions as (situation, rungs moved), leaving out default
    moves and a first move made from scratch (see first_move)."""
    start = config.game.nuclear.start
    history = start.history if start is not None else []
    reports = [record.reports[player] for record in state.rounds]
    rows = []
    for number, record in enumerate(state.rounds):
        if player in record.defaulted:
            continue
        if number == 0 and not history:
            continue
        situation = situation_after(reports[:number], history, me=player)
        rows.append((situation, reports[number]["ordered"] - situation.own))
    return rows


def first_move(state: EpisodeState, config: EpisodeConfig, player: PlayerId):
    """The side's first chosen rung in a game from scratch, or None."""
    if config.game.nuclear.start is not None or not state.rounds:
        return None
    if player in state.rounds[0].defaulted:
        return None
    return state.rounds[0].moves[player].action.rung


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


def _bin(value: int, edges: Sequence[int]) -> int:
    """The number of edges the value exceeds or equals."""
    return sum(value >= edge for edge in edges)


def features(situation: Situation, version: str) -> np.ndarray:
    band = next(i for i, top in enumerate(BAND_TOPS) if situation.own <= top)
    gap = situation.opponent - situation.own
    row = [1.0]
    row += _one_hot(band, len(BAND_TOPS))
    row += _one_hot(_bin(gap, (-2, -1, 0, 1, 2, 3)), 7)  # <=-3 ... >=+3
    row += _one_hot(_bin(situation.opponent_move, (-2, 0, 1, 4)), 5)  # <=-3 ... >=+4
    row += [float(situation.jump), float(situation.explained)]
    if version in ("T1", "T2", "T3"):
        row += _one_hot(_bin(situation.own_move, (-2, 0, 1, 3)), 5)
    if version in ("T2", "T3"):
        row += _one_hot(min(situation.exceeded, 3), 4)
    if version == "T3":
        row += _one_hot(_bin(situation.below_peak, (1, 3, 6)), 4)
    return np.array(row)


@dataclass
class Policy:
    """A fitted twin: class weights, plus the model's own sizes of the end moves.

    A copying twin has one more class, COPY: take the opponent's last executed
    rung, whatever the gap. The models in the pilot did that in about two moves
    out of three, which delta classes alone reproduce poorly once the gap is 3 or
    more (the end-class size would be drawn from moves made at other gaps).
    """

    version: str
    weights: np.ndarray  # features x classes
    end_moves: dict[int, list[int]] = field(default_factory=dict)  # class -> deltas
    first_moves: list[int] = field(default_factory=list)
    copy: bool = False

    def probabilities(self, situation: Situation) -> np.ndarray:
        scores = features(situation, self.version) @ self.weights
        scores -= scores.max()
        exp = np.exp(scores)
        return exp / exp.sum()


def fit(
    rows: Sequence[tuple[Situation, int]],
    version: str,
    first_moves: Sequence[int] = (),
    copy: bool = False,
    l2: float = 1.0,
    steps: int = 3000,
    rate: float = 0.5,
) -> Policy:
    """Multinomial logistic regression of the move class on the situation, with a
    ridge penalty on everything but the intercept."""
    x = np.array([features(situation, version) for situation, _ in rows])

    def copied(situation, delta):
        return copy and situation.own + delta == situation.opponent

    y = np.array([COPY if copied(s, delta) else move_class(delta) for s, delta in rows])
    rows = [(s, delta) for s, delta in rows if not copied(s, delta)]
    n, d = x.shape
    k = len(CLASSES) + (1 if copy else 0)
    target = np.eye(k)[y]
    weights = np.zeros((d, k))
    penalty = np.ones((d, 1))
    penalty[0] = 0.0
    for _ in range(steps):
        scores = x @ weights
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        gradient = x.T @ (probs - target) / n + 2 * l2 / n * penalty * weights
        weights -= rate * gradient
    end_moves = {
        0: [delta for _, delta in rows if delta <= CLASSES[0]] or [CLASSES[0]],
        len(CLASSES) - 1: [delta for _, delta in rows if delta >= CLASSES[-1]]
        or [CLASSES[-1]],
    }
    return Policy(version, weights, end_moves, list(first_moves), copy)


def twin(policy: Policy, rng: random.Random) -> Player:
    """A player that moves like the policy's model, announcing honestly and
    claiming its true worth."""
    reports: list[Report] = []
    history: list[dict] = []

    async def play(observation: Observation, context: PhaseContext) -> Move:
        reports.extend(observation.reports)
        briefing = observation.briefing
        if briefing is not None and briefing.start is not None:
            history[:] = briefing.start.history
        if not reports and not history:
            rung = (
                rng.choice(policy.first_moves)
                if policy.first_moves
                else RETURN_TO_START + 1
            )
        else:
            situation = situation_after(reports, history, me=observation.recipient)
            probabilities = policy.probabilities(situation)
            index = rng.choices(range(len(probabilities)), weights=probabilities)[0]
            if index == COPY:
                delta = situation.opponent - situation.own
            elif index in policy.end_moves:
                delta = rng.choice(policy.end_moves[index])
            else:
                delta = CLASSES[index]
            rung = min(max(situation.own + delta, 1), HIGHEST)
        return Move(
            action=Rung(rung=rung),
            declared_action=Rung(rung=rung),
            declared_value=_worth(observation, reports),
        )

    return play


def _worth(observation: Observation, reports: Sequence[Report]) -> float | None:
    if reports:
        return float(reports[-1]["worth"])
    briefing = observation.briefing
    if briefing is None:
        return None
    return (
        briefing.oil_value * briefing.oil_fields
        + briefing.town_value * briefing.towns
        + briefing.pass_value
    )
