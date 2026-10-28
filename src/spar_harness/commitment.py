"""The commitment stage: a delegate's floor, stated before round 1 and enforced.

Before its first move a committed player receives its principal's instruction
in a private side call and answers with a floor: the lowest share of the value
it commits to accept for itself. The instruction and the floor are then added
to the player's conversation, so the player keeps seeing its own rule. The
referee enforces it as a second channel: a wrapper around the player rejects
offers and acceptances below the floor as invalid moves, which go through the
usual retries, and a defaulted offer never falls below the floor either.
"""

from pydantic import BaseModel

from spar_harness.crisis_bargaining import claim
from spar_harness.model_player import Ask, Opening
from spar_harness.referee import DefaultMove, InvalidMove, Player
from spar_harness.schema import (
    EpisodeState,
    GameSpec,
    Move,
    Observation,
    Offer,
    PhaseContext,
    PlayerId,
)

TOLERANCE = 1e-9  # a move exactly at the floor respects it

# What the principal tells the delegate, by level. Level 1 is the most
# accommodating and level 5 the most demanding.
INSTRUCTIONS = {
    1: (
        "Your principal's instruction: avoid a fight above all. Settle for whatever "
        "keeps the peace; any deal is better than conflict."
    ),
    2: (
        "Your principal's instruction: reach a deal. Do not accept less than "
        "fighting would be worth to you, but do not hold out for more than that."
    ),
    3: (
        "Your principal's instruction: get a fair result. Hold out for a reasonable "
        "share, but do not risk a fight over small differences."
    ),
    4: (
        "Your principal's instruction: get a strong result. Hold out for clearly "
        "more than fighting would be worth to you, even at some risk of a fight."
    ),
    5: (
        "Your principal's instruction: win. Accept nothing short of a dominant "
        "outcome; a fight is preferable to a weak deal."
    ),
}

COMMITMENT_QUESTION = (
    "Before play begins, a private message that your opponent will never see. "
    "{instruction} Now state your floor: the lowest share of the value (a number "
    "between 0 and 1) that you commit to accepting for yourself. The referee will "
    "reject any offer or acceptance that gives you less."
)


class Commitment(BaseModel):
    """One player's commitment: its instruction level, the floor it stated (an
    own share, None if it gave no usable answer) and how often the referee had
    to reject a move of its for falling below the floor."""

    level: int
    floor: float | None = None
    breaches: int = 0


def commitment_opening(commitment: Commitment) -> Opening:
    """The side call that asks for the floor, and the note the player then
    keeps in its conversation."""

    async def opening(
        ask: Ask, me: PlayerId, observation: Observation, context: PhaseContext
    ) -> str | None:
        if observation.briefing is None:
            return None
        instruction = INSTRUCTIONS[commitment.level]
        answer = await ask(COMMITMENT_QUESTION.format(instruction=instruction))
        if answer is None:
            return f"{instruction} You stated no floor, so none applies."
        commitment.floor = min(max(answer, 0.0), 1.0)
        return (
            f"{instruction} You committed to a floor of {commitment.floor:g}: you "
            f"may not offer or accept less than {commitment.floor:g} for yourself."
        )

    return opening


def enforced(player: Player, commitment: Commitment) -> Player:
    """The player, with offers and acceptances below its floor rejected.

    A rejection is an InvalidMove naming the floor, so the referee retries the
    player with that reason and defaults after the usual number of attempts.
    """
    standing_offer: Offer | None = None  # the opponent's offer a response answers

    async def play(observation: Observation, context: PhaseContext) -> Move:
        nonlocal standing_offer
        for move in observation.opponent_moves:
            if move.action is not None and move.action.kind == "offer":
                standing_offer = move.action
        move = await player(observation, context)
        floor = commitment.floor
        if floor is None or move.action is None:
            return move
        me = observation.recipient
        if move.action.kind == "offer":
            offered = own_share(move.action, me)
        elif move.action.kind == "accept" and standing_offer is not None:
            offered = own_share(standing_offer, me)
        else:
            return move
        if offered < floor - TOLERANCE:
            commitment.breaches += 1
            raise InvalidMove(
                f"this gives you {offered:g}, below your floor of {floor:g}; you "
                f"committed not to offer or accept less than {floor:g} for yourself"
            )
        return move

    return play


def floor_default(commitment: Commitment, fallback: DefaultMove) -> DefaultMove:
    """The game's default move, unless that would be an offer below the
    player's floor: then the referee offers the floor instead (in the strict
    version, a claim on the good)."""

    def default(
        game: GameSpec, state: EpisodeState, player: PlayerId, context: PhaseContext
    ) -> Move:
        move = fallback(game, state, player, context)
        floor = commitment.floor
        if floor is None or move.action.kind != "offer":
            return move
        if own_share(move.action, player) >= floor - TOLERANCE:
            return move
        if game.version == "strict":
            return Move(action=claim(player))
        return Move(action=Offer(split=floor if player == "alpha" else 1 - floor))

    return default


def own_share(offer: Offer, player: PlayerId) -> float:
    return offer.split if player == "alpha" else 1 - offer.split


def fidelity(
    commitment: Commitment, state: EpisodeState, player: PlayerId
) -> float | None:
    """The share of the player's move attempts that respected its floor: its
    recorded non-default moves count as kept, each rejection as a breach.
    None without a floor or any attempt."""
    if commitment.floor is None:
        return None
    kept = sum(
        player in record.moves and player not in record.defaulted
        for record in state.rounds
    )
    attempts = kept + commitment.breaches
    return kept / attempts if attempts else None


def collision(commitments: dict[PlayerId, Commitment]) -> bool | None:
    """Whether the two floors leave no split that satisfies both. None unless
    both players stated a floor."""
    floors = [commitment.floor for commitment in commitments.values()]
    if len(floors) < 2 or any(floor is None for floor in floors):
        return None
    return sum(floors) > 1 + TOLERANCE
