"""Shared types of the testbed."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

PlayerId = Literal["alpha", "beta"]
TypeField = Literal["capability", "cost_of_conflict"]
TalkField = Literal["message", "declared_action", "declared_type"]
Ending = Literal["agreement", "fight", "horizon", "random_stop"]


class PrivateType(BaseModel):
    capability: float
    cost_of_conflict: float


class PartialType(BaseModel):
    """Some fields of a private type; None means unknown or unstated."""

    capability: float | None = None
    cost_of_conflict: float | None = None


class UniformPrior(BaseModel):
    """Uniform distribution over [low, high]; low == high fixes the value."""

    low: float
    high: float


class TypePrior(BaseModel):
    capability: UniformPrior
    cost_of_conflict: UniformPrior


class Offer(BaseModel):
    kind: Literal["offer"] = "offer"
    split: float = Field(ge=0.0, le=1.0)  # alpha's share of the stake


class Choice(BaseModel):
    kind: Literal["accept", "reject", "fight"]


Action = Annotated[Offer | Choice, Field(discriminator="kind")]


class Move(BaseModel):
    action: Action
    message: str | None = None  # cheap talk the opponent reads with this move
    declared_action: Action | None = None  # claim about the player's own next move
    declared_type: PartialType | None = (
        None  # claim about the player's own private fields
    )


class PhaseSpec(BaseModel):
    kind: Literal["act"]
    actors: Literal["proposer", "responder"]
    action_kind: Literal["offer", "response"]


class TerminationSpec(BaseModel):
    max_rounds: int
    stop_prob: float


class GameSpec(BaseModel):
    game_id: Literal["crisis_bargaining"]
    phases: list[PhaseSpec]  # one round's phase sequence
    proposer_rule: Literal["alternate", "alpha", "beta"]
    termination: TerminationSpec
    status_quo: float  # alpha's share if the episode ends without a deal


class HorizonTreatment(BaseModel):
    known: bool


class PrivateInfoTreatment(BaseModel):
    private_fields: set[TypeField]


class TalkTreatment(BaseModel):
    messages: bool
    declared_action: bool
    declared_type: bool


class Treatments(BaseModel):
    horizon: HorizonTreatment
    private_info: PrivateInfoTreatment | None = None
    talk: TalkTreatment | None = None


class EpisodeConfig(BaseModel):
    episode_id: str
    game: GameSpec
    treatments: Treatments
    priors: dict[PlayerId, TypePrior]
    seed: int
    paired_with: str | None = None  # episode id of the role-swapped twin


class Briefing(BaseModel):
    """What a player is told about the game before its first move."""

    own_type: PrivateType
    opponent_type: PartialType  # the fields that are not private
    opponent_prior: TypePrior | None  # None under complete information
    private_fields: set[TypeField]
    status_quo: float
    termination: TerminationSpec | None  # None if the horizon is unknown


class Observation(BaseModel):
    """What one player learns before making a move."""

    recipient: PlayerId
    round: int
    briefing: Briefing | None = None  # only on the player's first move
    opponent_moves: list[Move] = []  # since the recipient's last move


class PhaseContext(BaseModel):
    """What the referee is asking a player to do."""

    round: int
    role: Literal["proposer", "responder"]
    action_kind: Literal["offer", "response"]
    talk_fields: set[TalkField] = set()
    declarable_fields: set[TypeField] = (
        set()
    )  # private fields a declared type may claim
    error: str | None = None  # why the previous attempt was rejected


class RoundRecord(BaseModel):
    round: int
    proposer: PlayerId
    moves: dict[PlayerId, Move] = {}
    defaulted: set[PlayerId] = set()  # players whose move the referee made for them


class EpisodeState(BaseModel):
    types: dict[PlayerId, PrivateType]
    rounds: list[RoundRecord] = []
    ending: Ending | None = None
    payoffs: dict[PlayerId, float] | None = None
