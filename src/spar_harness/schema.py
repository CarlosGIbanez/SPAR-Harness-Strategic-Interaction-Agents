"""Shared types of the testbed."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

PlayerId = Literal["alpha", "beta"]
TypeField = Literal["capability", "cost_of_conflict"]
TalkField = Literal["message", "declared_action", "declared_type"]
Ending = Literal[
    # crisis bargaining
    "agreement",
    "fight",
    "random_stop",
    # contract dynamics
    "walked_away",
    "no_shock",
    "honored",
    "reneged",
    # both: the round cap was reached without agreement
    "horizon",
]


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


class ContractParams(BaseModel):
    """The fixed numbers of a contract dynamics game."""

    benefits: dict[PlayerId, float]  # b_i, paid to each side on signing
    shock_prob: float  # p, chance the shock hits a signed contract
    loss: float  # L, the cost the bearer absorbs on honoring
    disaster: float  # D, what each side loses if the bearer reneges
    penalty: float  # F, extra loss for a bearer who reneges
    transfer_cap: float  # largest transfer a contract may specify
    imposed_venture: bool = False  # control arm: no signing, the shock is certain


class Offer(BaseModel):
    kind: Literal["offer"] = "offer"
    split: float = Field(ge=0.0, le=1.0)  # alpha's share of the stake


class ContractOffer(BaseModel):
    kind: Literal["propose"] = "propose"
    alpha_bears_prob: float = Field(ge=0.0, le=1.0)  # q
    transfer: float = Field(ge=0.0)  # t, paid by the other to the bearer on honoring


class Choice(BaseModel):
    kind: Literal["accept", "reject", "fight", "walk_away", "honor", "renege"]


Action = Annotated[Offer | ContractOffer | Choice, Field(discriminator="kind")]


class Move(BaseModel):
    action: Action | None = None  # None only in message phases
    message: str | None = None  # cheap talk the opponent reads with this move
    declared_action: Action | None = None  # claim about the player's own next move
    declared_type: PartialType | None = (
        None  # claim about the player's own private fields
    )


GameId = Literal["crisis_bargaining", "contract_dynamics"]
Role = Literal["proposer", "responder", "bearer", "other"]
ActionKind = Literal["offer", "response", "message", "decision"]


class PhaseSpec(BaseModel):
    kind: Literal["act"]
    actors: Role
    action_kind: ActionKind


class TerminationSpec(BaseModel):
    max_rounds: int
    stop_prob: float


class StageSpec(BaseModel):
    name: str
    phases: list[PhaseSpec]  # one round's phase sequence
    termination: TerminationSpec
    proposer_rule: Literal["alternate", "alpha", "beta"] = "alternate"
    when_exhausted: Literal["horizon", "next_stage"]  # after max_rounds rounds


class GameSpec(BaseModel):
    game_id: GameId
    stages: list[StageSpec]
    status_quo: float = 0.0  # crisis bargaining: alpha's share without a deal
    contract: ContractParams | None = None  # contract dynamics only


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
    priors: dict[PlayerId, TypePrior] | None = None  # None: the game has no types
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


class ContractBriefing(BaseModel):
    """What a contract dynamics player is told before its first move."""

    params: ContractParams
    max_rounds: int  # of bargaining
    persuasion_rounds: int


class ContractState(BaseModel):
    """The public state of a contract: its terms, then nature's draws."""

    alpha_bears_prob: float
    transfer: float
    shock: bool | None = None  # None until nature draws it
    bearer: PlayerId | None = None


class Observation(BaseModel):
    """What one player learns before making a move."""

    recipient: PlayerId
    round: int
    briefing: Briefing | ContractBriefing | None = None  # only on the first move
    contract: ContractState | None = None  # contract dynamics, once signed
    opponent_moves: list[Move] = []  # since the recipient's last move


class PhaseContext(BaseModel):
    """What the referee is asking a player to do."""

    round: int
    role: Role
    action_kind: ActionKind
    allowed_actions: set[str]
    talk_fields: set[TalkField] = set()
    declarable_fields: set[TypeField] = (
        set()
    )  # private fields a declared type may claim
    error: str | None = None  # why the previous attempt was rejected


class RoundRecord(BaseModel):
    stage: str = "bargaining"
    round: int  # within the stage
    proposer: PlayerId | None = None  # None in stages without offers
    moves: dict[PlayerId, Move] = {}
    defaulted: set[PlayerId] = set()  # players whose move the referee made for them


class EpisodeState(BaseModel):
    types: dict[PlayerId, PrivateType] = {}
    contract: ContractState | None = None  # contract dynamics, once signed
    rounds: list[RoundRecord] = []
    ending: Ending | None = None
    payoffs: dict[PlayerId, float] | None = None
