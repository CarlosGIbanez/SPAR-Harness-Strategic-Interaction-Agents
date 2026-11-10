"""Shared types of the testbed."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

PlayerId = Literal["alpha", "beta"]
TypeField = Literal["capability", "cost_of_conflict"]
TalkField = Literal["message", "declared_action", "declared_type", "declared_value"]
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
    # nuclear crisis
    "knockout",
    "strategic_war",
    "ceasefire",
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


class NuclearWorld(BaseModel):
    """What the disputed territory holds. Each side values it differently."""

    oil_fields: int  # heavy fighting burns them
    towns: dict[PlayerId, int]  # towns of each side's own nationality
    pass_value: dict[PlayerId, float]  # what the mountain pass is worth to each side
    oil_value: float = 7.0  # per oil field, to either side
    town_value: float = 5.0  # per town of one's own nationality


class NuclearStart(BaseModel):
    """Where a game that begins mid-crisis starts: the state and the rounds so far."""

    alpha_share: float = Field(ge=0.0, le=1.0)
    oil_fields: int
    costs: dict[PlayerId, float]
    nuclear_used: bool  # whether someone already crossed the tactical threshold
    history: list[dict[PlayerId, int]]  # the rungs each side took, round by round


class NuclearParams(BaseModel):
    """The fixed numbers of a nuclear crisis game."""

    world: NuclearWorld
    payoffs: Literal["jervis", "greedy"] = "jervis"
    framing: Literal["military", "neutral"] = "military"
    shift_per_value: float = 0.001  # territory share per point of rung-value gap
    shift_cap: float = 0.15  # the most territory share either side gains in a round
    cost_per_value: float = 0.025  # what each side pays per point of the hottest rung
    opponent_value_range: tuple[float, float] = (30.0, 100.0)  # told to the players
    start: NuclearStart | None = None  # None: the crisis starts at the start line
    explains_jumps: set[PlayerId] = set()  # say "it was an accident" after a jump
    scripted: set[PlayerId] = set()  # played by a script: its orders never misfire


class Offer(BaseModel):
    kind: Literal["offer"] = "offer"
    split: float = Field(ge=0.0, le=1.0)  # alpha's share of the stake


class ContractOffer(BaseModel):
    kind: Literal["propose"] = "propose"
    alpha_bears_prob: float = Field(ge=0.0, le=1.0)  # q
    transfer: float = Field(ge=0.0)  # t, paid by the other to the bearer on honoring


class Choice(BaseModel):
    kind: Literal["accept", "reject", "fight", "walk_away", "honor", "renege"]


class Rung(BaseModel):
    kind: Literal["rung"] = "rung"
    rung: int = Field(ge=1, le=30)  # 1 is complete surrender, 30 strategic nuclear war


Action = Annotated[Offer | ContractOffer | Choice | Rung, Field(discriminator="kind")]


class Move(BaseModel):
    action: Action | None = None  # None only in message phases
    message: str | None = None  # cheap talk the opponent reads with this move
    declared_action: Action | None = None  # claim about the player's own next move
    declared_type: PartialType | None = (
        None  # claim about the player's own private fields
    )
    declared_value: float | None = None  # claim about what the stake is worth to it


GameId = Literal["crisis_bargaining", "contract_dynamics", "nuclear_crisis"]
Role = Literal["proposer", "responder", "bearer", "other", "both"]
ActionKind = Literal["offer", "response", "message", "decision", "rung"]


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
    nuclear: NuclearParams | None = None  # nuclear crisis only


class HorizonTreatment(BaseModel):
    known: bool


class PrivateInfoTreatment(BaseModel):
    private_fields: set[TypeField]


class TalkTreatment(BaseModel):
    messages: bool
    declared_action: bool
    declared_type: bool
    declared_value: bool = False


class NoiseTreatment(BaseModel):
    rate: float = Field(ge=0.0, le=1.0)  # chance that a move is carried out wrong


class Treatments(BaseModel):
    horizon: HorizonTreatment
    private_info: PrivateInfoTreatment | None = None
    talk: TalkTreatment | None = None
    noise: NoiseTreatment | None = None  # needs the game's NOISE rule


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


class NuclearBriefing(BaseModel):
    """What a nuclear crisis player is told before its first move: its own worth."""

    oil_fields: int
    oil_value: float
    towns: int  # of the player's own nationality
    town_value: float
    pass_value: float
    opponent_value_range: tuple[float, float]
    payoffs: Literal["jervis", "greedy"]
    framing: Literal["military", "neutral"]
    max_rounds: int | None  # None if the horizon is unknown
    start: NuclearStart | None  # the rounds already played, for a mid-crisis start


class ContractState(BaseModel):
    """The public state of a contract: its terms, then nature's draws."""

    alpha_bears_prob: float
    transfer: float
    shock: bool | None = None  # None until nature draws it
    bearer: PlayerId | None = None


class NuclearState(BaseModel):
    """The running state of a nuclear crisis."""

    alpha_share: float = 0.5  # of the territory
    oil_fields: int
    costs: dict[PlayerId, float] = {"alpha": 0.0, "beta": 0.0}
    nuclear_used: bool = False  # someone executed nuclear use (not just a threat)
    calm_rounds: int = 0  # in a row, with both at Return to Start Line or milder
    last_executed: dict[PlayerId, int] = {}  # each side's rung in the last round


# What one player privately learns when a round ends. The game that sends it
# defines the keys, and its presenter renders them.
Report = dict[str, str | int | float | bool | None]


class Observation(BaseModel):
    """What one player learns before making a move."""

    recipient: PlayerId
    round: int
    briefing: Briefing | ContractBriefing | NuclearBriefing | None = (
        None  # only on the first move
    )
    contract: ContractState | None = None  # contract dynamics, once signed
    opponent_moves: list[Move] = []  # since the recipient's last move
    reports: list[Report] = []  # end-of-round reports since the recipient's last move


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
    reports: dict[PlayerId, Report] = {}  # what each player was told at the end
    # moves that noise changed; every other move was executed as chosen
    executed: dict[PlayerId, Move] = {}


class RoundEnd(BaseModel):
    """What a game's end-of-round rule decides once every phase is played."""

    reports: dict[PlayerId, Report] = {}  # each goes only to its player
    outcome: Ending | Literal["next_stage"] | None = None  # None: play on


class EpisodeState(BaseModel):
    types: dict[PlayerId, PrivateType] = {}
    contract: ContractState | None = None  # contract dynamics, once signed
    nuclear: NuclearState | None = None  # nuclear crisis
    rounds: list[RoundRecord] = []
    ending: Ending | None = None
    payoffs: dict[PlayerId, float] | None = None
