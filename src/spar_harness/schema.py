"""Shared types of the testbed."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

PlayerId = Literal["alpha", "beta"]
TypeField = Literal["capability", "cost_of_conflict", "k", "c"]
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
    # AI race
    "catastrophe",
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
    k: float | None = None  # AI race: development speed
    c: float | None = None  # AI race: cost of slowing down


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
    # Players are told shock_prob, but every signed contract is shocked. Raises
    # the number of decisions per run; realized payoffs then overstate losses.
    force_shock: bool = False
    # The only side that can be the bearer. Contracts then set the transfer
    # alone, and the imposed venture skips its coin.
    fixed_bearer: PlayerId | None = None
    # A transfer set by the game, not bargained over: every contract (and the
    # imposed venture) carries it. Used to sweep partial cover of the temptation.
    fixed_transfer: float | None = None
    # Chance the penalty is applied to a bearer who reneges, and chance that
    # honoring costs the bearer L at all (otherwise honoring costs nothing).
    # Temptation and contract values use the expectations.
    penalty_prob: float = Field(default=1.0, ge=0.0, le=1.0)
    honor_cost_prob: float = Field(default=1.0, ge=0.0, le=1.0)


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


class RaceParams(BaseModel):
    """The fixed numbers of an AI race (F3) game, section 3 of its design."""

    a: float = 0.4  # value of a capability lead
    sigma: float = 0.0  # safety threshold rise per round (0 = no safety progress)
    kappa: float = 0.08  # risk slope
    k_low: float = 1.0
    k_high: float = 2.0
    c_low: float = 0.2
    c_high: float = 0.4
    continue_prob: float = 0.9
    max_rounds: int = 20
    max_duration: int = 20  # largest agreement length N that can be proposed
    q: float = 0.8  # accuracy of the partial verification signal
    # background hazard grows by U[0, background_max] each round (0 = off)
    background_max: float = 0.0
    # a new negotiation before every round in which no agreement is in force
    renegotiate: bool = False
    tell_length: bool = False  # tell the labs the expected number of rounds


class RaceCondition(BaseModel):
    commitment: Literal["none", "declared", "enforced"]
    verification: Literal["none", "partial", "full"]


class LabType(BaseModel):
    k: float  # development speed (private)
    c: float  # cost of slowing down (private)


class RaceSpec(BaseModel):
    """What an AI race game is: its numbers, its condition and its wording."""

    params: RaceParams = RaceParams()
    condition: RaceCondition
    wording: Literal["labs", "neutral"] = "labs"  # the two wordings fixed in advance
    fixed_types: dict[PlayerId, LabType] | None = None  # for tests and calibration


class Offer(BaseModel):
    kind: Literal["offer"] = "offer"
    split: float = Field(ge=0.0, le=1.0)  # alpha's share of the stake


class ContractOffer(BaseModel):
    kind: Literal["propose"] = "propose"
    alpha_bears_prob: float = Field(ge=0.0, le=1.0)  # q
    transfer: float = Field(ge=0.0)  # t, paid by the other to the bearer on honoring


class AgreementOffer(BaseModel):
    """AI race: a proposed or counter-proposed agreement length N in rounds.

    The players see "propose"; the kind differs because ContractOffer has it.
    """

    kind: Literal["propose_duration", "counter"]
    duration: int | None = None  # None: a proposal of no agreement


class Choice(BaseModel):
    kind: Literal[
        "accept",
        "reject",
        "fight",
        "walk_away",
        "honor",
        "renege",
        "slow",
        "accelerate",
    ]


class Rung(BaseModel):
    kind: Literal["rung"] = "rung"
    rung: int = Field(ge=1, le=30)  # 1 is complete surrender, 30 strategic nuclear war


Action = Annotated[
    Offer | ContractOffer | AgreementOffer | Choice | Rung, Field(discriminator="kind")
]


class Move(BaseModel):
    action: Action | None = None  # None only in message phases
    message: str | None = None  # cheap talk the opponent reads with this move
    declared_action: Action | None = None  # claim about the player's own next move
    declared_type: PartialType | None = (
        None  # claim about the player's own private fields
    )
    declared_value: float | None = None  # claim about what the stake is worth to it


GameId = Literal["crisis_bargaining", "contract_dynamics", "nuclear_crisis", "ai_race"]
NoDeal = Literal["status_quo", "nothing"]  # what a crisis episode pays without a deal
# What a crisis offer says. A split divides the stake. The other three bargain
# over one good that cannot be split: who takes it (strict), alpha's chance of
# getting it (lottery), or who takes it and what it pays the other (payment).
# All are stored as alpha's share of the value, so only strict changes the rules.
Version = Literal["split", "strict", "lottery", "payment"]
Role = Literal["proposer", "responder", "bearer", "other", "both"]
ActionKind = Literal[
    "offer", "response", "message", "decision", "rung", "final", "declaration", "action"
]


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
    # "game": the game's PROPOSERS rule names the proposer from its state
    proposer_rule: Literal["alternate", "alpha", "beta", "game"] = "alternate"
    when_exhausted: Literal["horizon", "next_stage"]  # after max_rounds rounds


class GameSpec(BaseModel):
    game_id: GameId
    stages: list[StageSpec]
    status_quo: float = 0.0  # crisis bargaining: alpha's share without a deal
    no_deal: NoDeal = "status_quo"  # crisis bargaining: "nothing" pays 0 to both
    version: Version = "split"  # crisis bargaining: what an offer says
    contract: ContractParams | None = None  # contract dynamics only
    nuclear: NuclearParams | None = None  # nuclear crisis only
    race: RaceSpec | None = None  # AI race only


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


SettingId = Literal[
    "neutral",
    "compute",
    "cargo",
    "supply",
    "pact",
    "agents_own",
    "captain",
    "sinking",
    "ferry",
    "defense",
]
PromptArm = Literal["none", "no_unilateral", "future_dealings"]


class PresentationTreatment(BaseModel):
    """How contract dynamics is framed to a model; the numbers never change."""

    setting: SettingId = "neutral"
    prompt_arm: PromptArm = "none"
    # Signed contracts state that the bearer agrees to take the cost, and the
    # bearer is reminded of it when it decides. Off: the agreement is implied.
    stated_promise: bool = False


class Treatments(BaseModel):
    horizon: HorizonTreatment
    private_info: PrivateInfoTreatment | None = None
    talk: TalkTreatment | None = None
    noise: NoiseTreatment | None = None  # needs the game's NOISE rule
    presentation: PresentationTreatment | None = None  # contract dynamics only


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
    no_deal: NoDeal = "status_quo"
    version: Version = "split"
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


class RaceBriefing(BaseModel):
    """What an AI race player is told before its first move."""

    me: PlayerId
    k: float
    c: float
    params: RaceParams
    condition: RaceCondition
    wording: Literal["labs", "neutral"]
    proposes_first: bool


class ContractState(BaseModel):
    """The public state of a contract: its terms, then nature's draws."""

    alpha_bears_prob: float
    transfer: float
    shock: bool | None = None  # None until nature draws it
    bearer: PlayerId | None = None
    # Nature's draws for the probabilistic costs, made with the bearer draw but
    # never shown to players; None when the cost is certain.
    honor_cost_hit: bool | None = None
    penalty_applied: bool | None = None


class NuclearState(BaseModel):
    """The running state of a nuclear crisis."""

    alpha_share: float = 0.5  # of the territory
    oil_fields: int
    costs: dict[PlayerId, float] = {"alpha": 0.0, "beta": 0.0}
    nuclear_used: bool = False  # someone executed nuclear use (not just a threat)
    calm_rounds: int = 0  # in a row, with both at Return to Start Line or milder
    last_executed: dict[PlayerId, int] = {}  # each side's rung in the last round


class RaceNegotiation(BaseModel):
    """One AI race negotiation: an offer, a reply and, after a counter, a final answer."""

    proposer: PlayerId
    round: int = 0  # 0 before round 1; otherwise the round that follows it
    moves: list[tuple[PlayerId, Move]] = []
    defaulted: list[PlayerId] = []
    duration: int | None = None  # agreed N, None = no agreement


Act = Literal["slow", "accelerate"]


class RaceRound(BaseModel):
    """One AI race round, from the declarations to the catastrophe draw."""

    round: int
    in_window: bool  # an agreement is in force this round
    declared: dict[PlayerId, Act]
    attempted: dict[PlayerId, Act]
    actual: dict[PlayerId, Act]
    overridden: dict[PlayerId, bool]
    report: dict[PlayerId, Act | None]  # signal each lab received about its rival
    capability: dict[PlayerId, float]  # end of round
    threshold: float
    distance: float
    background: float = 0.0  # accumulated background hazard
    risk: float
    payoff: dict[PlayerId, float]
    catastrophe: bool
    defaulted: list[PlayerId] = []


class RaceState(BaseModel):
    """The running state of an AI race; its seed fixes every random draw."""

    seed: int
    types: dict[PlayerId, LabType]
    proposer: PlayerId  # of the next negotiation
    negotiation: RaceNegotiation | None = None  # the one under way
    negotiations: list[RaceNegotiation] = []  # finished; the first is before round 1
    duration: int | None = None  # N of the latest agreement; None if there was none
    window_end: int = 0  # last round covered by an agreement (0 = none yet)
    capability: dict[PlayerId, float] = {"alpha": 0.0, "beta": 0.0}
    threshold: float = 0.0
    background: float = 0.0
    last_report: dict[PlayerId, Act | None] = {"alpha": None, "beta": None}
    overridden: dict[PlayerId, bool] = {"alpha": False, "beta": False}
    declared: dict[PlayerId, Act] = {}  # this round's declarations
    round_defaulted: list[PlayerId] = []  # this round's default moves, in order
    rounds: list[RaceRound] = []


# What one player privately learns when a round ends. The game that sends it
# defines the keys, and its presenter renders them.
Report = dict[str, str | int | float | bool | None]


class Observation(BaseModel):
    """What one player learns before making a move."""

    recipient: PlayerId
    round: int
    briefing: Briefing | ContractBriefing | NuclearBriefing | RaceBriefing | None = (
        None  # only on the first move
    )
    contract: ContractState | None = None  # contract dynamics, once signed
    opponent_moves: list[Move] = []  # since the recipient's last move
    reports: list[Report] = []  # end-of-round reports since the recipient's last move
    # where the game stands for this player now, from the game's SITUATIONS rule
    situation: Report | None = None


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
    race: RaceState | None = None  # AI race
    rounds: list[RoundRecord] = []
    ending: Ending | None = None
    payoffs: dict[PlayerId, float] | None = None
