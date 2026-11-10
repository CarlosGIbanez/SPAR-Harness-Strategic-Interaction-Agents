"""Types of the F3 game (commitment and verification in an AI race)."""

from typing import Literal

from pydantic import BaseModel

LabId = Literal["A", "B"]
Commitment = Literal["none", "declared", "enforced"]
Verification = Literal["none", "partial", "full"]
Act = Literal["slow", "accelerate"]
Phase = Literal["negotiation", "declaration", "action"]
NegotiationKind = Literal["propose", "accept", "reject", "counter"]
Ending = Literal["catastrophe", "random_stop", "horizon"]
Wording = Literal["labs", "neutral"]  # the two prompt wordings fixed in advance

LABS: tuple[LabId, LabId] = ("A", "B")


def rival_of(lab: LabId) -> LabId:
    return "B" if lab == "A" else "A"


class Params(BaseModel):
    """Baseline parameters of section 3 of the design."""

    a: float = 0.4  # value of a capability lead
    sigma: float = 0.0  # safety threshold rise per round (0 = no safety progress within an episode)
    kappa: float = 0.08  # risk slope
    k_low: float = 1.0
    k_high: float = 2.0
    c_low: float = 0.2
    c_high: float = 0.4
    continue_prob: float = 0.9
    max_rounds: int = 20
    max_duration: int = 20  # largest agreement length N that can be proposed
    q: float = 0.8  # accuracy of the partial verification signal
    background_max: float = (
        0.0  # background hazard grows by U[0, background_max] each round (0 = off)
    )
    renegotiate: bool = (
        False  # a new negotiation before every round in which no agreement is in force
    )
    tell_length: bool = (
        False  # tell the labs the expected number of rounds (in the prompt only)
    )


class Condition(BaseModel):
    commitment: Commitment
    verification: Verification


class EpisodeConfig(BaseModel):
    episode_id: str
    seed: int
    condition: Condition
    params: Params = Params()
    wording: Wording = "labs"
    paired_with: str | None = None  # the seat-swapped twin
    fixed_types: dict[LabId, "LabType"] | None = None  # for tests and calibration


class LabType(BaseModel):
    k: float  # development speed (private)
    c: float  # cost of slowing down (private)


EpisodeConfig.model_rebuild()


# moves


class NegotiationMove(BaseModel):
    kind: NegotiationKind
    duration: int | None = None  # N for propose and counter; None = no agreement
    message: str | None = None
    stated_k: float | None = None
    stated_c: float | None = None


class ActMove(BaseModel):
    """A declaration or an actual action."""

    action: Act


Move = NegotiationMove | ActMove


# what a lab sees


class Briefing(BaseModel):
    """Shown once, before the lab's first move."""

    me: LabId
    k: float
    c: float
    params: Params
    condition: Condition
    wording: Wording = "labs"
    proposes_first: bool


class AgreementStatus(BaseModel):
    duration: int  # N
    rounds_left: int  # including the current round; 0 once expired


class Observation(BaseModel):
    recipient: LabId
    phase: Phase
    round: int  # 0 during negotiation
    briefing: Briefing | None = None
    # negotiation
    rival_move: NegotiationMove | None = None
    # rounds
    own_capability: float | None = None
    threshold: float | None = None
    agreement: AgreementStatus | None = None
    agreement_expired: bool = False
    last_report: Act | None = None  # verification signal about the rival last round
    own_overridden: bool = False  # own attempt last round was changed to Slow down
    rival_declaration: Act | None = None  # action phase only


class PhaseContext(BaseModel):
    phase: Phase
    round: int
    options: list[str]  # allowed move kinds or actions
    max_duration: int
    pending_offer: int | None = None  # the duration on the table, if any
    error: str | None = None


# the record of an episode


class RoundRecord(BaseModel):
    round: int
    in_window: bool  # an agreement is in force this round
    declared: dict[LabId, Act]
    attempted: dict[LabId, Act]
    actual: dict[LabId, Act]
    overridden: dict[LabId, bool]
    report: dict[LabId, Act | None]  # signal each lab received about its rival
    capability: dict[LabId, float]  # end of round
    threshold: float
    distance: float
    background: float = (
        0.0  # accumulated background hazard (not controlled by the labs)
    )
    risk: float
    payoff: dict[LabId, float]
    catastrophe: bool
    defaulted: list[LabId] = []


class NegotiationRecord(BaseModel):
    proposer: LabId
    round: int = (
        0  # 0 for the negotiation before round 1; otherwise the round that follows it
    )
    moves: list[tuple[LabId, NegotiationMove]] = []
    defaulted: list[LabId] = []
    duration: int | None = None  # agreed N, None = no agreement


class EpisodeRecord(BaseModel):
    config: EpisodeConfig
    types: dict[LabId, LabType]
    negotiation: NegotiationRecord
    renegotiations: list[NegotiationRecord] = []  # only if params.renegotiate
    rounds: list[RoundRecord]
    ending: Ending
    payoffs: dict[LabId, float]
