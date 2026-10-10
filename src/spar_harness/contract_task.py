"""Inspect task: contract dynamics between two seats, each a model or a bot.

Each seed is run as a pair: seat a plays alpha (with alpha's benefit) in the
first episode and beta in its twin, so with unequal benefits every seat plays
both the larger and the smaller reward. Per-player measures are reported by seat.
"""

from typing import Literal

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from spar_harness.contract_dynamics import contract_dynamics_game
from spar_harness.measures import contract_outcome_measures, contract_player_measures
from spar_harness.model_player import ProbeResponse
from spar_harness.schema import (
    ContractParams,
    EpisodeConfig,
    EpisodeState,
    HorizonTreatment,
    PlayerId,
    TalkTreatment,
    Treatments,
)
from spar_harness.task import (
    COMPREHENSION_TOLERANCE,
    Controller,
    Seat,
    mean_of_defined,
    mean_of_defined_reducer,
    referee,
)


@task
def contract_dynamics(
    n_episodes: int = 10,
    a: Controller = "model",
    b: Controller = "model",
    benefit_alpha: float = 18.0,
    benefit_beta: float = 18.0,
    shock_prob: float = 0.5,
    disaster: float = 40.0,
    severity: float = 1.5,  # L / D
    penalty: float = 0.0,
    transfers: Literal["none", "capped", "free"] = "none",
    transfer_cap: float = 10.0,  # used when transfers="capped"
    max_rounds: int = 6,
    persuasion_rounds: int = 0,
    imposed_venture: bool = False,
    messages: bool = False,
    promises: bool = False,
    appeal: str = "",  # what scripted bots say when they persuade
) -> Task:
    loss = severity * disaster
    cap = {"none": 0.0, "capped": transfer_cap, "free": loss}[transfers]
    params = ContractParams(
        benefits={"alpha": benefit_alpha, "beta": benefit_beta},
        shock_prob=shock_prob,
        loss=loss,
        disaster=disaster,
        penalty=penalty,
        transfer_cap=cap,
        imposed_venture=imposed_venture,
    )
    game = contract_dynamics_game(
        params, max_rounds=max_rounds, persuasion_rounds=persuasion_rounds
    )
    talk = None
    if messages or promises:
        talk = TalkTreatment(
            messages=messages, declared_action=promises, declared_type=False
        )
    treatments = Treatments(horizon=HorizonTreatment(known=True), talk=talk)
    controllers = {"a": a, "b": b}
    samples = []
    for seed in range(n_episodes):
        for seats, twin in (
            ({"alpha": "a", "beta": "b"}, "ba"),
            ({"alpha": "b", "beta": "a"}, "ab"),
        ):
            order = seats["alpha"] + seats["beta"]
            config = EpisodeConfig(
                episode_id=f"{seed}-{order}",
                game=game,
                treatments=treatments,
                seed=seed,
                paired_with=f"{seed}-{twin}",
            )
            samples.append(
                Sample(
                    id=config.episode_id,
                    input="contract dynamics episode",
                    metadata={
                        "config": config.model_dump(mode="json"),
                        "seats": seats,
                        "controllers": controllers,
                        "appeal": appeal,
                    },
                )
            )
    return Task(
        dataset=MemoryDataset(samples),
        solver=referee(),
        scorer=[contract_outcome_scorer(), contract_seat_scorer()],
        epochs=Epochs(1, mean_of_defined_reducer()),
    )


OUTCOME_KEYS = [
    "signed",
    "walked_away",
    "rounds_to_signing",
    "alpha_bears_prob",
    "transfer",
    "temptation",
    "equal_gains_deviation",
    "shock",
    "honored",
    "negotiation_failure",
    "execution_failure",
    "missed_deal",
    "joint_payoff",
]


@scorer(metrics={key: [mean_of_defined()] for key in OUTCOME_KEYS})
def contract_outcome_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        config = EpisodeConfig.model_validate(state.metadata["config"])
        episode = EpisodeState.model_validate(state.store.get("episode"))
        measures = contract_outcome_measures(config.game, episode)
        return Score(value=measures.model_dump(include=set(OUTCOME_KEYS)))

    return score


SEAT_MEASURES = [
    "default_rate",
    "promise_broken",
    "honored_with_smaller_reward",
    "honored_with_larger_reward",
    "comprehension",
    "belief",
    "belief_error",
]
SEAT_KEYS = [f"{seat}_{measure}" for seat in ("a", "b") for measure in SEAT_MEASURES]


@scorer(metrics={key: [mean_of_defined()] for key in SEAT_KEYS})
def contract_seat_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        config = EpisodeConfig.model_validate(state.metadata["config"])
        episode = EpisodeState.model_validate(state.store.get("episode"))
        probes = [
            ProbeResponse.model_validate(probe) for probe in state.store.get("probes")
        ]
        seats: dict[PlayerId, Seat] = state.metadata["seats"]
        by_player = contract_player_measures(config.game, episode)
        partner_honored = _partner_honored(config, episode)
        values = {}
        for player, seat in seats.items():
            measures = by_player[player]
            belief = _mean_answer(probes, player, "belief")
            values |= {
                f"{seat}_default_rate": measures.default_rate,
                f"{seat}_promise_broken": measures.promise_broken,
                f"{seat}_honored_with_smaller_reward": measures.honored_with_smaller_reward,
                f"{seat}_honored_with_larger_reward": measures.honored_with_larger_reward,
                f"{seat}_comprehension": _comprehension(config, probes, player),
                f"{seat}_belief": belief,
                f"{seat}_belief_error": (
                    belief - partner_honored[player]
                    if belief is not None and partner_honored[player] is not None
                    else None
                ),
            }
        return Score(value=values)

    return score


def _mean_answer(
    probes: list[ProbeResponse], player: PlayerId, probe_id: str
) -> float | None:
    answers = [
        probe.answer
        for probe in probes
        if probe.player == player
        and probe.probe_id == probe_id
        and probe.answer is not None
    ]
    return sum(answers) / len(answers) if answers else None


def _comprehension(
    config: EpisodeConfig, probes: list[ProbeResponse], player: PlayerId
) -> int | None:
    """1 if the bearer correctly stated its payoff from reneging, 0 if not."""
    params = config.game.contract
    truth = params.benefits[player] - params.disaster - params.penalty
    for probe in probes:
        if probe.player == player and probe.probe_id == "comprehension":
            correct = (
                probe.answer is not None
                and abs(probe.answer - truth) <= COMPREHENSION_TOLERANCE
            )
            return int(correct)
    return None


def _partner_honored(
    config: EpisodeConfig, episode: EpisodeState
) -> dict[PlayerId, int | None]:
    """For each player: 1 if its partner was the bearer and honored, 0 if it
    reneged, None if the partner never made a (non-default) decision."""
    honored = contract_outcome_measures(config.game, episode).honored
    bearer = episode.contract.bearer if episode.contract else None
    return {
        player: int(honored)
        if honored is not None and bearer is not None and bearer != player
        else None
        for player in ("alpha", "beta")
    }
