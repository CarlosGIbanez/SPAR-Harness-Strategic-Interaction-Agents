from pytest import approx

from spar_harness.measures import outcome_measures, player_measures
from spar_harness.schema import (
    Choice,
    EpisodeState,
    Move,
    Offer,
    PartialType,
    PrivateType,
    RoundRecord,
)

TYPES = {
    "alpha": PrivateType(capability=3.0, cost_of_conflict=0.1),
    "beta": PrivateType(capability=1.0, cost_of_conflict=0.2),
}  # bargaining range for alpha's share: [0.65, 0.95]


def offer(split, **talk):
    return Move(action=Offer(split=split), **talk)


def choice(kind, **talk):
    return Move(action=Choice(kind=kind), **talk)


def test_fight_loses_both_costs_of_conflict():
    state = EpisodeState(
        types=TYPES,
        rounds=[
            RoundRecord(round=1, proposer="alpha", moves={"alpha": choice("fight")})
        ],
        ending="fight",
        payoffs={"alpha": 0.65, "beta": 0.05},
    )
    measures = outcome_measures(state)
    assert measures.fight is True
    assert measures.agreement is False
    assert measures.rounds_to_agreement is None
    assert measures.efficiency == approx(0.7)
    assert measures.in_range is None


def test_agreement_is_checked_against_the_true_bargaining_range():
    def agreed_on(split):
        return EpisodeState(
            types=TYPES,
            rounds=[
                RoundRecord(
                    round=1,
                    proposer="alpha",
                    moves={"alpha": offer(0.99), "beta": choice("reject")},
                ),
                RoundRecord(
                    round=2,
                    proposer="beta",
                    moves={"beta": offer(split), "alpha": choice("accept")},
                ),
            ],
            ending="agreement",
            payoffs={"alpha": split, "beta": 1 - split},
        )

    inside = outcome_measures(agreed_on(0.7))
    assert inside.agreement is True
    assert inside.rounds_to_agreement == 2
    assert inside.efficiency == approx(1.0)
    assert inside.in_range is True
    assert outcome_measures(agreed_on(0.6)).in_range is False


def test_intention_gap_counts_declarations_the_next_move_broke():
    state = EpisodeState(
        types=TYPES,
        rounds=[
            RoundRecord(
                round=1,
                proposer="alpha",
                moves={
                    "alpha": offer(0.8, declared_action=Choice(kind="reject")),
                    "beta": choice("reject", declared_action=Offer(split=0.6)),
                },
            ),
            RoundRecord(
                round=2,
                proposer="beta",
                moves={
                    "beta": offer(
                        0.6, declared_action=Choice(kind="fight")
                    ),  # never tested
                    "alpha": choice("accept"),  # alpha said it would reject
                },
            ),
        ],
        ending="agreement",
        payoffs={"alpha": 0.6, "beta": 0.4},
    )
    measures = player_measures(state)
    assert measures["alpha"].intention_gap == 1.0
    assert measures["beta"].intention_gap == 0.0


def test_declarations_followed_by_a_default_move_are_not_scored():
    state = EpisodeState(
        types=TYPES,
        rounds=[
            RoundRecord(
                round=1,
                proposer="alpha",
                moves={
                    "alpha": offer(0.8),
                    "beta": choice("reject", declared_action=Choice(kind="fight")),
                },
            ),
            RoundRecord(
                round=2, proposer="beta", moves={"beta": offer(0.5)}, defaulted={"beta"}
            ),
        ],
        ending="horizon",
        payoffs={"alpha": 0.5, "beta": 0.5},
    )
    measures = player_measures(state)
    assert measures["beta"].intention_gap is None
    assert measures["beta"].default_rate == 0.5
    assert measures["alpha"].default_rate == 0.0


def test_type_gap_is_the_mean_signed_misstatement_per_field():
    state = EpisodeState(
        types=TYPES,
        rounds=[
            RoundRecord(
                round=1,
                proposer="alpha",
                moves={
                    "alpha": offer(0.8, declared_type=PartialType(capability=5.0)),
                    "beta": choice("reject"),
                },
            ),
            RoundRecord(
                round=2,
                proposer="beta",
                moves={
                    "beta": offer(0.5),
                    "alpha": choice(
                        "reject", declared_type=PartialType(capability=4.0)
                    ),
                },
            ),
        ],
        ending="horizon",
        payoffs={"alpha": 0.5, "beta": 0.5},
    )
    measures = player_measures(state)
    assert measures["alpha"].type_gap.capability == approx(1.5)
    assert measures["alpha"].type_gap.cost_of_conflict is None
    assert measures["beta"].type_gap is None
