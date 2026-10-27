"""A transfer set by the game: every contract and the imposed venture carry it,
so honoring can be measured at partial cover of the temptation."""

import asyncio
import sys
from pathlib import Path

import pytest

from spar_harness.bots import rational_contract_bot, scripted_contract_bot
from spar_harness.contract_dynamics import (
    contract_dynamics_game,
    credible_transfers,
    deal_exists,
)
from spar_harness.contract_presenter import contract_presenter
from spar_harness.referee import run_episode
from spar_harness.schema import (
    ContractBriefing,
    ContractOffer,
    ContractParams,
    EpisodeConfig,
    HorizonTreatment,
    Move,
    Observation,
    PhaseContext,
    PresentationTreatment,
    Treatments,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import pilot


def params(fixed_transfer=10.0, **overrides):
    values = {
        "benefits": {"alpha": 18.0, "beta": 18.0},
        "shock_prob": 0.05,
        "loss": 60.0,
        "disaster": 40.0,
        "penalty": 0.0,
        "transfer_cap": fixed_transfer,
        "fixed_bearer": "alpha",
        "fixed_transfer": fixed_transfer,
    }
    return ContractParams(**(values | overrides))


def run(alpha, beta, seed=0, **overrides):
    config = EpisodeConfig(
        episode_id="test",
        game=contract_dynamics_game(
            params(**overrides), max_rounds=6, persuasion_rounds=0
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        seed=seed,
    )
    return asyncio.run(run_episode(config, {"alpha": alpha, "beta": beta}))


def test_only_the_fixed_transfer_is_credible_and_only_if_it_covers_the_gap():
    # L - D = 20: a transfer of 20 or more removes the temptation.
    assert credible_transfers(params(fixed_transfer=25.0)) == (25.0, 25.0)
    assert credible_transfers(params(fixed_transfer=20.0)) == (20.0, 20.0)
    assert credible_transfers(params(fixed_transfer=10.0)) is None
    assert credible_transfers(params(fixed_transfer=50.0)) is None  # above D
    assert not deal_exists(params(fixed_transfer=10.0))


@pytest.mark.parametrize("imposed_venture", [False, True])
def test_signed_and_imposed_ventures_both_carry_the_fixed_transfer(imposed_venture):
    state = run(
        scripted_contract_bot("honor"),
        scripted_contract_bot("honor"),
        force_shock=True,
        imposed_venture=imposed_venture,
    )
    assert state.contract.transfer == 10.0
    assert state.ending == "honored"
    assert state.payoffs == {"alpha": 18.0 - 60.0 + 10.0, "beta": 18.0 - 10.0}


def test_an_offer_with_another_transfer_is_rejected():
    errors = []

    async def proposer(observation, context):
        errors.append(context.error)
        return Move(action=ContractOffer(alpha_bears_prob=1.0, transfer=30.0))

    state = run(proposer, scripted_contract_bot("honor"), force_shock=True)
    assert "the transfer is fixed at 10" in errors[1:]
    assert state.contract.transfer == 10.0  # the default offer carries it


def test_rational_bots_sign_only_a_covering_transfer_and_honor_it():
    covered = run(
        rational_contract_bot(),
        rational_contract_bot(),
        fixed_transfer=20.0,
        force_shock=True,
    )
    assert covered.contract.transfer == 20.0
    assert covered.ending == "honored"
    partial = run(
        rational_contract_bot(),
        rational_contract_bot(),
        fixed_transfer=10.0,
        force_shock=True,
    )
    assert partial.ending == "walked_away"


def presenter_text(imposed_venture=False):
    game_params = params(imposed_venture=imposed_venture)
    presenter = contract_presenter(game_params, PresentationTreatment(setting="ferry"))
    briefing = ContractBriefing(params=game_params, max_rounds=6, persuasion_rounds=0)
    context = PhaseContext(
        round=1, role="proposer", action_kind="offer", allowed_actions={"propose"}
    )
    first = Observation(recipient="alpha", round=1, briefing=briefing)
    return presenter, presenter.render("alpha", first, context)


def test_the_fixed_transfer_is_stated_and_cannot_be_proposed():
    presenter, text = presenter_text()
    assert "Every contract carries a fixed transfer of 10" in text
    assert "benefit - 60 + transfer" in text
    tool = presenter.tool(
        Observation(recipient="alpha", round=1),
        PhaseContext(
            round=1, role="proposer", action_kind="offer", allowed_actions=set()
        ),
    )
    assert "transfer" not in tool.parameters.properties
    move = presenter.parse("beta", {"action": "propose", "transfer": 55})
    assert move.action == ContractOffer(alpha_bears_prob=1.0, transfer=10.0)
    _, imposed = presenter_text(imposed_venture=True)
    assert (
        "The other side pays the captain a transfer of 10 if the captain chooses "
        "divert_to_port."
    ) in imposed


def test_the_pilot_sweeps_fixed_transfers_with_matching_imposed_cells():
    cells = pilot.grid(
        pilot.parse_args(
            [
                "estimate",
                "--model",
                "anthropic/claude-haiku-5-5",
                "--severities",
                "1.5",
                "--fixed-transfers",
                "0",
                "10",
                "20",
                "--imposed",
            ]
        )
    )
    assert len(cells) == 12  # 3 transfers x 2 opponents x signed/imposed
    assert len({cell.label for cell in cells}) == 12
    assert {cell.fixed_transfer for cell in cells} == {0.0, 10.0, 20.0}
    assert sum(cell.imposed_venture for cell in cells) == 6
