"""The two analysis scripts: the floors run and the per-seed join."""

import asyncio
import importlib.util
import statistics
import sys
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from spar_harness.task import crisis_bargaining

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_floors_script_asks_each_position_in_the_episodes_own_words():
    floors = load("commitment_floors")
    questions = []

    def model(input, tools, tool_choice, config):
        questions.append(input[-1].text)
        return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.6})

    config = floors.config_for(
        type(
            "Args",
            (),
            {
                "private_fields": [],
                "max_rounds": 20,
                "stop_prob": 0.1,
                "status_quo": 0.5,
                "no_deal": "nothing",
            },
        )(),
        "lottery",
        seed=1,
    )
    fight_value, floor = asyncio.run(
        floors.ask_floor(
            get_model("mockllm/model", custom_outputs=model), config, "beta", level=4
        )
    )
    assert floor == 0.6
    assert -1 <= fight_value <= 1
    assert "never be exploited" in questions[-1]
    assert "state your floor" in questions[-1].lower()


def test_the_floors_script_hides_private_fields_and_logs_fidelity(capsys):
    floors = load("commitment_floors")
    args = type(
        "Args",
        (),
        {
            "private_fields": ["capability", "cost_of_conflict"],
            "max_rounds": 20,
            "stop_prob": 0.1,
            "status_quo": 0.5,
            "no_deal": "nothing",
        },
    )()
    config = floors.config_for(args, "strict", seed=2)
    assert config.treatments.private_info.private_fields == {
        "capability",
        "cost_of_conflict",
    }
    row = {
        "level": 5,
        "version": "strict",
        "seed": 2,
        "repeat": 0,
        "position": "alpha",
        "floor": 0.9,
        "fight_value": 0.4,
        "aggressiveness": 0.5,
        "fidelity": 1,
    }
    floors.log_call(row)
    assert "floor=0.900" in capsys.readouterr().err
    floors.summarize(
        [row, row | {"floor": None, "aggressiveness": None, "fidelity": None}]
    )
    out = capsys.readouterr().out
    assert "fidelity" in out
    assert "    5    2       1   0.500   0.500   1.00     1.00" in out


def test_the_seed_diff_script_joins_two_arms_on_seed():
    logs = []
    for version in ("strict", "lottery"):
        [log] = eval(
            crisis_bargaining(
                n_episodes=3, a="equilibrium", b="equilibrium", version=version
            ),
            model="mockllm/model",
            display="none",
            log_dir="logs/tests",
        )
        logs.append(Path(log.location))
    seed_diff = load("seed_diff")
    a = seed_diff.values_by_seed([logs[0]], "fight")
    b = seed_diff.values_by_seed([logs[1], logs[1]], "fight")  # two repeats
    assert set(a) == set(b) == {0, 1, 2}
    assert all(len(values) == 2 for values in a.values())
    assert all(len(values) == 4 for values in b.values())
    differences = [statistics.mean(b[s]) - statistics.mean(a[s]) for s in sorted(a)]
    assert all(diff <= 0 for diff in differences)  # lottery never fights more
    assert seed_diff.within_seed_sd(b) == 0.0
    low, high = seed_diff.bootstrap(
        differences, statistics.mean, 200, seed_diff.random.Random(0)
    )
    assert low <= statistics.mean(differences) <= high
    # an undefined measure is left out rather than counted as zero
    assert seed_diff.values_by_seed([logs[0]], "a_floor") == {}
