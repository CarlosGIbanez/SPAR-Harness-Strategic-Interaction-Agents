"""The two analysis scripts: the floors run and the per-seed join."""

import asyncio
import importlib.util
import statistics
import sys
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from spar_harness.commitment import INSTRUCTIONS
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
    assert INSTRUCTIONS[4] in questions[-1]
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


def test_the_floors_script_resumes_a_partial_sweep(tmp_path, monkeypatch, capsys):
    floors = load("commitment_floors")
    calls = []

    def model(input, tools, tool_choice, config):
        calls.append(input[-1].text)
        return ModelOutput.for_tool_call("mock", "answer_probe", {"answer": 0.5})

    mock = get_model("mockllm/model", custom_outputs=model)
    monkeypatch.setattr(floors, "get_model", lambda name: mock)
    out = tmp_path / "floors.csv"
    argv = [
        "commitment_floors.py",
        "--model",
        "mockllm/model",
        "--levels",
        "1",
        "4",
        "--versions",
        "lottery",
        "--seeds",
        "2",
        "--out",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    asyncio.run(floors.main())
    assert len(calls) == 8  # 2 levels x 2 seeds x 2 positions
    first = out.read_text().splitlines()
    assert len(first) == 9  # header + 8 rows
    assert first[0].split(",")[:5] == ["level", "version", "seed", "repeat", "position"]

    calls.clear()
    asyncio.run(floors.main())
    assert calls == []  # every row was already there
    assert out.read_text().splitlines() == first
    assert "8 rows already in" in capsys.readouterr().err

    # a partial file resumes: drop the last two rows and only they are re-asked
    out.write_text("\n".join(first[:-2]) + "\n")
    asyncio.run(floors.main())
    assert len(calls) == 2
    rows = out.read_text().splitlines()
    assert len(rows) == 9
    assert sorted(rows[1:]) == sorted(first[1:])


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
