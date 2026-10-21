"""The pilot estimate and the analysis script, end to end on stand-in logs."""

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_pilot
import pilot


def test_the_default_grid_is_three_severities_two_regimes_two_opponents():
    cells = pilot.grid(
        pilot.parse_args(["estimate", "--model", "anthropic/claude-haiku-5-5"])
    )
    assert len(cells) == 12
    assert {cell.severity for cell in cells} == {1.2, 1.5, 1.8}
    assert {cell.transfers for cell in cells} == {"none", "free"}
    assert {cell.opponent for cell in cells} == {"rational", "self"}
    assert not any(cell.imposed_venture for cell in cells)


def test_an_unpriced_model_needs_explicit_prices():
    args = pilot.parse_args(["estimate", "--model", "openai-api/dartmouth/gemma"])
    try:
        pilot.prices(args)
    except SystemExit as error:
        assert "--price-in" in str(error)
    else:
        raise AssertionError("expected an exit")
    free = pilot.parse_args(
        ["estimate", "--model", "x/y", "--price-in", "0", "--price-out", "0"]
    )
    assert pilot.prices(free) == (0, 0)


def test_estimate_then_analysis_on_the_stand_in_logs(tmp_path):
    log_dir = tmp_path / "logs"
    args = pilot.parse_args(
        [
            "estimate",
            "--model",
            "anthropic/claude-haiku-5-5",
            "--seeds",
            "2",
            "--severities",
            "1.2",
            "--persuasion",
            "0",
            "1",
            "--imposed",
            "--benefits",
            "28",
            "10",
            "--keep-logs",
            str(log_dir),
        ]
    )
    result = pilot.estimate(args)
    assert len(result["cells"]) == 12  # 2 K x 2 opponents x (2 regimes + imposed)
    for row in result["cells"]:
        assert row["quick"]["calls"] <= row["slow"]["calls"]
    assert 0 < result["dollars_typical"] <= result["dollars_upper"]

    out = tmp_path / "analysis"
    analyze_pilot.main([str(log_dir), "--out", str(out)])
    report = (out / "report.md").read_text()
    for heading in (
        "Honoring against temptation",
        "Contract effect",
        "Fairness",
        "Persuasion",
    ):
        assert heading in report
    with (out / "episodes.csv").open() as file:
        episodes = list(csv.DictReader(file))
    assert len(episodes) == 12 * 2 * 2  # each episode once
    with (out / "contract_effect.csv").open() as file:
        assert list(csv.DictReader(file))  # self-play signs at t = 0, as imposed


def test_intervals_and_the_logistic_fit():
    rate, low, high = analyze_pilot.wilson(8, 10)
    assert rate == 0.8 and 0.49 < low < 0.5 and 0.94 < high < 0.95
    assert math.isnan(analyze_pilot.wilson(0, 0)[0])
    diff, low, high = analyze_pilot.newcombe(8, 10, 2, 10)
    assert abs(diff - 0.6) < 1e-9 and low < diff < high
    xs = [x / 2 for x in range(-20, 21)] * 2
    ys = [int(x < 1) if i % 3 else int(x < -1) for i, x in enumerate(xs)]
    fit = analyze_pilot.logistic_fit(xs, ys)
    assert fit["slope"] < 0 and fit["slope_low"] < fit["slope"] < fit["slope_high"]
    separated = analyze_pilot.logistic_fit([1, 2, 3, 4, 5], [0, 0, 1, 1, 1])
    assert separated is None
