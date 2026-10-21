"""Tables and plots for the contract dynamics hypotheses, from Inspect logs.

Inspect reports one mean per metric per run. This script reads every episode
in a log directory and produces, with 95% confidence intervals:

1. an overview of each cell;
2. the honor rate against the temptation tau, with a logistic fit;
3. the contract effect: honoring after signing minus honoring in the imposed
   venture, at the same tau;
4. the fairness split: honoring by whether the bearer had the smaller or the
   larger benefit, and how the contract shared the risk;
5. persuasion by K: honoring, and the persuader's belief, by persuasion rounds;
6. token usage and cost per log, to check the pilot estimate.

    uv run --group analysis python scripts/analyze_pilot.py logs/pilot

Honoring measures count only decisions by model bearers that were not default
moves (bots decide by rule; pass --include-bots to keep them). Intervals are
Wilson intervals for rates and Newcombe intervals for differences of rates.
Episodes are treated as independent; the two episodes of a seed share nature's
draws, so the intervals are slightly too narrow when both are in a group.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from spar_harness.measures import contract_outcome_measures
from spar_harness.schema import EpisodeConfig, EpisodeState

sys.path.insert(0, str(Path(__file__).parent))
from pilot import PRICES

Z = 1.96
COMPREHENSION_TOLERANCE = 0.01

# Loading ------------------------------------------------------------------------


def load_episodes(log_dir: str) -> tuple[list[dict], list[dict]]:
    """One row per episode, and one usage row per log."""
    episodes, usage = [], []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        roles = {
            role: (config[0] if isinstance(config, list) else config).model
            for role, config in (log.eval.model_roles or {}).items()
        }
        rows = [
            _episode_row(log.eval.task, sample, roles)
            for sample in log.samples or []
            if sample.store.get("episode") and "config" in sample.metadata
        ]
        rows = [row for row in rows if row is not None]
        episodes += rows
        usage.append(_usage_row(log, len(rows)))
    return episodes, usage


def _episode_row(task: str, sample, roles: dict[str, str]) -> dict | None:
    config = EpisodeConfig.model_validate(sample.metadata["config"])
    if config.game.game_id != "contract_dynamics":
        return None
    episode = EpisodeState.model_validate(sample.store["episode"])
    probes = sample.store.get("probes") or []
    params = config.game.contract
    seats = sample.metadata["seats"]  # position -> seat
    controllers = sample.metadata["controllers"]  # seat -> controller
    models = {
        seat: roles.get(seat, "model")
        if controllers[seat] == "model"
        else controllers[seat]
        for seat in ("a", "b")
    }
    measures = contract_outcome_measures(config.game, episode)
    presentation = config.treatments.presentation
    stages = {stage.name: stage for stage in config.game.stages}
    bearer = episode.contract.bearer if episode.contract else None
    other = None if bearer is None else ("beta" if bearer == "alpha" else "alpha")
    benefits = params.benefits
    row = {
        "task": task,
        "episode_id": sample.id,
        "seed": config.seed,
        "opponent": _opponent(controllers, models),
        "model_a": models["a"],
        "model_b": models["b"],
        "setting": presentation.setting if presentation else "neutral",
        "prompt_arm": presentation.prompt_arm if presentation else "none",
        "severity": round(params.loss / params.disaster, 6),
        "transfers": _regime(params),
        "imposed": params.imposed_venture,
        "force_shock": params.force_shock,
        "K": stages["persuasion"].termination.max_rounds,
        "benefits": f"{benefits['alpha']:g}/{benefits['beta']:g}",
        "penalty": params.penalty,
        "ending": episode.ending,
    }
    row |= measures.model_dump(
        include={
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
        }
    )
    if params.imposed_venture:  # tau is defined by the fixed terms, t = 0
        row["temptation"] = params.loss - params.disaster - params.penalty
    bearer_seat = seats.get(bearer) if bearer else None
    decision = next(
        (record for record in episode.rounds if record.stage == "decision"), None
    )
    row |= {
        "bearer_seat": bearer_seat,
        "bearer_controller": controllers[bearer_seat] if bearer_seat else None,
        "bearer_model": models[bearer_seat] if bearer_seat else None,
        "bearer_rank": _rank(benefits, bearer) if bearer else None,
        "decision_defaulted": bool(decision and bearer in decision.defaulted),
        "persuader_controller": controllers[seats[other]] if other else None,
    }
    beliefs = [
        probe["answer"]
        for probe in probes
        if probe["player"] == other
        and probe["probe_id"] == "persuader_belief"
        and probe["answer"] is not None
    ]
    row["persuader_belief_first"] = beliefs[0] if beliefs else None
    row["persuader_belief_last"] = beliefs[-1] if beliefs else None
    comprehension = [
        probe["answer"]
        for probe in probes
        if probe["player"] == bearer and probe["probe_id"] == "comprehension"
    ]
    truth = benefits[bearer] - params.disaster - params.penalty if bearer else None
    row["bearer_comprehension"] = (
        int(
            comprehension[0] is not None
            and abs(comprehension[0] - truth) <= COMPREHENSION_TOLERANCE
        )
        if comprehension
        else None
    )
    usage = list((sample.model_usage or {}).values())
    row["input_tokens"] = sum(_all_input(u) for u in usage)
    row["output_tokens"] = sum(u.output_tokens for u in usage)
    return row


def _opponent(controllers: dict[str, str], models: dict[str, str]) -> str:
    a, b = controllers["a"], controllers["b"]
    if a == "model" and b == "model":
        return "self-play" if models["a"] == models["b"] else "cross-play"
    if a == "model":
        return f"vs {b}"
    if b == "model":
        return f"vs {a}"
    return f"{a} vs {b}"


def _regime(params) -> str:
    if params.transfer_cap == 0:
        return "none"
    return (
        "free"
        if params.transfer_cap == params.loss
        else f"capped {params.transfer_cap:g}"
    )


def _rank(benefits: dict[str, float], bearer: str) -> str:
    mine = benefits[bearer]
    theirs = benefits["beta" if bearer == "alpha" else "alpha"]
    return "equal" if mine == theirs else ("smaller" if mine < theirs else "larger")


def _all_input(usage) -> int:
    return (
        usage.input_tokens
        + (usage.input_tokens_cache_read or 0)
        + (usage.input_tokens_cache_write or 0)
    )


def _usage_row(log, episodes: int) -> dict:
    row = {"task": log.eval.task, "episodes": episodes, "status": log.status}
    usage = log.stats.model_usage if log.stats else {}
    row["models"] = ", ".join(usage)
    row["input_tokens"] = sum(_all_input(u) for u in usage.values())
    row["output_tokens"] = sum(u.output_tokens for u in usage.values())
    dollars = 0.0
    for name, u in usage.items():
        price = PRICES.get(name.split("/")[-1])
        if price is None:
            dollars = None
            break
        cache_write = (u.input_tokens_cache_write or 0) * 1.25
        cache_read = (u.input_tokens_cache_read or 0) * 0.1
        dollars += (
            (u.input_tokens + cache_write + cache_read) * price[0]
            + u.output_tokens * price[1]
        ) / 1e6
    row["dollars"] = dollars
    return row


# Statistics -----------------------------------------------------------------


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    """Rate and 95% Wilson interval."""
    if n == 0:
        return math.nan, math.nan, math.nan
    rate = successes / n
    centre = (rate + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * math.sqrt(rate * (1 - rate) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return rate, centre - half, centre + half


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference of two rates (first minus second) and its 95% interval."""
    if n1 == 0 or n2 == 0:
        return math.nan, math.nan, math.nan
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    diff = p1 - p2
    low = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, low, high


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, math.nan, math.nan
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    half = Z * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def logistic_fit(xs: list[float], ys: list[int]) -> dict | None:
    """Fit P(y = 1) = 1 / (1 + exp(-(b0 + b1 x))) by Newton's method.

    Returns None when the slope is not identified: too few points, no
    variation in x or y, or complete separation.
    """
    if len(xs) < 5 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    b0 = b1 = 0.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-max(min(b0 + b1 * x, 35), -35)))
            w = p * (1 - p)
            g0 += y - p
            g1 += (y - p) * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01**2
        if det <= 1e-12:
            return None
        step0 = (h11 * g0 - h01 * g1) / det
        step1 = (h00 * g1 - h01 * g0) / det
        b0, b1 = b0 + step0, b1 + step1
        if abs(step0) + abs(step1) < 1e-9:
            break
    if abs(b1) > 20:  # separation: the likelihood has no finite maximum
        return None
    se1 = math.sqrt(h00 / det)
    return {
        "intercept": b0,
        "slope": b1,
        "slope_low": b1 - Z * se1,
        "slope_high": b1 + Z * se1,
        "tau_at_half": -b0 / b1 if b1 != 0 else math.nan,
    }


# Tables -------------------------------------------------------------------------


def group_by(
    rows: Iterable[dict], key: Callable[[dict], tuple]
) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return dict(sorted(groups.items(), key=lambda item: _sort_key(item[0])))


def _sort_key(key: tuple) -> tuple:
    """Numbers in numeric order, everything else alphabetically."""
    return tuple(
        (0, value, "") if isinstance(value, int | float) else (1, 0, str(value))
        for value in key
    )


def decisions(rows: list[dict], include_bots: bool) -> list[dict]:
    """Honoring decisions that count: after a shock, not defaulted, by a model."""
    return [
        row
        for row in rows
        if row["honored"] is not None
        and (include_bots or row["bearer_controller"] == "model")
    ]


def rate_cells(rows: list[dict]) -> dict:
    k = sum(bool(row["honored"]) for row in rows)
    rate, low, high = wilson(k, len(rows))
    return {"n": len(rows), "honored": k, "rate": rate, "low": low, "high": high}


def overview(rows: list[dict], include_bots: bool) -> list[dict]:
    table = []
    for (arm, severity, opponent, k), group in group_by(
        rows,
        lambda r: (
            "imposed" if r["imposed"] else f"transfers {r['transfers']}",
            r["severity"],
            r["opponent"],
            r["K"],
        ),
    ).items():
        signed = [r["signed"] for r in group if r["signed"] is not None]
        taus = [r["temptation"] for r in group if r["temptation"] is not None]
        comp = [
            r["bearer_comprehension"]
            for r in group
            if r["bearer_comprehension"] is not None
        ]
        honor = rate_cells(decisions(group, include_bots))
        table.append(
            {
                "arm": arm,
                "severity": severity,
                "opponent": opponent,
                "K": k,
                "episodes": len(group),
                "signed": _share(signed),
                "walked_away": _share(
                    [r["walked_away"] for r in group if r["walked_away"] is not None]
                ),
                "missed_deal": _share(
                    [r["missed_deal"] for r in group if r["missed_deal"] is not None]
                ),
                "negotiation_failure": _share(
                    [
                        r["negotiation_failure"]
                        for r in group
                        if r["negotiation_failure"] is not None
                    ]
                ),
                "mean_tau": mean_ci(taus)[0],
                "decisions": honor["n"],
                "honor_rate": honor["rate"],
                "honor_low": honor["low"],
                "honor_high": honor["high"],
                "defaulted_decisions": sum(r["decision_defaulted"] for r in group),
                "comprehension": _share(comp),
            }
        )
    return table


def honor_by_tau(rows: list[dict], include_bots: bool) -> tuple[list[dict], list[dict]]:
    signed = [r for r in decisions(rows, include_bots) if not r["imposed"]]
    table = [
        {"opponent": opponent, "tau": tau, **rate_cells(group)}
        for (opponent, tau), group in group_by(
            signed, lambda r: (r["opponent"], round(r["temptation"], 1))
        ).items()
    ]
    fits = []
    for (opponent,), group in group_by(signed, lambda r: (r["opponent"],)).items():
        fit = logistic_fit(
            [r["temptation"] for r in group], [int(r["honored"]) for r in group]
        )
        fits.append(
            {
                "opponent": opponent,
                "n": len(group),
                **(fit or {"note": "slope not identified"}),
            }
        )
    return table, fits


def contract_effect(rows: list[dict], include_bots: bool) -> list[dict]:
    """Signed minus imposed honoring, matched on everything but the arm."""

    def match(r: dict) -> tuple:
        return (
            r["opponent"],
            r["setting"],
            r["prompt_arm"],
            r["K"],
            r["benefits"],
            round(r["temptation"], 1),
        )

    counted = decisions(rows, include_bots)
    signed = group_by([r for r in counted if not r["imposed"]], match)
    imposed = group_by([r for r in counted if r["imposed"]], match)
    table = []
    for key in sorted(set(signed) & set(imposed), key=_sort_key):
        s, i = rate_cells(signed[key]), rate_cells(imposed[key])
        diff, low, high = newcombe(s["honored"], s["n"], i["honored"], i["n"])
        opponent, setting, arm, k, benefits, tau = key
        table.append(
            {
                "opponent": opponent,
                "setting": setting,
                "prompt_arm": arm,
                "K": k,
                "benefits": benefits,
                "tau": tau,
                "signed_n": s["n"],
                "signed_rate": s["rate"],
                "imposed_n": i["n"],
                "imposed_rate": i["rate"],
                "effect": diff,
                "effect_low": low,
                "effect_high": high,
            }
        )
    return table


def fairness(
    rows: list[dict], include_bots: bool
) -> tuple[list[dict], list[dict], list[dict]]:
    signed = [r for r in decisions(rows, include_bots) if not r["imposed"]]
    by_rank = [
        {"opponent": opponent, "bearer_benefit": rank, **rate_cells(group)}
        for (opponent, rank), group in group_by(
            signed, lambda r: (r["opponent"], r["bearer_rank"])
        ).items()
    ]
    gaps = []
    for (opponent,), group in group_by(signed, lambda r: (r["opponent"],)).items():
        larger = rate_cells([r for r in group if r["bearer_rank"] == "larger"])
        smaller = rate_cells([r for r in group if r["bearer_rank"] == "smaller"])
        diff, low, high = newcombe(
            larger["honored"], larger["n"], smaller["honored"], smaller["n"]
        )
        if larger["n"] and smaller["n"]:
            gaps.append(
                {
                    "opponent": opponent,
                    "larger_minus_smaller": diff,
                    "low": low,
                    "high": high,
                }
            )
    terms = []
    contracts = [r for r in rows if r["signed"]]
    for (opponent, benefits), group in group_by(
        contracts, lambda r: (r["opponent"], r["benefits"])
    ).items():
        deviation = mean_ci([r["equal_gains_deviation"] for r in group])
        terms.append(
            {
                "opponent": opponent,
                "benefits": benefits,
                "contracts": len(group),
                "mean_q_alpha": mean_ci([r["alpha_bears_prob"] for r in group])[0],
                "share_q_half": _share(
                    [abs(r["alpha_bears_prob"] - 0.5) < 1e-9 for r in group]
                ),
                "equal_gains_deviation": deviation[0],
                "deviation_low": deviation[1],
                "deviation_high": deviation[2],
                "mean_transfer": mean_ci([r["transfer"] for r in group])[0],
            }
        )
    return by_rank, gaps, terms


def persuasion(rows: list[dict], include_bots: bool) -> list[dict]:
    table = []
    for (arm, opponent, k), group in group_by(
        decisions(rows, include_bots),
        lambda r: ("imposed" if r["imposed"] else "signed", r["opponent"], r["K"]),
    ).items():
        first = [
            r["persuader_belief_first"]
            for r in group
            if r["persuader_belief_first"] is not None
        ]
        last = [
            r["persuader_belief_last"]
            for r in group
            if r["persuader_belief_last"] is not None
        ]
        shift = [
            r["persuader_belief_last"] - r["persuader_belief_first"]
            for r in group
            if r["persuader_belief_first"] is not None
        ]
        brier = [
            (r["persuader_belief_last"] - int(r["honored"])) ** 2
            for r in group
            if r["persuader_belief_last"] is not None
        ]
        shift_ci = mean_ci(shift)
        table.append(
            {
                "arm": arm,
                "opponent": opponent,
                "K": k,
                **rate_cells(group),
                "persuader_belief_first": mean_ci(first)[0],
                "persuader_belief_last": mean_ci(last)[0],
                "belief_shift": shift_ci[0],
                "shift_low": shift_ci[1],
                "shift_high": shift_ci[2],
                "persuader_brier": mean_ci(brier)[0],
            }
        )
    return table


def _share(values: list) -> float:
    return sum(map(bool, values)) / len(values) if values else math.nan


# Output -----------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict], empty: str) -> str:
    if not rows:
        return f"_{empty}_\n"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        lines.append(
            "| " + " | ".join(_cell(row.get(field)) for field in fields) + " |"
        )
    return "\n".join(lines) + "\n"


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.3g}"
    return str(value)


def plots(out: Path, tau_table, fits, effect, by_rank, persuasion_table) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plots (use uv run --group analysis).")
        return []
    made = []

    def errorbars(ax, xs, rows, label, rate="rate", low="low", high="high"):
        ys = [r[rate] for r in rows]
        lower = [max(r[rate] - r[low], 0) for r in rows]  # rounding can dip below 0
        upper = [max(r[high] - r[rate], 0) for r in rows]
        ax.errorbar(xs, ys, yerr=[lower, upper], fmt="o", capsize=3, label=label)

    if tau_table:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (opponent,), group in group_by(
            tau_table, lambda r: (r["opponent"],)
        ).items():
            errorbars(ax, [r["tau"] for r in group], group, opponent)
            fit = next(
                (f for f in fits if f["opponent"] == opponent and "slope" in f), None
            )
            if fit:
                taus = [r["tau"] for r in group]
                grid = [min(taus) + i * (max(taus) - min(taus)) / 50 for i in range(51)]
                curve = [
                    1 / (1 + math.exp(-(fit["intercept"] + fit["slope"] * x)))
                    for x in grid
                ]
                ax.plot(grid, curve, alpha=0.6)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set(
            xlabel="temptation τ = L − t − D − F",
            ylabel="honor rate",
            ylim=(-0.05, 1.05),
        )
        ax.set_title("Honoring against temptation (95% CI)")
        ax.legend()
        made.append(_save(fig, out / "honor_vs_tau.png"))

    if effect:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f"{r['opponent']}\nτ={r['tau']:g}, K={r['K']}" for r in effect]
        errorbars(
            ax,
            range(len(effect)),
            effect,
            "signed − imposed",
            "effect",
            "effect_low",
            "effect_high",
        )
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(effect)), labels, fontsize=8)
        ax.set(ylabel="contract effect on honoring")
        ax.set_title("Contract effect (95% CI)")
        made.append(_save(fig, out / "contract_effect.png"))

    ranked = [r for r in by_rank if r["bearer_benefit"] in ("smaller", "larger")]
    if ranked:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f"{r['opponent']}\n{r['bearer_benefit']}" for r in ranked]
        errorbars(ax, range(len(ranked)), ranked, "honor rate")
        ax.set_xticks(range(len(ranked)), labels, fontsize=8)
        ax.set(ylabel="honor rate", ylim=(-0.05, 1.05))
        ax.set_title("Honoring by the bearer's benefit (95% CI)")
        made.append(_save(fig, out / "fairness.png"))

    if len({r["K"] for r in persuasion_table}) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (arm, opponent), group in group_by(
            persuasion_table, lambda r: (r["arm"], r["opponent"])
        ).items():
            errorbars(ax, [r["K"] for r in group], group, f"{arm}, {opponent}")
        ax.set(xlabel="persuasion rounds K", ylabel="honor rate", ylim=(-0.05, 1.05))
        ax.set_title("Honoring by persuasion rounds (95% CI)")
        ax.legend()
        made.append(_save(fig, out / "persuasion_by_k.png"))
    return made


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path.name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log_dir")
    parser.add_argument("--out", help="output directory (default: <log_dir>/analysis)")
    parser.add_argument(
        "--include-bots", action="store_true", help="count bot bearers' decisions"
    )
    args = parser.parse_args(argv)

    episodes, usage = load_episodes(args.log_dir)
    if not episodes:
        sys.exit(f"No contract dynamics episodes found in {args.log_dir}.")
    out = Path(args.out or Path(args.log_dir) / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    bots = args.include_bots
    cells = overview(episodes, bots)
    tau_table, fits = honor_by_tau(episodes, bots)
    effect = contract_effect(episodes, bots)
    by_rank, gaps, terms = fairness(episodes, bots)
    persuasion_table = persuasion(episodes, bots)

    tables = {
        "episodes": episodes,
        "overview": cells,
        "honor_by_tau": tau_table,
        "honor_tau_fit": fits,
        "contract_effect": effect,
        "fairness_by_benefit": by_rank,
        "fairness_gap": gaps,
        "contract_terms": terms,
        "persuasion_by_k": persuasion_table,
        "usage": usage,
    }
    for name, rows in tables.items():
        write_csv(out / f"{name}.csv", rows)
    images = plots(out, tau_table, fits, effect, by_rank, persuasion_table)

    who = "all bearers" if bots else "model bearers only"
    report = [
        f"# Contract dynamics pilot: {args.log_dir}\n",
        (
            f"{len(episodes)} episodes from {len(usage)} logs. Honoring counts {who}, "
            "non-default decisions. 95% Wilson intervals; Newcombe for differences.\n"
        ),
        "## 1. Cells\n",
        markdown(cells, "no episodes"),
        "## 2. Honoring against temptation τ\n",
        markdown(tau_table, "no signed contracts with a model bearer's decision"),
        "\nLogistic fit of honoring on τ (slope per unit of τ; τ at which the fitted rate is one half):\n\n",
        markdown(fits, "no fits"),
        "## 3. Contract effect (signed minus imposed, same τ)\n",
        markdown(
            effect,
            "no signed and imposed decisions at the same τ: run the pilot with --imposed, "
            "and note the rational bot never signs without transfers",
        ),
        "## 4. Fairness split\n",
        markdown(by_rank, "no decisions"),
        "\n",
        markdown(
            gaps,
            "no gap: benefits were equal, so no bearer had the smaller or larger benefit "
            "(run the pilot with unequal --benefits)",
        ),
        "\nContract terms (equal-gains deviation is agreed q minus the q that equalizes gains):\n\n",
        markdown(terms, "no signed contracts"),
        "## 5. Persuasion by K\n",
        markdown(persuasion_table, "no decisions"),
        "\nThe last persuader belief is taken before the final message (issue #5).\n",
        "## 6. Usage and cost\n",
        markdown(usage, "no logs"),
        (
            "\nDollars use first-party list prices with cache reads at 0.1x and writes "
            "at 1.25x input; blank when a model has no listed price.\n"
        ),
    ]
    if images:
        report += ["## Figures\n"] + [f"![{name}]({name})\n" for name in images]
    (out / "report.md").write_text("\n".join(report))
    print(
        f"Wrote {out / 'report.md'}, {len(tables)} CSV tables and {len(images)} figures."
    )


if __name__ == "__main__":
    main()
