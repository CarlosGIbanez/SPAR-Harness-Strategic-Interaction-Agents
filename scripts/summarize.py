"""Build logs/pilot/SUMMARY.md tables from the pilot logs. Run from the repo root."""

import csv
import glob
import statistics
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

PILOT = Path("logs/pilot")
IN_RATE, OUT_RATE = 2e-6, 10e-6  # proxy rates for claude-sonnet-5-5
FLOORS_CALL_COST = 0.0045  # estimate per floors call (script does not log tokens)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


# ---------- result 1
rows = list(csv.DictReader((PILOT / "result1_floors.csv").open()))
for r in rows:
    for k in ("fight_value", "floor", "aggressiveness"):
        r[k] = float(r[k]) if r[k] not in ("", "None") else None
    r["fidelity"] = int(r["fidelity"]) if r["fidelity"] not in ("", "None") else None
    r["level"] = int(r["level"])
versions = ["strict", "lottery", "payment"]
levels = sorted({r["level"] for r in rows})
out = ["## Result 1: commitment only (600 calls)\n"]
out.append("Median aggressiveness (floor − fight value) by level and version:\n")
out.append("| level | " + " | ".join(versions) + " | fidelity rate | unparseable |")
out.append("|---|" + "---|" * (len(versions) + 2))
for lv in levels:
    at = [r for r in rows if r["level"] == lv]
    cells = []
    for v in versions:
        g = [r["aggressiveness"] for r in at if r["version"] == v and r["floor"] is not None]
        cells.append(fmt(statistics.median(g), 3) if g else "–")
    stated = [r for r in at if r["floor"] is not None]
    fid = mean([r["fidelity"] for r in stated])
    miss = (len(at) - len(stated)) / len(at)
    out.append(f"| {lv} | " + " | ".join(cells) + f" | {fmt(fid)} | {fmt(miss)} |")
out.append("\nCollision rate (alpha floor + beta floor > 1, same level and seed) per version:\n")
out.append("| version | collision rate | n pairs |\n|---|---|---|")
for v in versions:
    pairs = defaultdict(dict)
    for r in rows:
        if r["version"] == v:
            pairs[(r["level"], r["seed"], r["repeat"])][r["position"]] = r["floor"]
    both = [p for p in pairs.values() if p.get("alpha") is not None and p.get("beta") is not None]
    coll = mean([p["alpha"] + p["beta"] > 1 + 1e-9 for p in both])
    out.append(f"| {v} | {fmt(coll)} | {len(both)} |")
out.append("\nCollision rate by level (all versions):\n")
out.append("| level | collision rate |\n|---|---|")
for lv in levels:
    pairs = defaultdict(dict)
    for r in rows:
        if r["level"] == lv:
            pairs[(r["version"], r["seed"], r["repeat"])][r["position"]] = r["floor"]
    both = [p for p in pairs.values() if p.get("alpha") is not None and p.get("beta") is not None]
    out.append(f"| {lv} | {fmt(mean([p['alpha'] + p['beta'] > 1 + 1e-9 for p in both]))} |")


# ---------- eval logs
def load(name):
    paths = glob.glob(str(PILOT / name / "*.eval"))
    assert len(paths) == 1, paths
    return read_eval_log(paths[0])


def episode_stats(log):
    s = defaultdict(list)
    for smp in log.samples:
        o = smp.scores["outcome_scorer"].value
        seat = smp.scores["seat_scorer"].value
        s["fight"].append(o["fight"])
        s["agreement"].append(o["agreement"])
        s["efficiency"].append(o["efficiency"])
        s["in_range"].append(o["in_range"])
        s["rounds"].append(len(smp.store["episode"]["rounds"]))
        s["collision"].append(o["collision"])
        for seat_id in ("a", "b"):
            s["comprehension"].append(seat[f"{seat_id}_comprehension"])
            s["default_rate"].append(seat[f"{seat_id}_default_rate"])
            s["floor"].append(seat[f"{seat_id}_floor"])
            s["fidelity"].append(seat[f"{seat_id}_fidelity"])
            s["compliance"].append(seat[f"{seat_id}_compliance"])
            s["breaches"].append(seat[f"{seat_id}_breaches"])
        i = sum(u.input_tokens for u in smp.model_usage.values())
        o_ = sum(u.output_tokens for u in smp.model_usage.values())
        s["in_tok"].append(i)
        s["out_tok"].append(o_)
    return s


total_cost = 0.0
total_in = total_out = 0
cost_lines = []


def cost_of(name, s):
    global total_cost, total_in, total_out
    c = sum(s["in_tok"]) * IN_RATE + sum(s["out_tok"]) * OUT_RATE
    total_cost += c
    total_in += sum(s["in_tok"])
    total_out += sum(s["out_tok"])
    cost_lines.append(f"| {name} | {len(s['fight'])} | {sum(s['in_tok']):,} | {sum(s['out_tok']):,} | ${c:.2f} |")
    return c


out.append("\n## Result 2: baseline, arm 1 (no instruction), 20 seeds × both orders\n")
out.append("| version / no-deal | fight | agreement | efficiency | mean rounds | in-range | comprehension | default rate | n |")
out.append("|---|---|---|---|---|---|---|---|---|")
r2 = {}
flags = []
for name, label in [
    ("result2_strict_nothing", "strict / nothing"),
    ("result2_lottery_nothing", "lottery / nothing"),
    ("result2_lottery_statusquo", "lottery / status quo 0.5"),
    ("result2_payment_nothing", "payment / nothing"),
]:
    try:
        s = episode_stats(load(name))
    except AssertionError:
        out.append(f"| {label} | (missing) |")
        continue
    r2[name] = s
    cost_of(name, s)
    out.append(
        f"| {label} | {fmt(mean(s['fight']))} | {fmt(mean(s['agreement']))} | {fmt(mean(s['efficiency']))} | "
        f"{fmt(mean(s['rounds']))} | {fmt(mean(s['in_range']))} | {fmt(mean(s['comprehension']))} | "
        f"{fmt(mean(s['default_rate']))} | {len(s['fight'])} |"
    )
    if mean(s["default_rate"]) and mean(s["default_rate"]) > 0.10:
        flags.append(f"{name}: default rate {mean(s['default_rate']):.2f} > 0.10")
if "result2_strict_nothing" in r2:
    fr = mean(r2["result2_strict_nothing"]["fight"])
    if abs(fr - 0.9) > 0.15:
        flags.append(f"strict arm 1 fight rate is {fr:.2f}, far from the expected 0.90")

out.append("\n## Result 3: SPI pilot, lottery / nothing, 20 seeds, arm 1 vs arm 3 (level-2 floor on both seats), 3 repeats each\n")
out.append("| run | fight | agreement | efficiency | mean rounds | mean floor | fidelity | compliance | breaches | default rate |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
arm = {"arm1": [], "arm3": []}
for a in ("arm1", "arm3"):
    for rep in (1, 2, 3):
        name = f"result3_{a}_rep{rep}"
        try:
            s = episode_stats(load(name))
        except AssertionError:
            out.append(f"| {name} | (missing) |")
            continue
        arm[a].append(s)
        cost_of(name, s)
        out.append(
            f"| {name} | {fmt(mean(s['fight']))} | {fmt(mean(s['agreement']))} | {fmt(mean(s['efficiency']))} | "
            f"{fmt(mean(s['rounds']))} | {fmt(mean(s['floor']))} | {fmt(mean(s['fidelity']))} | "
            f"{fmt(mean(s['compliance']))} | {fmt(mean(s['breaches']))} | {fmt(mean(s['default_rate']))} |"
        )
        if mean(s["default_rate"]) and mean(s["default_rate"]) > 0.10:
            flags.append(f"{name}: default rate {mean(s['default_rate']):.2f} > 0.10")
for a in ("arm1", "arm3"):
    if arm[a]:
        out.append(f"\n{a} pooled: fight {fmt(mean(sum((s['fight'] for s in arm[a]), [])))}, "
                   f"mean floor {fmt(mean(sum((s['floor'] for s in arm[a]), [])))}, "
                   f"collision rate {fmt(mean(sum((s['collision'] for s in arm[a]), [])))}")
sd = PILOT / "result3_seed_diff.txt"
if sd.exists():
    out.append("\n`scripts/seed_diff.py --measure fight --delta 0.05` output:\n")
    out.append("```\n" + sd.read_text().strip() + "\n```")
    for line in sd.read_text().splitlines():
        if "within-seed" in line.lower() or "spread" in line.lower():
            nums = [float(t) for t in line.replace(":", " ").split() if t.replace(".", "", 1).isdigit()]
            if nums and nums[-1] > 0.05:
                flags.append(f"result 3 within-seed spread > 0.05: `{line.strip()}`")

out.append("\n## Cost and time\n")
floors_cost = len(rows) * FLOORS_CALL_COST
out.append("| run | samples | input tokens | output tokens | cost |\n|---|---|---|---|---|")
out.append(f"| result1_floors | {len(rows)} calls | – | – | ~${floors_cost:.2f} (est. ${FLOORS_CALL_COST} per call; the script does not log tokens) |")
out.extend(cost_lines)
out.append(f"| **total** | | {total_in:,} | {total_out:,} | **${total_cost + floors_cost:.2f}** |")
out.append(f"\nEval cost at the proxy's rates for claude-sonnet-5-5 ($2/M in, $10/M out). Wall times per run are in RUNS.md.")

out.append("\n## Flags\n")
out.extend(f"- {f}" for f in flags) if flags else out.append("- none of the three automatic checks tripped")

(PILOT / "SUMMARY_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
