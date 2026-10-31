"""Compare two floor sweeps (old vs new instruction wording) and plot them.

usage: uv run python logs/pilot/compare_floors.py OLD.csv NEW.csv OUTPREFIX
"""

import csv
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["level"] = int(r["level"])
        for k in ("fight_value", "floor", "aggressiveness"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
        r["fidelity"] = int(r["fidelity"]) if r["fidelity"] not in ("", "None") else None
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def collision_rate(rows):
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r["level"], r["version"], r["seed"], r["repeat"])][r["position"]] = r["floor"]
    both = [p for p in pairs.values() if p.get("alpha") is not None and p.get("beta") is not None]
    return mean([p["alpha"] + p["beta"] > 1 + 1e-9 for p in both]), len(both)


def by_level(rows):
    out = {}
    for lv in sorted({r["level"] for r in rows}):
        at = [r for r in rows if r["level"] == lv]
        stated = [r for r in at if r["floor"] is not None]
        agg = [r["aggressiveness"] for r in stated]
        out[lv] = dict(
            n=len(at),
            mean=mean(agg),
            median=statistics.median(agg) if agg else None,
            sd=statistics.stdev(agg) if len(agg) > 1 else None,
            fidelity=mean([r["fidelity"] for r in stated]),
            unparseable=(len(at) - len(stated)) / len(at),
            collision=collision_rate(at)[0],
        )
    return out


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


old, new = load(sys.argv[1]), load(sys.argv[2])
prefix = sys.argv[3]
O, N = by_level(old), by_level(new)

lines = ["| level | n | mean floor − fight (old) | mean (new) | Δ | median old | median new | fidelity old | fidelity new | unparseable old | new | collision old | new |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for lv in sorted(O):
    o, n = O[lv], N[lv]
    d = None if o["mean"] is None or n["mean"] is None else n["mean"] - o["mean"]
    lines.append(f"| {lv} | {n['n']} | {f(o['mean'])} | {f(n['mean'])} | {f(d)} | {f(o['median'])} | {f(n['median'])} | "
                 f"{f(o['fidelity'],2)} | {f(n['fidelity'],2)} | {f(o['unparseable'],2)} | {f(n['unparseable'],2)} | "
                 f"{f(o['collision'],2)} | {f(n['collision'],2)} |")
co, nco = collision_rate(old)
cn, ncn = collision_rate(new)
lines.append(f"\nOverall: unparseable old {f(1 - sum(r['floor'] is not None for r in old) / len(old), 3)}, "
             f"new {f(1 - sum(r['floor'] is not None for r in new) / len(new), 3)}; "
             f"collision old {f(co, 3)} (n={nco}), new {f(cn, 3)} (n={ncn}).")
table = "\n".join(lines)
print(table)
open(prefix + "_table.md", "w").write(table + "\n")

levels = sorted(O)
# Aggressiveness: box plots old vs new, by level
fig, ax = plt.subplots(figsize=(8, 4))
data_o = [[r["aggressiveness"] for r in old if r["level"] == lv and r["floor"] is not None] for lv in levels]
data_n = [[r["aggressiveness"] for r in new if r["level"] == lv and r["floor"] is not None] for lv in levels]
pos = range(len(levels))
ax.boxplot(data_o, positions=[p - 0.18 for p in pos], widths=0.3, patch_artist=True,
           boxprops=dict(facecolor="#bbbbbb"), medianprops=dict(color="black"))
ax.boxplot(data_n, positions=[p + 0.18 for p in pos], widths=0.3, patch_artist=True,
           boxprops=dict(facecolor="#4c9be8"), medianprops=dict(color="black"))
ax.axhline(0, color="red", lw=0.8, ls="--")
ax.set_xticks(list(pos)); ax.set_xticklabels(levels)
ax.set_xlabel("instruction level"); ax.set_ylabel("floor − fight value")
ax.set_title("Aggressiveness by level: grey = old wording, blue = new wording")
fig.tight_layout(); fig.savefig(prefix + "_aggressiveness.png", dpi=130)

# Fidelity rate by level
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(levels, [O[lv]["fidelity"] for lv in levels], "o-", color="#888", label="old wording")
ax.plot(levels, [N[lv]["fidelity"] for lv in levels], "o-", color="#4c9be8", label="new wording")
ax.set_ylim(0, 1.05); ax.set_xticks(levels)
ax.set_xlabel("instruction level"); ax.set_ylabel("fidelity rate"); ax.legend()
ax.set_title("Fidelity: floor on the predicted side of the fight value")
fig.tight_layout(); fig.savefig(prefix + "_fidelity.png", dpi=130)
print("wrote", prefix + "_table.md", prefix + "_aggressiveness.png", prefix + "_fidelity.png")
