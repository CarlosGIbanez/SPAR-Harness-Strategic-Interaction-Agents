"""usage: uv run python logs/full/evalstats.py <eval paths...>  -> one line per eval"""
import glob, statistics, sys
from inspect_ai.log import read_eval_log
RATES = {"claude-sonnet-5-5": (2e-6, 10e-6), "claude-haiku-5-5": (1e-7, 5e-7), "luna": (1.1e-7, 5.5e-7), "sol": (2.2e-6, 1.1e-5)}  # from the proxy /model/info
def m(xs):
    xs = [x for x in xs if x is not None]; return statistics.mean(xs) if xs else None
def f(x, d=3): return "–" if x is None else f"{x:.{d}f}"
tot = 0.0
for p in sys.argv[1:]:
    l = read_eval_log(p); u = l.stats.model_usage
    inp = sum(x.input_tokens for x in u.values()); out = sum(x.output_tokens for x in u.values())
    name = next(iter(u)).split("/")[-1]; ri, ro = RATES.get(name, (2e-6, 10e-6))
    c = inp * ri + out * ro; tot += c
    o = [s.scores["outcome_scorer"].value for s in l.samples]; se = [s.scores["seat_scorer"].value for s in l.samples]
    print(f"{p.split('/')[-2]:<20} {l.status:<8} n={len(l.samples):<3} fight={f(m([x['fight'] for x in o]))} agree={f(m([x['agreement'] for x in o]))} eff={f(m([x['efficiency'] for x in o]))} "
          f"rounds={f(m([len(s.store['episode']['rounds']) for s in l.samples]),2)} in_range={f(m([x['in_range'] for x in o]),2)} "
          f"compr={f(m([v[k] for v in se for k in ('a_comprehension','b_comprehension')]),2)} default={f(m([v[k] for v in se for k in ('a_default_rate','b_default_rate')]),3)} "
          f"floor={f(m([v[k] for v in se for k in ('a_floor','b_floor')]),2)} fidelity={f(m([v[k] for v in se for k in ('a_fidelity','b_fidelity')]),2)} compliance={f(m([v[k] for v in se for k in ('a_compliance','b_compliance')]),2)} "
          f"in={inp} out={out} ${c:.2f} ({name})")
print(f"total ${tot:.2f}")
