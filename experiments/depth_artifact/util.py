import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # repo root, where results*/ live
def R(p): return str(ROOT / p)
MODELS={"1b":("results",["4","8","12","full","no_balcony"],16),
        "7b":("results_7b",["15","18","21","full","no_balcony"],32),
        "8b":("results_8b",["15","18","21","full","no_balcony"],32)}
CHANCE={"hellaswag":.25,"arc_challenge":.25,"arc_easy":.25,"mmlu":.25,"winogrande":.5,"piqa":.5}
TASKS=["hellaswag","arc_challenge","arc_easy","mmlu","winogrande","piqa"]

# Retention chance-corrects each task as (score - chance) / (full - chance), so a
# task where the full model sits at chance divides by noise. The 1B's full model
# scores 0.2555 on MMLU against a 0.25 floor, a denominator of 0.55pp, and its
# MMLU term then swings between -99% and +28% across exits with no ordering.
# MMLU is therefore excluded from retention for every model, so the three stay
# comparable. The per-task MMLU accuracies are still reported in full.
RETENTION_TASKS=[t for t in TASKS if t!="mmlu"]

def collapse(d):
    """per-task score.

    MMLU comes from lm-eval's own `mmlu` aggregate, which is what
    make_paper_assets.py puts in the paper tables. Do not average the mmlu_*
    keys instead: that set mixes the 57 leaf subtasks with 4 group-level
    rollups, and even excluding those an unweighted mean over subtasks of very
    different sizes disagrees with the aggregate by up to 1.7pp.
    """
    out={}
    for t in TASKS:
        out[t]=d[t]["value"]
    return out

res={}
for m,(dd,arms,layers) in MODELS.items():
    clean=json.load(open(R(f"{dd}/utility_eval.json")))
    att=json.load(open(R(f"{dd}/utility_eval_attacked.json")))
    base=collapse(clean["no_balcony"])
    print(f"=== {m} ===")
    for a in arms:
        c=collapse(clean[a])
        rets=[(c[t]-CHANCE[t])/(base[t]-CHANCE[t]) for t in RETENTION_TASKS]
        ret=100*sum(rets)/len(rets)
        depth = 1.0 if a in ("full","no_balcony") else int(a)/layers
        ac = collapse(att[a]) if a in att else None
        res[(m,a)]=dict(clean=c, attacked=ac, retention=ret, depth=depth)
        print(f"  {a:11} depth={depth:.3f} retention={ret:6.1f}%  hellaswag={c['hellaswag']:.4f} "
              f"arc_e={c['arc_easy']:.4f} mmlu={c['mmlu']:.4f} wino={c['winogrande']:.4f} piqa={c['piqa']:.4f} arc_c={c['arc_challenge']:.4f}"
              + ("" if ac else "   [NO ATTACKED]"))
    print(f"  attacked arms present: {sorted(att.keys())}")
json.dump({f"{k[0]}|{k[1]}":v for k,v in res.items()},
    open(str(HERE / "util.json"),"w"),indent=1)
