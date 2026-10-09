import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "results/generated/depth_explorer"
HERE.mkdir(parents=True, exist_ok=True)
def R(p): return str(ROOT / p)
FILES={"harmbench":("phase2_attacks.json","no_balcony_attacks.json"),
       "advbench":("advbench_attacks.json","gcg_no_balcony_advbench_attacks.json")}
MODELS={"1b":("results/archive/1b",["4","8","12","full"],16),
        "7b":("results/archive/7b",["15","18","21","full"],32),
        "8b":("results/archive/8b",["15","18","21","full"],32)}
out={}
for m,(d,exits,layers) in MODELS.items():
    for bench,(bf,basef) in FILES.items():
        bal=json.load(open(R(f"{d}/{bf}")))
        base=json.load(open(R(f"{d}/{basef}")))
        arms=[(e,bal[e]) for e in exits]+[("baseline", base["no_balcony"] if "no_balcony" in base else list(base.values())[0])]
        for label,entries in arms:
            init=[e["loss_curve"][0] for e in entries if e.get("loss_curve")]
            fin =[e["loss_curve"][-1] for e in entries if e.get("loss_curve")]
            # late slope: mean improvement per step over last 20 steps
            late=[(e["loss_curve"][-21]-e["loss_curve"][-1])/20 for e in entries if e.get("loss_curve") and len(e["loss_curve"])>21]
            i=sum(init)/len(init); f=sum(fin)/len(fin); l=sum(late)/len(late)
            out[f"{m}|{bench}|{label}"]=dict(init=i,final=f,late=l,n=len(init))
            print(f"{m:3} {bench:9} {label:9} init {i:5.2f} final {f:5.2f} cut {i-f:5.2f} ({100*(i-f)/i:4.1f}%) late-slope {l:.4f}/step")
json.dump(out,open(str(HERE / "loss.json"),"w"),indent=1)
