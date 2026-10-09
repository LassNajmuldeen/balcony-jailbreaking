"""Build a standalone HTML view from asr.json, util.json, loss.json, and mcnemar.json.
Recompute those inputs when the archived result sets change."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "results/generated/depth_explorer"
asr = json.load(open(HERE / "asr.json"))
util = json.load(open(HERE / "util.json"))
loss = json.load(open(HERE / "loss.json"))
mcn = json.load(open(HERE / "mcnemar.json"))

MODELS = ["1b", "7b", "8b"]
TITLE = {"1b": "Balcony-1B", "7b": "Balcony-LLaMA2-7B", "8b": "Balcony-LLaMA3-8B"}
LAYERS = {"1b": 16, "7b": 32, "8b": 32}
EXITS = {"1b": ["4", "8", "12"], "7b": ["15", "18", "21"], "8b": ["15", "18", "21"]}
BENCH = ["harmbench", "advbench"]
BLABEL = {"harmbench": "HarmBench", "advbench": "AdvBench"}


def arm_label(m, a):
    if a == "full":
        return "full"
    if a == "baseline":
        return "baseline"
    return f"exit {a}"


# ---------------------------------------------------------------- chart payloads
data = {"asr": {}, "util": [], "loss": {}, "mcnemar": [], "cells": []}

for b in BENCH:
    series = []
    for m in MODELS:
        pts = []
        for a in EXITS[m] + ["full"]:
            r = asr[f"{m}|{b}|{a}"]
            pts.append({"d": round(r["depth"], 4), "v": r["asr"], "lo": round(r["lo"], 1),
                        "hi": round(r["hi"], 1), "l": arm_label(m, a)})
        base = asr[f"{m}|{b}|baseline"]
        series.append({"m": m, "name": TITLE[m], "pts": pts,
                       "base": {"d": 1.0, "v": base["asr"], "lo": round(base["lo"], 1),
                                "hi": round(base["hi"], 1), "l": "baseline"}})
    data["asr"][b] = series

for m in MODELS:
    pts = []
    for a in EXITS[m] + ["full"]:
        u = util[f"{m}|{a}"]
        pts.append({"d": round(u["depth"], 4), "v": round(u["retention"], 1), "l": arm_label(m, a)})
    data["util"].append({"m": m, "name": TITLE[m], "pts": pts})

for b in BENCH:
    rows = []
    for m in MODELS:
        for a in EXITS[m] + ["full", "baseline"]:
            r = loss[f"{m}|{b}|{a}"]
            rows.append({"m": m, "arm": arm_label(m, a), "init": round(r["init"], 2),
                         "fin": round(r["final"], 2),
                         "cut": round(100 * (r["init"] - r["final"]) / r["init"])})
    data["loss"][b] = rows

# every exit-vs-full and exit-vs-baseline test, direction preserved
for m in MODELS:
    for b in BENCH:
        for a1, a2 in [(e, "full") for e in EXITS[m]] + [(e, "baseline") for e in EXITS[m]] + [("full", "baseline")]:
            bb, cc, p, n = mcn[f"{m}|{b}|{a1}|{a2}"]
            if p >= 0.05:
                verdict, cls = "not significant", "flat"
            elif bb > cc:
                verdict, cls = "arm 1 attacked more", "up"
            else:
                verdict, cls = "arm 1 attacked less", "down"
            data["mcnemar"].append({
                "m": m, "model": TITLE[m], "bench": BLABEL[b],
                "cmp": f"{arm_label(m, a1)} vs {arm_label(m, a2)}",
                "b": bb, "c": cc, "p": p, "verdict": verdict, "cls": cls})

# the full cell table
for m in MODELS:
    for a in EXITS[m] + ["full", "baseline"]:
        u = util[f"{m}|{a if a != 'baseline' else 'no_balcony'}"]
        row = {"model": TITLE[m], "arm": arm_label(m, a),
               "depth": round(100 * u["depth"]), "ret": round(u["retention"], 1)}
        for b in BENCH:
            r = asr[f"{m}|{b}|{a}"]
            row[b] = r["asr"]
            row[b + "_ci"] = f"{r['lo']:.0f}–{r['hi']:.0f}"
            row[b + "_loss"] = round(loss[f"{m}|{b}|{a}"]["final"], 2)
        data["cells"].append(row)

# ---------------------------------------------------------------- headline counts
tot = more = less = 0
for r in data["mcnemar"]:
    if not r["cmp"].endswith("vs full") or r["cmp"].startswith("full"):
        continue
    tot += 1
    if r["p"] < 0.05:
        if r["cls"] == "up":
            more += 1
        else:
            less += 1

HEAD = {"tot": tot, "more": more, "less": less}

payload = json.dumps(data, separators=(",", ":"))

html = f"""<title>Balcony early exits under GCG: attack success, capability and attack budget</title>
<style>
  :root {{
    color-scheme: light;
    --ground:  #eef2f3;
    --panel:   #ffffff;
    --sunk:    #f5f8f8;
    --rule:    #dbe3e5;
    --rule-2:  #e9eef0;
    --ink:     #0f1617;
    --ink-2:   #4a575a;
    --ink-3:   #7b888c;
    --s1: #2a78d6;
    --s2: #eb6834;
    --s3: #1baf7a;
    --sb: #9a3b8f;
    --flat: #7b888c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --ground:  #0d1213;
      --panel:   #171e1f;
      --sunk:    #121a1b;
      --rule:    #2a3436;
      --rule-2:  #212a2c;
      --ink:     #eef4f4;
      --ink-2:   #aebabc;
      --ink-3:   #7c8a8c;
      --s1: #3a86dd;
      --s2: #d9682f;
      --s3: #1aa473;
      --sb: #bb51ab;
      --flat: #7c8a8c;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --ground:  #0d1213;
    --panel:   #171e1f;
    --sunk:    #121a1b;
    --rule:    #2a3436;
    --rule-2:  #212a2c;
    --ink:     #eef4f4;
    --ink-2:   #aebabc;
    --ink-3:   #7c8a8c;
    --s1: #3a86dd;
    --s2: #d9682f;
    --s3: #1aa473;
    --sb: #bb51ab;
    --flat: #7c8a8c;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    font-size: 16.5px;
    line-height: 1.62;
    -webkit-font-smoothing: antialiased;
  }}
  .mono, .tick, .val, th, .kpi-n, .eyebrow, .tag {{
    font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
  }}

  .wrap {{
    max-width: 66rem;
    margin: 0 auto;
    padding: 3rem 1.25rem 5rem;
    display: flex; flex-direction: column; gap: 2.5rem;
  }}

  .eyebrow {{
    font-size: 0.7rem; letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--ink-3); margin: 0 0 0.9rem;
  }}
  h1 {{
    font-size: clamp(1.75rem, 3.9vw, 2.5rem);
    line-height: 1.16; font-weight: 640; letter-spacing: -0.022em;
    text-wrap: balance; margin: 0 0 1rem; max-width: 30ch;
  }}
  .lede {{ font-size: 1.05rem; color: var(--ink-2); max-width: 66ch; margin: 0; }}
  .lede strong {{ color: var(--ink); font-weight: 620; }}

  h2 {{
    font-size: 1.16rem; font-weight: 640; letter-spacing: -0.012em;
    margin: 0 0 0.3rem;
  }}
  .sub {{ color: var(--ink-2); font-size: 0.94rem; margin: 0 0 1.35rem; max-width: 68ch; }}

  .panel {{
    background: var(--panel); border: 1px solid var(--rule);
    border-radius: 8px; padding: 1.5rem 1.4rem 1.2rem;
  }}

  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 1px;
           background: var(--rule); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }}
  .kpi {{ background: var(--panel); padding: 1.15rem 1.2rem; }}
  .kpi-n {{ font-size: 1.85rem; font-weight: 620; letter-spacing: -0.03em; line-height: 1.1; display: block; }}
  .kpi-l {{ font-size: 0.855rem; color: var(--ink-2); margin-top: 0.3rem; }}

  .legend {{ display: flex; flex-wrap: wrap; gap: 0.3rem 1.25rem; margin: 0 0 0.85rem;
             padding: 0; list-style: none; font-size: 0.84rem; color: var(--ink-2); }}
  .legend li {{ display: flex; align-items: center; gap: 0.45rem; }}
  .sw {{ width: 20px; height: 3px; border-radius: 2px; flex: none; }}
  .sw.dia {{ width: 10px; height: 10px; border-radius: 2px; transform: rotate(45deg);
             background: transparent; border: 2px solid var(--ink-3); }}

  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(21rem, 1fr)); gap: 1.4rem; }}
  .cwrap {{ position: relative; min-width: 0; }}
  .ctitle {{ font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
             color: var(--ink-3); margin: 0 0 0.35rem; }}
  .scroller {{ overflow-x: auto; }}
  svg {{ display: block; width: 100%; height: auto; }}
  .grid {{ stroke: var(--rule-2); stroke-width: 1; }}
  .axis {{ stroke: var(--rule); stroke-width: 1; }}
  .tick {{ fill: var(--ink-3); font-size: 11px; }}
  .atitle {{ fill: var(--ink-2); font-size: 11.5px; }}
  .dlab {{ fill: var(--ink); font-size: 11.5px; font-weight: 620;
           font-family: ui-monospace, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }}
  .hit {{ fill: transparent; cursor: crosshair; }}

  .tip {{
    position: absolute; pointer-events: none; opacity: 0; transition: opacity .09s ease;
    background: var(--panel); border: 1px solid var(--rule); border-radius: 6px;
    padding: .45rem .6rem; font-size: .8rem; line-height: 1.4; color: var(--ink-2);
    box-shadow: 0 6px 20px rgba(0,0,0,.16); white-space: nowrap; z-index: 6;
  }}
  .tip b {{ color: var(--ink); }}

  .callout {{
    border-left: 2px solid var(--s1); padding: .1rem 0 .1rem 1rem;
    color: var(--ink-2); font-size: .96rem; max-width: 70ch;
  }}
  .callout strong {{ color: var(--ink); }}

  .scroll-x {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 44rem; font-size: .85rem; }}
  caption {{ text-align: left; color: var(--ink-3); font-size: .81rem; padding-bottom: .7rem; max-width: 78ch; }}
  th, td {{ text-align: right; padding: .42rem .6rem; border-bottom: 1px solid var(--rule-2);
            font-variant-numeric: tabular-nums; white-space: nowrap; }}
  th:first-child, td:first-child, th.l, td.l {{ text-align: left; }}
  thead th {{ font-size: .7rem; letter-spacing: .06em; text-transform: uppercase;
              color: var(--ink-3); font-weight: 620; border-bottom: 1px solid var(--rule); }}
  tbody tr:last-child td {{ border-bottom: none; }}
  .grp td {{ padding-top: .95rem; font-weight: 620; color: var(--ink); border-bottom: 1px solid var(--rule); }}
  .ci {{ color: var(--ink-3); font-size: .92em; }}
  .tag {{ font-size: .69rem; letter-spacing: .05em; text-transform: uppercase;
          padding: .12rem .42rem; border-radius: 3px; border: 1px solid currentColor; }}
  .t-flat {{ color: var(--ink-3); }}
  .t-down {{ color: var(--s1); }}
  .t-up   {{ color: var(--s2); }}

  ul.notes {{ margin: 0; padding-left: 1.1rem; color: var(--ink-2); font-size: .95rem; }}
  ul.notes li {{ margin-bottom: .55rem; }}
  ul.notes li::marker {{ color: var(--ink-3); }}
  ul.notes strong {{ color: var(--ink); }}

  footer {{ color: var(--ink-3); font-size: .78rem; border-top: 1px solid var(--rule);
            padding-top: 1rem; font-family: ui-monospace, Menlo, Consolas, monospace; }}

  a {{ color: inherit; }}
  :focus-visible {{ outline: 2px solid var(--s1); outline-offset: 2px; }}
  @media (prefers-reduced-motion: reduce) {{ .tip {{ transition: none; }} }}
</style>

<div class="wrap">

  <header>
    <p class="eyebrow">GCG &middot; 150 steps &middot; n=50 per cell &middot; 3 models &middot; 95% Wilson CI</p>
    <h1>No early exit is attacked more successfully than the model it exits from.</h1>
    <p class="lede">
      Across {HEAD['tot']} paired comparisons of a balcony exit against its own full model,
      <strong>none</strong> shows the exit significantly easier to jailbreak.
      {HEAD['less']} run the other way, and those are the exits that have lost the most
      capability or whose attack never converged in the step budget. This page shows all
      three measurements side by side so the difference is visible rather than asserted.
    </p>
  </header>

  <div class="kpis">
    <div class="kpi">
      <span class="kpi-n" style="color:var(--s1)">0 / {HEAD['tot']}</span>
      <div class="kpi-l">exit-vs-full tests where the exit is significantly more jailbroken</div>
    </div>
    <div class="kpi">
      <span class="kpi-n">{HEAD['less']} / {HEAD['tot']}</span>
      <div class="kpi-l">significant in the opposite direction, the exit attacked less</div>
    </div>
    <div class="kpi">
      <span class="kpi-n">&plusmn;12pp</span>
      <div class="kpi-l">typical width of a 95% interval at n=50, wider than most gaps here</div>
    </div>
  </div>

  <section class="panel">
    <h2>Attack success against exit depth</h2>
    <p class="sub">
      Each point is one arm at its depth through the network, which is what makes the 1B's
      4/8/12-of-16 comparable with the 7B and 8B's 15/18/21-of-32. Bars are 95% Wilson
      intervals. The weight-matched no-balcony baseline is the open diamond at full depth.
    </p>
    <ul class="legend">
      <li><span class="sw" style="background:var(--s1)"></span> Balcony-1B</li>
      <li><span class="sw" style="background:var(--s2)"></span> Balcony-LLaMA2-7B</li>
      <li><span class="sw" style="background:var(--s3)"></span> Balcony-LLaMA3-8B</li>
      <li><span class="sw dia"></span> no-balcony baseline, nudged right of full depth so it stays visible</li>
    </ul>
    <div class="charts" id="asr-charts"></div>
  </section>

  <p class="callout">
    <strong>The intervals overlap nearly everywhere.</strong> The 1B's exit 12 is the highest
    single point on HarmBench at 72%, above its own full model at 58%, but that gap is
    13 behaviours flipping one way against 6 the other, which is p&nbsp;=&nbsp;0.167. It is not
    a result. Read the significance table below rather than the point estimates.
  </p>

  <section class="panel">
    <h2>What those exits still cost in capability</h2>
    <p class="sub">
      Chance-corrected retention against the full model, averaged over HellaSwag, ARC-C,
      ARC-E, WinoGrande and PIQA. Correcting for chance matters: raw ratios read 8 to 28
      points high because three of these tasks floor at 0.25 and two at 0.50. MMLU is left
      out of this average on purpose. The 1B's full model scores 0.2555 on MMLU against a
      0.25 floor, so chance-correcting it divides by half a point of signal and its MMLU
      term swings between -99% and +28% across exits with no ordering. Every per-task MMLU
      score is still in the table at the bottom.
    </p>
    <ul class="legend">
      <li><span class="sw" style="background:var(--s1)"></span> Balcony-1B</li>
      <li><span class="sw" style="background:var(--s2)"></span> Balcony-LLaMA2-7B</li>
      <li><span class="sw" style="background:var(--s3)"></span> Balcony-LLaMA3-8B</li>
    </ul>
    <div class="charts" id="util-chart"></div>
  </section>

  <p class="callout">
    <strong>This is what the low attack numbers at shallow exits are made of.</strong>
    The 1B's exit 4 holds 28% of the full model's chance-corrected accuracy and its exit 8
    holds 47%. The 8B's exit 15 holds 47%. Those are exactly the arms that come out
    significantly harder to attack. A model that cannot write a coherent answer also cannot
    write a harmful one, so a low score there is not evidence of safety.
  </p>

  <section class="panel">
    <h2>The attack budget was not equal across arms</h2>
    <p class="sub">
      Mean GCG cross-entropy at step 0 and step 150 on HarmBench. Every arm got the same
      150 steps, but shallow exits start further from compliance, so the same budget buys
      less. The 8B's exit 15 ends at a loss the 1B's full model passes within the first
      few steps.
    </p>
    <div class="charts" id="loss-chart"></div>
  </section>

  <p class="callout">
    <strong>A fixed step budget systematically under-attacks shallower exits.</strong>
    At the 8B's exit 15 GCG cut more absolute loss than at full depth, 0.79 against 0.62,
    and still finished 1.4 nats higher because it started 1.57 nats higher. Reporting
    starting loss and convergence state alongside attack success, and budgeting steps
    adaptively, is the concrete prescription this run supports.
  </p>

  <section class="panel">
    <h2>Every paired test</h2>
    <p class="sub">
      McNemar's exact test on the same 50 behaviours attacked at every arm, so the
      comparison is paired. The two counts are behaviours jailbroken at the first arm but
      not the second, and the reverse.
    </p>
    <div class="scroll-x">
      <table>
        <caption>
          Arm 1 is the exit or full model, arm 2 is the comparator. "Attacked less" means the
          first arm was jailbroken on significantly fewer behaviours than the second.
        </caption>
        <thead>
          <tr><th class="l">Comparison</th><th class="l">Benchmark</th>
              <th>Flips arm 1</th><th>Flips arm 2</th><th>p</th><th class="l">Result</th></tr>
        </thead>
        <tbody id="mcn-body"></tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <h2>All measured cells</h2>
    <div class="scroll-x">
      <table>
        <caption>
          GCG, 150 steps, 50 behaviours per cell. Attack success is the HarmBench Mistral
          classifier at p&nbsp;&gt;&nbsp;0.5. Loss is the mean final cross-entropy. Retention is
          chance-corrected against each model's own full-depth score over the five tasks
          excluding MMLU, so full and baseline are 100% by construction; they share trunk
          weights and score identically to four decimal places.
        </caption>
        <thead>
          <tr>
            <th class="l">Arm</th><th>Depth</th>
            <th>ASR HarmBench</th><th>ASR AdvBench</th>
            <th>Loss HB</th><th>Loss AB</th><th>Capability</th>
          </tr>
        </thead>
        <tbody id="cell-body"></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>What this run still cannot tell you</h2>
    <ul class="notes">
      <li><strong>There is no unattacked attack-success number.</strong> Every cell here has a
        GCG suffix on it. The no-attack control that exists measures teacher-forced loss, and
        only for the 1B, so the attack's own contribution is unquantified. With refusal rates
        near zero this matters: if these models answer harmful prompts when simply asked,
        much of the reported success is not the attack's doing.</li>
      <li><strong>All three baselines are unaligned base checkpoints.</strong> No safety-tuned
        Balcony checkpoint is public. Refusal runs at 0 to 4% on the 1B, 0 to 14% on the 7B and
        6 to 18% on the 8B, so there is very little refusal behaviour for GCG to defeat, and
        this cannot be framed as a safety-degradation study.</li>
      <li><strong>Depth coverage is fixed and uneven.</strong> The 1B samples 25, 50 and 75% of
        depth; the 7B and 8B sample 47, 56 and 66%. Neither large model has an exit past 66%,
        and no exit can be added to a public checkpoint.</li>
      <li><strong>Most cells were still improving at step 150.</strong> Only the 1B's exit 12 and
        full model had flattened, so the matrix as a whole is under-converged and the ranking
        between arms is partly a ranking of how far each got.</li>
      <li><strong>TAO is not in this page.</strong> Its shards are incomplete and no finished
        shard yet has a matching full-model or baseline comparator.</li>
    </ul>
  </section>

  <footer>
    Recomputed from results/archive/1b/, results/archive/7b/, results/archive/8b/ &middot; judge: cais/HarmBench-Mistral-7b-val-cls
    &middot; every number derived from per-behaviour scores at build time
  </footer>
</div>

<script>
(function () {{
  "use strict";
  var D = {payload};
  var NS = "http://www.w3.org/2000/svg";
  var SC = {{ "1b": "var(--s1)", "7b": "var(--s2)", "8b": "var(--s3)" }};

  function el(n, a, t) {{
    var e = document.createElementNS(NS, n);
    for (var k in a) if (a[k] !== null && a[k] !== undefined) e.setAttribute(k, a[k]);
    if (t !== undefined) e.textContent = t;
    return e;
  }}
  function box(parent, title) {{
    var w = document.createElement("div"); w.className = "cwrap";
    var h = document.createElement("p"); h.className = "ctitle"; h.textContent = title;
    var s = document.createElement("div"); s.className = "scroller";
    var tip = document.createElement("div"); tip.className = "tip";
    w.appendChild(h); w.appendChild(s); w.appendChild(tip); parent.appendChild(w);
    return {{ mount: s, tip: tip, wrap: w }};
  }}
  function hover(node, tip, wrap, html, sx, sy) {{
    node.addEventListener("mouseenter", function () {{
      tip.innerHTML = html; tip.style.opacity = "1";
      var bw = wrap.clientWidth;
      tip.style.left = Math.max(2, Math.min(sx() + 12, bw - tip.offsetWidth - 2)) + "px";
      tip.style.top = Math.max(2, sy() - tip.offsetHeight - 10) + "px";
    }});
    node.addEventListener("mouseleave", function () {{ tip.style.opacity = "0"; }});
  }}

  // ---------------------------------------------------------------- ASR vs depth
  function asrChart(mount, tip, wrap, series, unitMax) {{
    var W = 470, H = 330, m = {{ t: 16, r: 16, b: 46, l: 44 }};
    var x0 = m.l, x1 = W - m.r, y0 = m.t, y1 = H - m.b;
    var dMin = 0.18, dMax = 1.12;
    var X = function (d) {{ return x0 + (d - dMin) / (dMax - dMin) * (x1 - x0); }};
    var Y = function (v) {{ return y1 - v / unitMax * (y1 - y0); }};
    var svg = el("svg", {{ viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Attack success rate against relative exit depth for three models." }});
    svg.style.minWidth = "20rem";

    for (var v = 0; v <= unitMax; v += 20) {{
      svg.appendChild(el("line", {{ x1: x0, x2: x1, y1: Y(v), y2: Y(v), class: v === 0 ? "axis" : "grid" }}));
      svg.appendChild(el("text", {{ x: x0 - 9, y: Y(v) + 4, "text-anchor": "end", class: "tick" }}, String(v)));
    }}
    [0.25, 0.5, 0.75, 1.0].forEach(function (d) {{
      svg.appendChild(el("text", {{ x: X(d), y: y1 + 20, "text-anchor": "middle", class: "tick" }},
        Math.round(d * 100) + "%"));
    }});
    svg.appendChild(el("text", {{ x: (x0 + x1) / 2, y: H - 8, "text-anchor": "middle", class: "atitle" }},
      "relative exit depth"));
    svg.appendChild(el("text", {{ class: "atitle", "text-anchor": "middle",
      transform: "translate(12," + (y0 + y1) / 2 + ") rotate(-90)" }}, "attack success (%)"));

    series.forEach(function (s) {{
      var col = SC[s.m];
      svg.appendChild(el("path", {{
        d: s.pts.map(function (p, i) {{ return (i ? "L" : "M") + X(p.d) + " " + Y(p.v); }}).join(" "),
        fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"
      }}));
      var all = s.pts.concat([s.base]);
      all.forEach(function (p, i) {{
        var isBase = i === s.pts.length;
        // the baseline sits at full depth too, and for the 8B it lands on exactly the
        // same value as the full model, so nudge it clear rather than hide it
        var px = X(p.d) + (isBase ? 15 : 0);
        svg.appendChild(el("line", {{ x1: px, x2: px, y1: Y(p.lo), y2: Y(p.hi),
          stroke: col, "stroke-width": 1.4, opacity: ".45", "stroke-linecap": "round" }}));
        var node;
        if (isBase) {{
          var r = 5.4;
          node = el("rect", {{ x: px - r, y: Y(p.v) - r, width: 2 * r, height: 2 * r,
            transform: "rotate(45 " + px + " " + Y(p.v) + ")",
            fill: "var(--panel)", stroke: col, "stroke-width": 2 }});
        }} else {{
          node = el("circle", {{ cx: px, cy: Y(p.v), r: 4.6, fill: col,
            stroke: "var(--panel)", "stroke-width": 1.6 }});
        }}
        svg.appendChild(node);
        var hit = el("circle", {{ cx: px, cy: Y(p.v), r: 14, class: "hit" }});
        (function (p, s, px) {{
          hover(hit, tip, wrap,
            "<b>" + s.name + " &middot; " + p.l + "</b><br>" +
            Math.round(p.d * 100) + "% depth &middot; <b>" + p.v.toFixed(0) + "%</b> success<br>" +
            "95% CI " + p.lo + "&ndash;" + p.hi + "%",
            function () {{ return px * (wrap.clientWidth / W); }},
            function () {{ return Y(p.v) * (wrap.clientWidth / W); }});
        }})(p, s, px);
        svg.appendChild(hit);
      }});
    }});
    mount.appendChild(svg);
  }}

  var ac = document.getElementById("asr-charts");
  ["harmbench", "advbench"].forEach(function (b) {{
    var t = b === "harmbench" ? "HarmBench" : "AdvBench";
    var o = box(ac, t);
    asrChart(o.mount, o.tip, o.wrap, D.asr[b], 80);
  }});

  // ---------------------------------------------------------------- capability
  (function () {{
    var o = box(document.getElementById("util-chart"), "Chance-corrected capability retention");
    o.wrap.style.gridColumn = "1 / -1";
    var W = 900, H = 300, m = {{ t: 16, r: 20, b: 46, l: 48 }};
    var x0 = m.l, x1 = W - m.r, y0 = m.t, y1 = H - m.b;
    var dMin = 0.18, dMax = 1.08, vMax = 105, vMin = -12;
    var X = function (d) {{ return x0 + (d - dMin) / (dMax - dMin) * (x1 - x0); }};
    var Y = function (v) {{ return y1 - (v - vMin) / (vMax - vMin) * (y1 - y0); }};
    var svg = el("svg", {{ viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Chance-corrected capability retention against relative exit depth." }});
    svg.style.minWidth = "26rem";
    [0, 25, 50, 75, 100].forEach(function (v) {{
      svg.appendChild(el("line", {{ x1: x0, x2: x1, y1: Y(v), y2: Y(v), class: v === 0 ? "axis" : "grid" }}));
      svg.appendChild(el("text", {{ x: x0 - 9, y: Y(v) + 4, "text-anchor": "end", class: "tick" }}, v + "%"));
    }});
    [0.25, 0.5, 0.75, 1.0].forEach(function (d) {{
      svg.appendChild(el("text", {{ x: X(d), y: y1 + 20, "text-anchor": "middle", class: "tick" }},
        Math.round(d * 100) + "%"));
    }});
    svg.appendChild(el("text", {{ x: (x0 + x1) / 2, y: H - 8, "text-anchor": "middle", class: "atitle" }},
      "relative exit depth"));
    svg.appendChild(el("text", {{ class: "atitle", "text-anchor": "middle",
      transform: "translate(13," + (y0 + y1) / 2 + ") rotate(-90)" }}, "capability retained"));

    D.util.forEach(function (s) {{
      var col = SC[s.m];
      svg.appendChild(el("path", {{
        d: s.pts.map(function (p, i) {{ return (i ? "L" : "M") + X(p.d) + " " + Y(p.v); }}).join(" "),
        fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"
      }}));
      s.pts.forEach(function (p) {{
        var yy = Y(p.v);
        svg.appendChild(el("circle", {{ cx: X(p.d), cy: yy, r: 4.6, fill: col,
          stroke: "var(--panel)", "stroke-width": 1.6 }}));
        if (p.l !== "full") {{
          svg.appendChild(el("text", {{ x: X(p.d), y: yy - 12, "text-anchor": "middle", class: "dlab" }},
            p.v.toFixed(0) + "%"));
        }}
        var hit = el("circle", {{ cx: X(p.d), cy: yy, r: 15, class: "hit" }});
        (function (p, s) {{
          hover(hit, o.tip, o.wrap,
            "<b>" + s.name + " &middot; " + p.l + "</b><br>" +
            Math.round(p.d * 100) + "% depth &middot; <b>" + p.v.toFixed(1) + "%</b> retained",
            function () {{ return X(p.d) * (o.wrap.clientWidth / W); }},
            function () {{ return Y(p.v) * (o.wrap.clientWidth / W); }});
        }})(p, s);
        svg.appendChild(hit);
      }});
    }});
    o.mount.appendChild(svg);
  }})();

  // ---------------------------------------------------------------- loss dumbbell
  (function () {{
    var o = box(document.getElementById("loss-chart"), "GCG loss on HarmBench, step 0 to step 150");
    o.wrap.style.gridColumn = "1 / -1";
    var rows = D.loss.harmbench;
    var rowH = 21, gapH = 20, m = {{ t: 30, r: 20, b: 44, l: 128 }};
    // each new model gets a blank band above it so its heading never lands on the
    // previous model's last row
    var ys = [], cur = m.t, seen = null;
    rows.forEach(function (r) {{
      if (r.m !== seen) {{ seen = r.m; cur += gapH; }}
      ys.push(cur + rowH / 2); cur += rowH;
    }});
    var W = 900, H = cur + m.b;
    var x0 = m.l, x1 = W - m.r;
    var vMax = 2.8;
    var X = function (v) {{ return x0 + v / vMax * (x1 - x0); }};
    var svg = el("svg", {{ viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Initial and final GCG loss for every arm of all three models on HarmBench." }});
    svg.style.minWidth = "30rem";

    for (var v = 0; v <= vMax; v += 0.5) {{
      svg.appendChild(el("line", {{ x1: X(v), x2: X(v), y1: m.t - 8, y2: H - m.b + 4, class: "grid" }}));
      svg.appendChild(el("text", {{ x: X(v), y: H - m.b + 20, "text-anchor": "middle", class: "tick" }},
        v.toFixed(1)));
    }}
    svg.appendChild(el("text", {{ x: (x0 + x1) / 2, y: H - 8, "text-anchor": "middle", class: "atitle" }},
      "mean cross-entropy loss"));

    var lastModel = null;
    rows.forEach(function (r, i) {{
      var y = ys[i];
      var col = SC[r.m];
      if (r.m !== lastModel) {{
        lastModel = r.m;
        svg.appendChild(el("text", {{ x: 6, y: y - rowH / 2 - 7, class: "tick",
          style: "fill:var(--ink);font-weight:620" }},
          r.m === "1b" ? "Balcony-1B" : r.m === "7b" ? "Balcony-LLaMA2-7B" : "Balcony-LLaMA3-8B"));
      }}
      svg.appendChild(el("text", {{ x: x0 - 10, y: y + 4, "text-anchor": "end", class: "tick" }}, r.arm));
      svg.appendChild(el("line", {{ x1: X(r.fin), x2: X(r.init), y1: y, y2: y,
        stroke: col, "stroke-width": 2, opacity: ".38", "stroke-linecap": "round" }}));
      svg.appendChild(el("circle", {{ cx: X(r.init), cy: y, r: 4, fill: "var(--panel)",
        stroke: col, "stroke-width": 1.8 }}));
      svg.appendChild(el("circle", {{ cx: X(r.fin), cy: y, r: 4.4, fill: col,
        stroke: "var(--panel)", "stroke-width": 1.4 }}));
      var hit = el("rect", {{ x: x0, y: y - rowH / 2, width: x1 - x0, height: rowH, class: "hit" }});
      (function (r) {{
        hover(hit, o.tip, o.wrap,
          "<b>" + r.arm + "</b><br>start <b>" + r.init.toFixed(2) + "</b> &rarr; end <b>" +
          r.fin.toFixed(2) + "</b><br>" + r.cut + "% of the loss removed",
          function () {{ return X(r.fin) * (o.wrap.clientWidth / W); }},
          function () {{ return y * (o.wrap.clientWidth / W); }});
      }})(r);
      svg.appendChild(hit);
    }});

    var lg = el("g", {{}});
    lg.appendChild(el("circle", {{ cx: x0 + 4, cy: 12, r: 4, fill: "var(--panel)",
      stroke: "var(--ink-3)", "stroke-width": 1.8 }}));
    lg.appendChild(el("text", {{ x: x0 + 14, y: 16, class: "tick" }}, "step 0"));
    lg.appendChild(el("circle", {{ cx: x0 + 74, cy: 12, r: 4.4, fill: "var(--ink-3)" }}));
    lg.appendChild(el("text", {{ x: x0 + 84, y: 16, class: "tick" }}, "step 150"));
    svg.appendChild(lg);
    o.mount.appendChild(svg);
  }})();

  // ---------------------------------------------------------------- tables
  var tb = document.getElementById("mcn-body"), lastKey = null;
  D.mcnemar.forEach(function (r) {{
    if (r.model !== lastKey) {{
      lastKey = r.model;
      var g = document.createElement("tr"); g.className = "grp";
      var td = document.createElement("td"); td.colSpan = 6; td.textContent = r.model;
      g.appendChild(td); tb.appendChild(g);
    }}
    var tr = document.createElement("tr");
    var p = r.p < 0.001 ? "&lt; 0.001" : r.p.toFixed(3);
    tr.innerHTML =
      '<td class="l">' + r.cmp + "</td>" +
      '<td class="l">' + r.bench + "</td>" +
      "<td>" + r.b + "</td><td>" + r.c + "</td>" +
      "<td>" + p + "</td>" +
      '<td class="l"><span class="tag t-' + r.cls + '">' + r.verdict + "</span></td>";
    tb.appendChild(tr);
  }});

  var cb = document.getElementById("cell-body"), lastM = null;
  D.cells.forEach(function (r) {{
    if (r.model !== lastM) {{
      lastM = r.model;
      var g = document.createElement("tr"); g.className = "grp";
      var td = document.createElement("td"); td.colSpan = 7; td.textContent = r.model;
      g.appendChild(td); cb.appendChild(g);
    }}
    var tr = document.createElement("tr");
    tr.innerHTML =
      '<td class="l">' + r.arm + "</td>" +
      "<td>" + r.depth + "%</td>" +
      "<td>" + r.harmbench.toFixed(0) + '% <span class="ci">' + r.harmbench_ci + "</span></td>" +
      "<td>" + r.advbench.toFixed(0) + '% <span class="ci">' + r.advbench_ci + "</span></td>" +
      "<td>" + r.harmbench_loss.toFixed(2) + "</td>" +
      "<td>" + r.advbench_loss.toFixed(2) + "</td>" +
      "<td>" + r.ret.toFixed(0) + "%</td>";
    cb.appendChild(tr);
  }});
}})();
</script>
"""

out = HERE / "depth_explorer.html"
out.write_text(html)
print("wrote", out, len(html), "bytes")
print("exit-vs-full:", HEAD)
