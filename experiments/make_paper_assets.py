"""Build PNG/PDF figures and LaTeX tables from the archived result sets. Uses the same
attack and ASR aggregation as export_results.py. Requires NumPy and Matplotlib; no GPU
is needed."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_ROOT  # noqa: E402
import export_results as er  # noqa: E402

FIG_DIR = PROJECT_ROOT / "paper_figures"
TAB_DIR = PROJECT_ROOT / "paper_tables"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

MODELS = ["1b", "7b", "8b"]
MODEL_TITLE = {"1b": "Balcony-1B", "7b": "Balcony-LLaMA2-7B", "8b": "Balcony-LLaMA3-8B"}
MODEL_SHORT = {"1b": "1B", "7b": "7B", "8b": "8B"}
LAYERS = {"1b": 16, "7b": 32, "8b": 32}
# balcony exits shallow -> deep, then the full model, then the separate baseline.
EXIT_ORDER = {"1b": ["4", "8", "12", "full", "no_balcony"],
              "7b": ["15", "18", "21", "full", "no_balcony"],
              "8b": ["15", "18", "21", "full", "no_balcony"]}
RESULTS_DIR = {name: spec["dir"] for name, spec in er.MODELS.items()}
BALCONY_FILE = {"harmbench": "phase2_attacks.json", "advbench": "advbench_attacks.json"}
BASELINE_FILE = {"harmbench": "no_balcony_attacks.json",
                 "advbench": "gcg_no_balcony_advbench_attacks.json"}

UTIL_TASKS = [("hellaswag", "HellaSwag"), ("arc_challenge", "ARC-C"),
              ("arc_easy", "ARC-E"), ("winogrande", "WinoGrande"),
              ("piqa", "PIQA"), ("mmlu", "MMLU")]

# dataviz canonical categorical order (validated in the skill's palette.md)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
# slot 3 (aqua), not green: green vs orange is ΔE 3.2 under protanopia, and in
# figs 04/05/06 colour is the only thing separating the three models.
MODEL_COLOR = {"1b": C_BLUE, "7b": C_ORANGE, "8b": C_AQUA}
EXIT_COLORS = [C_BLUE, C_AQUA, C_YELLOW, C_GREEN]  # shallow -> full
BASE_COLOR = "#9a3b8f"  # baseline model: its own hue, distinct from the exits

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


# ------------------------------------------------------------------- labels/data
def fig_exit_label(k, model):
    if k == "full":
        return "full"
    if k == "no_balcony":
        return "baseline"
    return f"exit {k}"


def tex_exit_label(k, model):
    if k == "full":
        return f"Full ({LAYERS[model]})"
    if k == "no_balcony":
        return "No-Balcony"
    return f"Exit {k}"


def rel_depth(k, model):
    return 1.0 if k in ("full", "no_balcony") else int(k) / LAYERS[model]


def attack_lookup():
    """{(model, benchmark, exit): {mean_loss, asr_pct}} incl the no_balcony arm."""
    out = {}
    for r in er.collect(MODELS):
        if r["attack"] != "gcg":
            continue
        out[(r["model"], r["benchmark"], r["exit"])] = {
            "mean_loss": r["mean_loss"], "asr_pct": r["asr_pct"]}
    return out


def load_util(model):
    d = PROJECT_ROOT / RESULTS_DIR[model]
    clean = json.load(open(d / "utility_eval.json"))
    ap = d / "utility_eval_attacked.json"
    return clean, (json.load(open(ap)) if ap.exists() else {})


def mean_loss_curve(path, key):
    data = json.load(open(path))
    arr = np.array([e["loss_curve"] for e in data[key] if len(e["loss_curve"]) == 150])
    return arr.mean(0)


def _save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  paper_figures/{name}.{{pdf,png}}")


# ------------------------------------------------------------------------ figures
def fig_01_asr_bar(look):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    width = 0.38
    for ax, m in zip(axes, MODELS):
        keys = EXIT_ORDER[m]
        x = np.arange(len(keys))
        hb = [look.get((m, "harmbench", k), {}).get("asr_pct", np.nan) for k in keys]
        ab = [look.get((m, "advbench", k), {}).get("asr_pct", np.nan) for k in keys]
        b1 = ax.bar(x - width / 2, hb, width, color=C_BLUE, label="HarmBench")
        b2 = ax.bar(x + width / 2, ab, width, color=C_ORANGE, label="AdvBench")
        for bars, vals in ((b1, hb), (b2, ab)):
            for rect, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.0f}",
                            ha="center", va="bottom", fontsize=8, color=INK)
        ax.axvline(len(keys) - 1.5, color=GRID, lw=1.2, zorder=0)
        ax.set_title(MODEL_TITLE[m], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([fig_exit_label(k, m) for k in keys], fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("GCG attack success rate (%)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    _save(fig, "01_asr_bar")


def _loss_fig(bench, name):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharex=True)
    for ax, m in zip(axes, MODELS):
        rdir = PROJECT_ROOT / RESULTS_DIR[m]
        bal = [k for k in EXIT_ORDER[m] if k not in ("no_balcony",)]
        for i, k in enumerate(bal):
            ax.plot(np.arange(150), mean_loss_curve(rdir / BALCONY_FILE[bench], k),
                    color=EXIT_COLORS[i], lw=2, label=fig_exit_label(k, m))
        ax.plot(np.arange(150), mean_loss_curve(rdir / BASELINE_FILE[bench], "no_balcony"),
                color=BASE_COLOR, lw=1.8, ls="--", label="baseline")
        ax.set_title(MODEL_TITLE[m], fontsize=11)
        ax.set_xlabel("GCG step")
        ax.set_xlim(0, 149)
        ax.set_ylim(0, None)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0].set_ylabel("mean cross-entropy loss")
    _save(fig, name)


def fig_04_utility(clean_att):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (task, label) in zip(axes.flat, UTIL_TASKS):
        for m in MODELS:
            clean, att = clean_att[m]
            bal = [k for k in EXIT_ORDER[m] if k != "no_balcony"]
            xs = [rel_depth(k, m) for k in bal]
            yc = [clean[k][task]["value"] for k in bal]
            ya = [att[k][task]["value"] if task in att.get(k, {}) else np.nan for k in bal]
            ax.plot(xs, yc, "-o", color=MODEL_COLOR[m], lw=1.9, ms=5.5)
            ax.plot(xs, ya, "--s", color=MODEL_COLOR[m], lw=1.4, ms=4.5,
                    mfc="white", alpha=0.85)
            # no-balcony baseline as diamonds at depth 1.0 (clean filled, attacked open)
            bc = clean["no_balcony"][task]["value"]
            ba = att.get("no_balcony", {}).get(task, {}).get("value", np.nan)
            ax.plot(1.0, bc, "D", color=MODEL_COLOR[m], ms=6, mec=INK, mew=0.5)
            if not np.isnan(ba):
                ax.plot(1.0, ba, "D", color="white", ms=6, mec=MODEL_COLOR[m], mew=1.2)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 0.85)
        ax.set_xlim(0.18, 1.08)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.axhline(0.25, color=GRID, lw=1.0, ls=":", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("relative exit depth (exit / #layers; 1.0 = full)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    handles = ([Line2D([0], [0], color=MODEL_COLOR[m], lw=2.2, label=MODEL_TITLE[m])
                for m in MODELS]
               + [Line2D([0], [0], color=INK2, lw=1.9, marker="o", label="clean"),
                  Line2D([0], [0], color=INK2, lw=1.5, ls="--", marker="s",
                         mfc="white", label="with GCG suffix"),
                  Line2D([0], [0], color=INK2, ls="", marker="D", mfc=INK2,
                         label="no-balcony baseline")])
    fig.legend(handles=handles, frameon=False, ncol=6, fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    _save(fig, "04_utility_clean_vs_attacked")


def fig_05_loss_vs_asr(look):
    fig, ax = plt.subplots(figsize=(7, 5))
    # markers by exit rank so points stay identifiable without crowded text labels
    exit_marker = ["o", "^", "s", "P", "D"]  # shallow, mid, deep, full, baseline
    for m in MODELS:
        for i, k in enumerate(EXIT_ORDER[m]):
            s = look.get((m, "harmbench", k), {})
            if s.get("mean_loss") is not None and s.get("asr_pct") is not None:
                ax.scatter(s["mean_loss"], s["asr_pct"], color=MODEL_COLOR[m],
                           marker=exit_marker[i], s=70, zorder=3,
                           edgecolor="white", linewidth=0.6)
    model_handles = [Line2D([0], [0], color=MODEL_COLOR[m], marker="o", ls="",
                            ms=8, label=MODEL_TITLE[m]) for m in MODELS]
    shape_handles = [Line2D([0], [0], color=INK2, marker=exit_marker[i], ls="", ms=8,
                            label=lab) for i, lab in
                     enumerate(["shallowest exit", "mid exit", "deep exit",
                                "full", "baseline"])]
    leg1 = ax.legend(handles=model_handles, frameon=False, fontsize=9,
                     loc="upper right")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=shape_handles, frameon=True, fontsize=8.5,
                     loc="center right", title="marker = exit depth",
                     title_fontsize=8.5)
    leg2.get_frame().set_edgecolor(GRID)
    leg2.get_frame().set_facecolor("white")
    ax.set_xlabel("final GCG loss (HarmBench)")
    ax.set_ylabel("attack success rate (%)")
    ax.set_ylim(-3, 100)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "05_loss_vs_asr")


def fig_06_harmbench_vs_advbench(look):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    lim = 0
    for m in MODELS:
        xs, ys = [], []
        for k in EXIT_ORDER[m]:
            hb = look.get((m, "harmbench", k), {}).get("mean_loss")
            ab = look.get((m, "advbench", k), {}).get("mean_loss")
            if hb is not None and ab is not None:
                xs.append(hb); ys.append(ab); lim = max(lim, hb, ab)
        ax.scatter(xs, ys, color=MODEL_COLOR[m], s=55, label=MODEL_TITLE[m], zorder=3)
    lim = lim * 1.08
    ax.plot([0, lim], [0, lim], ls="--", color=INK2, lw=1, zorder=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("final GCG loss, HarmBench")
    ax.set_ylabel("final GCG loss, AdvBench")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, "06_harmbench_vs_advbench")


# ------------------------------------------------------------------------- tables
def _fmt(v, nd=3):
    return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _delta(att, clean):
    if att is None or clean is None or np.isnan(att) or np.isnan(clean):
        return ""
    d = (att - clean) * 100
    sign = "+" if d >= 0 else "$-$"
    return f" \\scriptsize{{({sign}{abs(d):.1f})}}"


def _wrap_table(body_rows, colspec, header, caption, label):
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\small", r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{" + colspec + "}", r"\toprule",
        header + r" \\", r"\midrule", *body_rows,
        r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def table_utility_clean(clean_att):
    header = (r"\textbf{Model} & \textbf{Exit} & HellaSwag & ARC-C & ARC-E & "
              r"WinoGrande & PIQA & MMLU")
    rows = []
    for mi, m in enumerate(MODELS):
        clean, _ = clean_att[m]
        keys = EXIT_ORDER[m]
        for j, k in enumerate(keys):
            lead = (r"\multirow{%d}{*}{%s}" % (len(keys), MODEL_SHORT[m])) if j == 0 else ""
            cells = " & ".join(_fmt(clean[k][t]["value"]) for t, _ in UTIL_TASKS)
            rows.append(f"{lead} & {tex_exit_label(k, m)} & {cells} \\\\")
        if mi < len(MODELS) - 1:
            rows.append(r"\midrule")
    cap = ("Utility on six multiple choice benchmarks with no attack, for every exit "
           "of each model together with the full model and the separate no-balcony "
           "baseline. Scores are accuracy from lm-eval-harness (acc\\_norm for "
           "HellaSwag, ARC and PIQA, acc for WinoGrande and MMLU). The full exit and "
           "the no-balcony baseline share trunk weights, so their scores match.")
    tex = _wrap_table(rows, "ll cccccc", header, cap, "tab:utility_clean")
    (TAB_DIR / "utility_clean.tex").write_text(tex + "\n")
    return tex


def table_utility_attacked(clean_att):
    header = (r"\textbf{Model} & \textbf{Exit} & HellaSwag & ARC-C & ARC-E & "
              r"WinoGrande & PIQA & MMLU")
    rows = []
    for mi, m in enumerate(MODELS):
        clean, att = clean_att[m]
        keys = EXIT_ORDER[m]
        for j, k in enumerate(keys):
            lead = (r"\multirow{%d}{*}{%s}" % (len(keys), MODEL_SHORT[m])) if j == 0 else ""
            cells = []
            for t, _ in UTIL_TASKS:
                a = att.get(k, {}).get(t, {}).get("value", np.nan)
                c = clean[k][t]["value"]
                cells.append(_fmt(a) + _delta(a, c))
            rows.append(f"{lead} & {tex_exit_label(k, m)} & " + " & ".join(cells) + r" \\")
        if mi < len(MODELS) - 1:
            rows.append(r"\midrule")
    cap = ("Utility with each exit's strongest GCG suffix appended to every question. "
           "Numbers in parentheses are the change from the clean scores in "
           "Table~\\ref{tab:utility_clean}, in percentage points. Most cells move by "
           "under 3 points. The exception is ARC-E on the 7B, which drops between 13 "
           "and 19 points at every arm, including the full model and the no-balcony "
           "baseline.")
    tex = _wrap_table(rows, "ll cccccc", header, cap, "tab:utility_attacked")
    (TAB_DIR / "utility_attacked.tex").write_text(tex + "\n")
    return tex


def table_attack(look):
    header = (r"\textbf{Model} & \textbf{Exit} & HB Loss & HB ASR\% & "
              r"AB Loss & AB ASR\%")
    rows = []
    for mi, m in enumerate(MODELS):
        keys = EXIT_ORDER[m]
        for j, k in enumerate(keys):
            lead = (r"\multirow{%d}{*}{%s}" % (len(keys), MODEL_SHORT[m])) if j == 0 else ""
            hb = look.get((m, "harmbench", k), {})
            ab = look.get((m, "advbench", k), {})
            rows.append(
                f"{lead} & {tex_exit_label(k, m)} & "
                f"{_fmt(hb.get('mean_loss'))} & {_fmt(hb.get('asr_pct'), 1)} & "
                f"{_fmt(ab.get('mean_loss'))} & {_fmt(ab.get('asr_pct'), 1)} \\\\")
        if mi < len(MODELS) - 1:
            rows.append(r"\midrule")
    cap = ("GCG attack results on both benchmarks. Final loss is the mean cross "
           "entropy GCG minimizes over 50 behaviors after 150 steps. ASR is the share "
           "of those behaviors the HarmBench judge scores above 0.5. Each model is "
           "listed for every exit, the full model, and its no-balcony baseline.")
    tex = _wrap_table(rows, "ll cccc", header, cap, "tab:attack_results")
    (TAB_DIR / "attack_results.tex").write_text(tex + "\n")
    return tex


# --------------------------------------------------------------- combined snippet
# banner comments mirror the layout of the template this replaced, so the blocks
# stay findable by eye in a long .tex file
FIG_BLOCK = r"""% ============================================================
% FIGURE {num}: {title}
% ============================================================

\begin{{figure}}[t]
\centering
\includegraphics[width={width}]{{paper_figures/{file}.pdf}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""

FIGURES = [
    ("01_asr_bar", "ASR Bar Chart", "0.85\\columnwidth", "fig:asr_bar",
     "Attack success rate for every exit of the three models on HarmBench and "
     "AdvBench, using GCG with 150 steps on 50 behaviors. ASR is the share of "
     "behaviors the HarmBench judge scores above 0.5. The rightmost pair in each "
     "panel is that model's no-balcony baseline."),
    ("02_loss_curves_harmbench", "GCG Loss Curves (HarmBench)", "0.85\\columnwidth",
     "fig:loss_curves",
     "Mean GCG loss across 150 optimization steps on HarmBench, averaged over 50 "
     "behaviors, with one line per exit and a separate line for the no-balcony "
     "baseline. Most of the drop lands in the first 30 steps and the curves are "
     "close to flat by the end."),
    ("03_loss_curves_advbench", "GCG Loss Curves (AdvBench)", "0.85\\columnwidth",
     "fig:advbench_loss_curves",
     "The same GCG loss curves on AdvBench. The exits keep the same ordering as on "
     "HarmBench, with deeper exits reaching lower final loss."),
    ("04_utility_clean_vs_attacked", "Utility, Clean vs Attacked", "\\columnwidth",
     "fig:utility_comparison",
     "Utility on the six benchmarks against relative exit depth, clean (solid) and "
     "with the strongest GCG suffix attached (dashed). Each color is one model. The "
     "no-balcony baseline is drawn as a diamond at depth 1.0, and on the clean runs "
     "it lands on the full model because the two share weights."),
    ("05_loss_vs_asr", "Final Loss vs ASR", "0.75\\columnwidth", "fig:loss_vs_asr",
     "Final GCG loss against attack success rate on HarmBench, one point per exit "
     "and per baseline. Marker shape gives the exit depth and color gives the model. "
     "The arms with the lowest final loss are not the arms with the highest ASR."),
    ("06_harmbench_vs_advbench", "HarmBench vs AdvBench Loss", "0.7\\columnwidth",
     "fig:harmbench_vs_advbench",
     "Final GCG loss on HarmBench against AdvBench for each exit and baseline. The "
     "dashed line is y equals x. Points below it reached a lower loss on AdvBench."),
]


TAB_BANNER = """% ============================================================
% TABLE {num}: {title}
% ============================================================
"""


def write_snippet(tables):
    """tables is [(title, tex), ...] in the order they should appear."""
    parts = [r"% Auto-generated by experiments/make_paper_assets.py",
             r"% Requires: booktabs, multirow, graphicx (resizebox).",
             r"% Figures first, then the three tables.",
             r"% There is no figure 7 onward. The template's 07 pareto, 08 p_harmful",
             r"% strip, 09 loss_reduction and 10 combined have no counterpart here:",
             r"% 07 and 10 rest on framing the significance tests do not support, 08",
             r"% needs mean p_harmful which was dropped, and 09 needs the filler-suffix",
             r"% control that only ever ran for the 1B.", ""]
    for i, (file, title, width, label, cap) in enumerate(FIGURES, 1):
        parts.append(FIG_BLOCK.format(num=i, title=title, width=width, file=file,
                                      label=label, caption=cap))
    for i, (title, tex) in enumerate(tables, 1):
        parts.append(TAB_BANNER.format(num=i, title=title))
        parts.append(tex + "\n")
    (TAB_DIR / "paper_snippet.tex").write_text("\n".join(parts))
    print("  paper_tables/paper_snippet.tex  (all figure + table blocks)")


def main():
    look = attack_lookup()
    clean_att = {m: load_util(m) for m in MODELS}
    print("figures:")
    fig_01_asr_bar(look)
    _loss_fig("harmbench", "02_loss_curves_harmbench")
    _loss_fig("advbench", "03_loss_curves_advbench")
    fig_04_utility(clean_att)
    fig_05_loss_vs_asr(look)
    fig_06_harmbench_vs_advbench(look)
    print("tables:")
    t1 = table_utility_clean(clean_att)
    t2 = table_utility_attacked(clean_att)
    t3 = table_attack(look)
    print("  paper_tables/{utility_clean,utility_attacked,attack_results}.tex")
    # same order as the template this replaced: clean utility, attacked utility, attacks
    write_snippet([("Utility Benchmarks, No Attack", t1),
                   ("Utility Under Attack", t2),
                   ("Main Attack Results", t3)])
    print("done.")


if __name__ == "__main__":
    main()
