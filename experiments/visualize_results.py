# builds all 7 figures for RESULTS.md from the json result files.
# no gpu needed, no model loading, just reads json and makes pngs.
# safe to rerun whenever, it just overwrites the old figures.
#
# i use a fixed color per exit across every figure so you can always
# tell which line/bar is which exit without checking the legend.
# the no-balcony baseline gets its own color (red) in figures where
# it makes sense to include it.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = Path("/path/to/project/results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

EXITS = [4, 8, 12, "full"]
# includes no-balcony baseline for figures that show it
ALL_MODELS = ["no_balcony", 4, 8, 12, "full"]
EXIT_LABELS = {4: "exit 4", 8: "exit 8", 12: "exit 12", "full": "full", "no_balcony": "baseline\n(no balcony)"}

# consistent colors across all figures
EXIT_COLOR = {
    4: "#2a78d6",              # blue
    8: "#1baf7a",              # aqua
    12: "#eda100",             # yellow
    "full": "#008300",         # green
    "no_balcony": "#d9434f",   # red
}

# for heatmaps where color means magnitude not identity
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2dc"


def load_json(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def fig_baseline_vs_attacked(attacks, baseline, nb_attacks, nb_baseline):
    """grouped bar chart showing no-suffix, filler, and attacked loss side by side.
    now includes the no-balcony baseline model as a 5th group."""
    conditions = ["no_suffix_loss", "filler_loss", "attacked_final_loss"]
    cond_labels = ["No suffix", "Unoptimized filler", "GCG-attacked"]
    cond_colors = ["#2a78d6", "#86b6ef", "#eda100"]

    models = ALL_MODELS
    means = {c: [] for c in conditions}

    # no-balcony baseline first
    nb_b = nb_baseline["no_balcony"]
    nb_a = nb_attacks["no_balcony"]
    means["no_suffix_loss"].append(np.mean([x["no_suffix_loss"] for x in nb_b]))
    means["filler_loss"].append(np.mean([x["filler_loss"] for x in nb_b]))
    means["attacked_final_loss"].append(np.mean([x["final_loss"] for x in nb_a]))

    # then each exit
    for exit_layer in EXITS:
        b = baseline[str(exit_layer)]
        a = attacks[str(exit_layer)]
        means["no_suffix_loss"].append(np.mean([x["no_suffix_loss"] for x in b]))
        means["filler_loss"].append(np.mean([x["filler_loss"] for x in b]))
        means["attacked_final_loss"].append(np.mean([x["final_loss"] for x in a]))

    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, cond in enumerate(conditions):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, means[cond], width,
            label=cond_labels[i], color=cond_colors[i], zorder=3,
            edgecolor="white", linewidth=0.5,
        )
        for rect, val in zip(bars, means[cond]):
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.04,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([EXIT_LABELS[e] for e in models], color=TEXT_PRIMARY)
    ax.set_ylabel("Mean loss on target string (lower = closer to compliance)", color=TEXT_SECONDARY)
    ax.set_title(
        "Attack effect size: baseline vs. GCG-attacked loss",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=14,
    )
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_baseline_vs_attacked.png", dpi=180, facecolor="white")
    plt.close(fig)


def fig_transfer_heatmap(transfer):
    """5x5 heatmap of mean transfer loss (includes no-balcony baseline if data exists).
    rows = suffix source, cols = eval target."""
    has_baseline = "no_balcony->no_balcony" in transfer
    models = ALL_MODELS if has_baseline else EXITS
    grid = np.zeros((len(models), len(models)))
    for i, src in enumerate(models):
        for j, tgt in enumerate(models):
            entries = transfer[f"{src}->{tgt}"]
            grid[i, j] = np.mean([e["loss"] for e in entries])

    fig, ax = plt.subplots(figsize=(7.2, 6.2) if has_baseline else (6.5, 5.8))
    im = ax.imshow(grid, cmap=SEQ_BLUE, aspect="equal")
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([EXIT_LABELS[e] for e in models], color=TEXT_PRIMARY)
    ax.set_yticklabels([EXIT_LABELS[e] for e in models], color=TEXT_PRIMARY)
    ax.set_xlabel("Evaluated on exit", color=TEXT_SECONDARY)
    ax.set_ylabel("Suffix optimized for exit", color=TEXT_SECONDARY)
    ax.set_title(
        "Transfer loss: rows = suffix source, cols = eval target\n(darker = higher loss = weaker attack)",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
    )
    vmin, vmax = grid.min(), grid.max()
    for i in range(len(models)):
        for j in range(len(models)):
            val = grid[i, j]
            frac = (val - vmin) / (vmax - vmin + 1e-9)
            txt_color = "white" if frac > 0.6 else TEXT_PRIMARY
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=txt_color,
                     fontsize=10.5, fontweight=weight)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    cbar.set_label("Mean transfer loss", color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_transfer_heatmap.png", dpi=180, facecolor="white")
    plt.close(fig)


def fig_per_behavior_heatmap(attacks):
    """exits x behaviors heatmap so you can see which behaviors buck the trend."""
    behaviors = attacks[str(EXITS[0])]
    n_behaviors = len(behaviors)
    behavior_labels = [
        (b["user_content"][:40] + "...") if len(b["user_content"]) > 40 else b["user_content"]
        for b in behaviors
    ]

    grid = np.zeros((len(EXITS), n_behaviors))
    for i, exit_layer in enumerate(EXITS):
        for j, b in enumerate(attacks[str(exit_layer)]):
            grid[i, j] = b["final_loss"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(grid, cmap=SEQ_BLUE, aspect="auto")
    ax.set_yticks(range(len(EXITS)))
    ax.set_yticklabels([EXIT_LABELS[e] for e in EXITS], color=TEXT_PRIMARY)
    ax.set_xticks(range(n_behaviors))
    ax.set_xticklabels([f"b{j}" for j in range(n_behaviors)], color=TEXT_PRIMARY)
    ax.set_xlabel("Behavior (see legend below)", color=TEXT_SECONDARY)
    ax.set_title(
        "Final attacked loss per exit x behavior\n(darker = higher loss = attack failed to fully break this cell)",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
    )
    vmin, vmax = grid.min(), grid.max()
    for i in range(len(EXITS)):
        for j in range(n_behaviors):
            val = grid[i, j]
            frac = (val - vmin) / (vmax - vmin + 1e-9)
            txt_color = "white" if frac > 0.6 else TEXT_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    cbar.set_label("Final loss", color=TEXT_SECONDARY)

    legend_text = "\n".join(f"b{j}: {behavior_labels[j]}" for j in range(n_behaviors))
    fig.text(0.02, -0.02, legend_text, fontsize=8, color=TEXT_SECONDARY, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_behavior_heatmap.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_loss_curves(attacks, nb_attacks):
    """loss vs GCG step, one line per exit with min-max band across behaviors.
    includes the no-balcony baseline as its own line."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # no-balcony baseline first
    nb_behaviors = nb_attacks["no_balcony"]
    nb_curves = np.array([b["loss_curve"] for b in nb_behaviors])
    nb_mean = nb_curves.mean(axis=0)
    nb_min = nb_curves.min(axis=0)
    nb_max = nb_curves.max(axis=0)
    steps = np.arange(len(nb_mean))
    color = EXIT_COLOR["no_balcony"]
    ax.plot(steps, nb_mean, color=color, linewidth=2, label="baseline (no balcony)", zorder=3)
    ax.fill_between(steps, nb_min, nb_max, color=color, alpha=0.12, zorder=2, linewidth=0)
    ax.text(steps[-1] + 1, nb_mean[-1], "baseline", color=color, fontsize=9.5, va="center", fontweight="bold")

    for exit_layer in EXITS:
        behaviors = attacks[str(exit_layer)]
        curves = np.array([b["loss_curve"] for b in behaviors])
        mean_curve = curves.mean(axis=0)
        min_curve = curves.min(axis=0)
        max_curve = curves.max(axis=0)
        steps = np.arange(len(mean_curve))
        color = EXIT_COLOR[exit_layer]
        ax.plot(steps, mean_curve, color=color, linewidth=2, label=EXIT_LABELS[exit_layer], zorder=3)
        ax.fill_between(steps, min_curve, max_curve, color=color, alpha=0.12, zorder=2, linewidth=0)
        ax.text(
            steps[-1] + 1, mean_curve[-1], EXIT_LABELS[exit_layer],
            color=color, fontsize=9.5, va="center", fontweight="bold",
        )

    ax.set_xlabel("GCG step", color=TEXT_SECONDARY)
    ax.set_ylabel("Loss on target string", color=TEXT_SECONDARY)
    ax.set_title(
        "Mean loss vs. GCG optimization step\n(shaded band = min-max across 6 behaviors)",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlim(0, 112)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_loss_curves.png", dpi=180, facecolor="white")
    plt.close(fig)


def fig_variance_dotplot(attacks, nb_attacks):
    """dots for each behavior's final loss, plus a mean bar, per exit.
    includes the no-balcony baseline."""
    models = ALL_MODELS
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rng = np.random.default_rng(0)

    for i, model_key in enumerate(models):
        if model_key == "no_balcony":
            behaviors = nb_attacks["no_balcony"]
        else:
            behaviors = attacks[str(model_key)]
        losses = [b["final_loss"] for b in behaviors]
        jitter = rng.uniform(-0.12, 0.12, size=len(losses))
        color = EXIT_COLOR[model_key]
        ax.scatter(
            np.full(len(losses), i) + jitter, losses,
            color=color, s=70, zorder=3, edgecolor="white", linewidth=0.8,
        )
        mean_val = np.mean(losses)
        ax.hlines(mean_val, i - 0.28, i + 0.28, color=TEXT_PRIMARY, linewidth=2, zorder=4)
        ax.text(i + 0.32, mean_val, f"mean {mean_val:.2f}", va="center", fontsize=8.5, color=TEXT_PRIMARY)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([EXIT_LABELS[e] for e in models], color=TEXT_PRIMARY)
    ax.set_ylabel("Final attacked loss (one dot per behavior)", color=TEXT_SECONDARY)
    ax.set_title(
        "Spread of final loss across the 6 behaviors",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlim(-0.5, len(models) - 0.2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_variance_dotplot.png", dpi=180, facecolor="white")
    plt.close(fig)


def fig_judge_scores_dotplot(judge_scores):
    """same as variance dotplot but for p_harmful instead of loss.
    includes the no-balcony baseline if its scores exist."""
    # figure out which models have judge scores
    models = []
    if "no_balcony" in judge_scores["own_exit"]:
        models.append("no_balcony")
    models.extend(EXITS)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rng = np.random.default_rng(0)
    for i, model_key in enumerate(models):
        entries = judge_scores["own_exit"][str(model_key)]
        scores = [e["p_harmful"] for e in entries]
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        color = EXIT_COLOR[model_key]
        ax.scatter(
            np.full(len(scores), i) + jitter, scores,
            color=color, s=70, zorder=3, edgecolor="white", linewidth=0.8,
        )
        mean_val = np.mean(scores)
        ax.hlines(mean_val, i - 0.28, i + 0.28, color=TEXT_PRIMARY, linewidth=2, zorder=4)
        ax.text(i + 0.32, mean_val, f"mean {mean_val:.2f}", va="center", fontsize=8.5, color=TEXT_PRIMARY)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([EXIT_LABELS[e] for e in models], color=TEXT_PRIMARY)
    ax.set_ylabel("HarmBench judge p_harmful (one dot per behavior)", color=TEXT_SECONDARY)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Real judge score: was the free-generation completion\nactually an instance of the harmful behavior?",
        color=TEXT_PRIMARY, fontsize=12.5, fontweight="bold", pad=14,
    )
    ax.set_xlim(-0.5, len(models) - 0.2)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_judge_scores_dotplot.png", dpi=180, facecolor="white")
    plt.close(fig)


def fig_judge_transfer_heatmap(judge_scores):
    """same layout as the loss transfer heatmap but showing p_harmful.
    includes no-balcony baseline if data exists."""
    has_baseline = "no_balcony->no_balcony" in judge_scores.get("transfer", {})
    models = ALL_MODELS if has_baseline else EXITS
    grid = np.zeros((len(models), len(models)))
    for i, src in enumerate(models):
        for j, tgt in enumerate(models):
            entries = judge_scores["transfer"][f"{src}->{tgt}"]
            grid[i, j] = np.mean([e["p_harmful"] for e in entries])

    fig, ax = plt.subplots(figsize=(7.2, 6.2) if has_baseline else (6.5, 5.8))
    im = ax.imshow(grid, cmap=SEQ_BLUE, aspect="equal", vmin=0, vmax=1)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([EXIT_LABELS[e] for e in models], color=TEXT_PRIMARY)
    ax.set_yticklabels([EXIT_LABELS[e] for e in models], color=TEXT_PRIMARY)
    ax.set_xlabel("Evaluated on exit", color=TEXT_SECONDARY)
    ax.set_ylabel("Suffix optimized for exit", color=TEXT_SECONDARY)
    ax.set_title(
        "Real judge transfer matrix: rows = suffix source, cols = eval target\n(darker = higher p_harmful = more successful jailbreak)",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
    )
    for i in range(len(models)):
        for j in range(len(models)):
            val = grid[i, j]
            txt_color = "white" if val > 0.6 else TEXT_PRIMARY
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color,
                     fontsize=10.5, fontweight=weight)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    cbar.set_label("Mean p_harmful", color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_judge_transfer_heatmap.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    attacks = load_json("phase2_attacks.json")
    transfer = load_json("phase2_transfer.json")
    baseline = load_json("baseline_control.json")
    nb_attacks = load_json("no_balcony_attacks.json")
    nb_baseline = load_json("no_balcony_baseline.json")

    fig_baseline_vs_attacked(attacks, baseline, nb_attacks, nb_baseline)
    fig_transfer_heatmap(transfer)
    fig_per_behavior_heatmap(attacks)
    fig_loss_curves(attacks, nb_attacks)
    fig_variance_dotplot(attacks, nb_attacks)

    try:
        judge_scores = load_json("judge_scores.json")
        fig_judge_scores_dotplot(judge_scores)
        fig_judge_transfer_heatmap(judge_scores)
    except FileNotFoundError:
        print("No judge_scores.json found, skipping judge figures")

    print(f"Saved figures to {FIG_DIR}")

    print("\n=== Attack effect size (mean filler baseline - mean attacked final loss) ===")
    print(f"{'exit':>6} | {'filler baseline':>16} | {'attacked final':>14} | {'effect (drop)':>14} | {'relative drop':>14}")
    for exit_layer in EXITS:
        b = baseline[str(exit_layer)]
        a = attacks[str(exit_layer)]
        mean_filler = np.mean([x["filler_loss"] for x in b])
        mean_final = np.mean([x["final_loss"] for x in a])
        effect = mean_filler - mean_final
        rel = effect / mean_filler
        print(f"{str(exit_layer):>6} | {mean_filler:16.4f} | {mean_final:14.4f} | {effect:14.4f} | {rel:14.1%}")


if __name__ == "__main__":
    main()
