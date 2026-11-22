"""Rebuild the report's architecture, process and frozen-evaluation figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
BLUE = "#35658c"
GREEN = "#487967"


def box(ax, x, y, width, height, label, color=BLUE):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.015",
        linewidth=1, edgecolor=color, facecolor="#f6f7f8"))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)


def arrow(ax, start, end, color=BLUE):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4})


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.02, .94, "Teacher · privileged observations", weight="bold", color=BLUE, fontsize=12)
    box(ax, .02, .52, .22, .27, "Own 24 + other 24\n48 inputs")
    box(ax, .34, .70, .30, .17, "Actor: 128 → 128 → 2\ntanh hidden layers")
    box(ax, .34, .43, .30, .17, "Critic: 128 → 128 → 1\ntanh hidden layers")
    box(ax, .74, .70, .24, .17, "Gaussian + tanh\n2 bounded actions")
    box(ax, .74, .43, .24, .17, "State value\nPPO training only")
    arrow(ax, (.265, .67), (.315, .78))
    arrow(ax, (.265, .63), (.315, .51))
    arrow(ax, (.665, .785), (.715, .785))
    arrow(ax, (.665, .515), (.715, .515))
    ax.axhline(.35, color="#dddddd", linewidth=1)
    ax.text(.02, .28, "Student · local observations", weight="bold", color=GREEN, fontsize=12)
    box(ax, .02, .04, .22, .17, "Own 24\n24 inputs", GREEN)
    box(ax, .34, .04, .30, .17, "Actor: 128 → 128 → 2\ntanh hidden layers", GREEN)
    box(ax, .74, .04, .24, .17, "tanh output\n2 bounded actions", GREEN)
    arrow(ax, (.265, .125), (.315, .125), GREEN)
    arrow(ax, (.665, .125), (.715, .125), GREEN)
    save(fig, "teacher_student_architecture.png")


def plot_process():
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set(xlim=(-.025, 1.025), ylim=(0, 1))
    ax.axis("off")
    labels = ["1. Train teacher\nPPO self-play\nUnity + rewards",
              "2. Collect examples\nFrozen teacher\nParallel Unity workers",
              "3. Fit student\nLocal obs → action\nOffline MSE",
              "4. Evaluate\nFrozen policies\nFresh Unity matches"]
    for i, label in enumerate(labels):
        x = .01 + i * .255
        box(ax, x, .32, .19, .55, label, GREEN if i == 2 else BLUE)
        if i < 3:
            arrow(ax, (x + .21, .595), (x + .235, .595))
    ax.text(.5, .11, "Collect local observations and teacher actions; split complete matches before fitting.",
            ha="center", fontsize=10, color="#444444")
    save(fig, "teacher_student_process.png")


def plot_evaluation():
    names = ["teacher_stochastic", "teacher_deterministic", "student_deterministic"]
    main = json.loads((ROOT / "runs/experiment_summary.json").read_text()).get("main_experiment")
    fig, ax = plt.subplots(figsize=(8, 3.7))
    all_means = []
    for i, name in enumerate(names):
        directory = ROOT / main["evaluations"][name] if main else ROOT / "runs" / f"evaluation_{name}"
        summary = json.loads((directory / "summary.json").read_text())
        means = [row["mean_score"] for row in summary["per_seed"]]
        all_means.extend(means)
        x = i + np.linspace(-.075, .075, len(means))
        color = GREEN if name.startswith("student") else BLUE
        ax.scatter(x, means, color=color, s=34, zorder=3)
        ax.plot([i - .18, i + .18], [np.mean(means)] * 2, color=color, linewidth=2)
    ax.axhline(.5, color="black", linestyle="--", linewidth=1, label="Solve threshold")
    ax.set(xticks=range(3), xticklabels=["Teacher\nstochastic", "Teacher\ndeterministic", "Student\ndeterministic"],
           ylabel="100-match mean score", ylim=(0, max(2.55, max(all_means) + .15)), xlim=(-.5, 2.5))
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "policy_evaluation.png")


def plot_extended_training():
    main = json.loads((ROOT / "runs/experiment_summary.json").read_text()).get("main_experiment")
    if not main:
        return
    import csv
    directory = ROOT / main["teacher_run"]
    config = json.loads((directory / "config.json").read_text())
    rows = list(csv.DictReader((directory / "episodes.csv").open()))
    updates = list(csv.DictReader((directory / "updates.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot([int(r["environment_steps"]) for r in rows],
                 [float(r["rolling100"]) if r["rolling100"] else np.nan for r in rows], color=BLUE)
    for item in config["score_milestones"]:
        color = GREEN if item["threshold"] == .5 else "#95568b"
        axes[0].axhline(item["threshold"], color=color, linestyle="--", linewidth=.8)
        axes[0].scatter([item["environment_steps"]], [item["rolling100"]], color=color, zorder=3,
                        label=f"{item['threshold']:g} at {item['environment_steps']:,} steps")
    axes[0].set(xlabel="Total environment steps", ylabel="100-episode mean score",
                title="8 environments × 512 steps; trained to 1.0")
    axes[0].legend(fontsize=8)
    axes[1].plot([int(r["environment_steps"]) for r in updates],
                 [float(r["latent_std_mean"]) for r in updates], color=BLUE)
    axes[1].set(xlabel="Total environment steps", ylabel="Mean latent Gaussian standard deviation",
                title="Learned exploration noise")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "teacher_extended_training.png")

    rows = list(csv.DictReader((ROOT / main["student_run"] / "losses.csv").open()))
    config = json.loads((ROOT / main["student_run"] / "config.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    epochs = [int(r["epoch"]) for r in rows]
    axes[0].plot(epochs, [float(r["train_mse"]) for r in rows], label="Training")
    axes[0].plot(epochs, [float(r["validation_mse"]) for r in rows], label="Validation")
    axes[0].axvline(config["best_epoch"], color=GREEN, linestyle=":", label="Selected epoch")
    axes[0].set(xlabel="Epoch", ylabel="Action MSE", title="Offline student fitting")
    axes[0].legend()
    axes[1].plot(epochs, [float(r["movement_mse"]) for r in rows], label="Movement MSE")
    axes[1].plot(epochs, [float(r["jump_mse"]) for r in rows], label="Jump MSE")
    axes[1].set(xlabel="Epoch", ylabel="Held-out action MSE", title="Validation by action")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "student_distillation.png")


if __name__ == "__main__":
    plot_architecture()
    plot_process()
    plot_evaluation()
    plot_extended_training()
    print("Saved architecture, process, evaluation and available training figures in assets/.")
