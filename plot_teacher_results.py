"""Plot teacher optimization attempts and per-episode rewards for the report."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def plot_teacher_results(run_dirs, output):
    """Compare retained trials and show raw rewards for the main teacher."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    successful = None
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text())
        if config.get("target_score") == 1 and config["status"] == "interrupted":
            continue  # Infrastructure retry is not a distinct hyperparameter trial.
        with (run_dir / "episodes.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        steps = [int(row["environment_steps"]) for row in rows]
        means = [float(row["rolling100"]) if row["rolling100"] else np.nan for row in rows]
        label = (f"{config.get('num_envs', 1)} env × {config['rollout_steps']} steps, seed {config['seed']}"
                 + (", target 1" if config.get("target_score") == 1 else ""))
        axes[0].plot(steps, means, label=label, linewidth=1)
        if config.get("solved_episode") is not None and (
                successful is None or config.get("target_score", .5) >= successful[0].get("target_score", .5)):
            successful = (config, rows, means)
    for ax in axes:
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Solve threshold")
        ax.grid(alpha=0.2)
    axes[0].set(xlabel="Environment steps", ylabel="100-episode mean score", title="Teacher configurations")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if successful is not None:
        config, rows, means = successful
        episodes = [int(row["episode"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        axes[1].scatter(episodes, scores, s=2, color="gray", alpha=0.2, label="Episode score")
        axes[1].plot(episodes, means, linewidth=1.2, label="100-episode mean")
        axes[1].axvline(config["solved_episode"], color="green", linestyle=":", linewidth=1,
                       label=f"Solved: episode {config['solved_episode']:,}")
        if config.get("target_episode") and config.get("target_score", .5) != .5:
            axes[1].axhline(config["target_score"], color="purple", linestyle="--", linewidth=1,
                           label=f"Target {config['target_score']:g}")
        axes[1].set(xlabel="Episode", ylabel="Maximum player return", title="Successful teacher training")
        axes[1].legend(fontsize=7, loc="upper left")
    else:
        axes[1].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "teacher_training.png")
    args = parser.parse_args()
    runs = args.run or sorted(path.parent for path in (ROOT / "runs").glob("teacher*/episodes.csv"))
    if not runs:
        parser.error("No teacher episode CSVs found.")
    plot_teacher_results(runs, args.output)
    print(f"Saved {args.output.name}")
