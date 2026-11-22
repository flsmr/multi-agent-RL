"""Train a shared privileged PPO teacher on Udacity Tennis."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from teacher_rollouts import TeacherRolloutPool
from teacher_ppo import TeacherPPOAgent

ROOT = Path(__file__).resolve().parent
EPISODE_FIELDS = ("episode", "steps", "return_agent0", "return_agent1", "score",
                  "rolling100", "terminated", "environment_steps", "transitions", "elapsed_seconds",
                  "environment_index", "environment_episode", "environment_seed", "time_limit")
UPDATE_FIELDS = ("iteration", "environment_steps", "transitions", "elapsed_seconds",
                 "policy_loss", "value_loss", "optimizer_steps", "latent_std_mean",
                 "collection_seconds", "update_seconds", "environment_steps_per_second")
HYPERPARAMETERS = {"learning_rate", "gamma", "clip_epsilon", "update_epochs",
                   "minibatch_size", "value_loss_coef", "gae_lambda"}


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def validate_config(path, config):
    """Reject malformed settings before claiming a fresh run or launching Unity."""
    if not isinstance(config, dict) or config.get("status") != "prepared":
        raise ValueError("Only prepared runs can start; create a new run folder.")
    fields = {"name", "status", "seed", "worker_id", "iterations", "rollout_steps",
              "hidden_size", "environment", "hyperparameters", "checkpoint_every"}
    if not fields <= set(config) or set(config) - fields - {"num_envs", "worker_timeout_seconds", "target_score"}:
        raise ValueError("Prepared config has missing or unknown fields.")
    target = config.get("target_score", 0.5)
    if type(target) not in (int, float) or not np.isfinite(target) or target <= 0:
        raise ValueError("target_score must be positive and finite.")
    count = config.get("num_envs", 1)
    if type(count) is not int or not 1 <= count <= len(os.sched_getaffinity(0)):
        raise ValueError("num_envs must fit the available CPU allocation.")
    timeout = config.get("worker_timeout_seconds", 120)
    if type(timeout) not in (int, float) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker_timeout_seconds must be positive and finite.")
    name = config["name"]
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
            or path != ROOT / "runs" / name / "config.json"):
        raise ValueError("Use runs/<name>/config.json with a matching simple run name.")
    for key, low, high in (("seed", 0, 2**31 - 1), ("worker_id", 0, 60530),
                           ("iterations", 1, 10**7), ("rollout_steps", 1, 10**7),
                           ("hidden_size", 1, 4096), ("checkpoint_every", 1, 10**7)):
        value = config[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid integer setting: {key}.")
    if config["worker_id"] + count - 1 > 60530 or config["seed"] + count - 1 > 2**31 - 1:
        raise ValueError("Environment seeds or worker IDs exceed supported ranges.")
    executable = config["environment"]
    if (not isinstance(executable, str) or not executable or Path(executable).is_absolute()
            or ".." in Path(executable).parts):
        raise ValueError("environment must be a relative executable path without '..'.")
    parameters = config["hyperparameters"]
    if not isinstance(parameters, dict) or set(parameters) != HYPERPARAMETERS:
        raise ValueError("Invalid hyperparameter fields.")
    for key in ("update_epochs", "minibatch_size"):
        if type(parameters[key]) is not int or parameters[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for key in HYPERPARAMETERS - {"update_epochs", "minibatch_size"}:
        value = parameters[key]
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        if key in ("gamma", "gae_lambda"):
            valid = 0 <= value <= 1
        elif key == "clip_epsilon":
            valid = 0 < value < 1
        else:
            valid = value > 0
        if not valid:
            raise ValueError(f"Invalid hyperparameter: {key}.")


def save_checkpoint(run_dir, agent, config, filename="checkpoint.pt"):
    """Save weights and optimizer state with portable reconstruction metadata."""
    checkpoint = {
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "state_size": 48, "action_size": 2, "hidden_size": config["hidden_size"],
        "observation_order": "self_then_other", "config": config.copy(),
    }
    temporary = run_dir / f"{filename}.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(run_dir / filename)


def train_from_config(config_path):
    """Collect independent player trajectories, then update their shared PPO model."""
    path = Path(config_path).resolve()
    run_dir = path.parent
    descriptor = os.open(run_dir, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This run is already in use.") from None
        config = json.loads(path.read_text())
        validate_config(path, config)
        if {entry.name for entry in run_dir.iterdir()} != {"config.json"}:
            raise ValueError("Prepared run must contain only config.json.")
        if not (ROOT / config["environment"]).is_file():
            raise FileNotFoundError("Download Tennis as described in the README.")

        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(1)
        agent = TeacherPPOAgent(48, 2, hidden_size=config["hidden_size"], **config["hyperparameters"])
        num_envs = config.get("num_envs", 1)
        config.update(num_envs=num_envs, worker_timeout_seconds=config.get("worker_timeout_seconds", 120),
                      target_score=config.get("target_score", 0.5), target_episode=None,
                      target_environment_steps=None, score_milestones=[],
                      worker_seeds=[config["seed"] + index for index in range(num_envs)],
                      worker_ids=[config["worker_id"] + index for index in range(num_envs)],
                      samples_per_rollout=2 * num_envs * config["rollout_steps"],
                      episode_order="rollout step, then environment index",
                      collection="frozen_worker_rollouts",
                      status="running", algorithm="ppo_gae_advnorm", policy="privileged_teacher",
                      state_size=48, action_size=2, num_agents=2, num_trajectories=2 * num_envs,
                      observation_order="self_then_other", started_at=datetime.now(timezone.utc).isoformat(),
                      episodes=0, time_limit_episodes=0, iterations_completed=0,
                      environment_steps=0, transitions=0, solved_episode=None,
                      final_rolling100=None, best_rolling100=None,
                      checkpoint=f"runs/{config['name']}/checkpoint.pt",
                      software={"torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu"})
        write_json(path, config)
        started = time.monotonic()
        env = None
        try:
            env = TeacherRolloutPool(agent.model, config["environment"], config["seed"], config["worker_id"],
                                     num_envs, config["worker_timeout_seconds"])
            config["environment_startup_seconds"] = time.monotonic() - started
            recent = deque(maxlen=100)

            with (run_dir / "episodes.csv").open("x", newline="") as episode_file, \
                    (run_dir / "updates.csv").open("x", newline="") as update_file:
                episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS, lineterminator="\n")
                update_writer = csv.DictWriter(update_file, fieldnames=UPDATE_FIELDS, lineterminator="\n")
                episode_writer.writeheader()
                update_writer.writeheader()
                episode_file.flush()
                update_file.flush()
                for iteration in range(1, config["iterations"] + 1):
                    collection_started = time.monotonic()
                    previous_steps = config["environment_steps"]
                    rollout, completed = env.collect(config["rollout_steps"])
                    config["environment_steps"] += rollout.position * num_envs
                    config["transitions"] += rollout.position * 2 * num_envs
                    for completed_episode in completed:
                        episode = dict(completed_episode)
                        event_steps = previous_steps + episode.pop("rollout_step") * num_envs
                        config["episodes"] += 1
                        config["time_limit_episodes"] += int(episode["time_limit"])
                        recent.append(episode["score"])
                        mean = float(np.mean(recent)) if len(recent) == 100 else None
                        config["final_rolling100"] = mean
                        previous_best = config["best_rolling100"]
                        if mean is not None and (previous_best is None or mean > previous_best):
                            config["best_rolling100"] = mean
                            save_checkpoint(run_dir, agent, config, "best.pt")
                        episode_writer.writerow(dict(
                            episode=config["episodes"], **episode, rolling100=mean,
                            environment_seed=config["seed"] + episode["environment_index"],
                            environment_steps=event_steps, transitions=2 * event_steps,
                            elapsed_seconds=time.monotonic() - started))
                        episode_file.flush()
                        if mean is not None and mean >= 0.5 and config["solved_episode"] is None:
                            config["solved_episode"] = config["episodes"]
                            config["solved_environment_steps"] = event_steps
                            print(f"Solved at episode {config['episodes']}: rolling100={mean:.3f}", flush=True)
                        if mean is not None and mean >= config["target_score"] and config["target_episode"] is None:
                            config["target_episode"] = config["episodes"]
                            config["target_environment_steps"] = event_steps
                            print(f"Target {config['target_score']:g} at episode {config['episodes']}: "
                                  f"rolling100={mean:.3f}", flush=True)
                        recorded = {item["threshold"] for item in config["score_milestones"]}
                        for threshold in sorted({0.5, 1.0, config["target_score"]} - recorded):
                            if mean is not None and mean >= threshold:
                                filename = f"milestone_{float(threshold)!r}.pt"
                                config["score_milestones"].append(dict(
                                    threshold=threshold, episode=config["episodes"],
                                    environment_steps=event_steps, rolling100=mean,
                                    elapsed_seconds=time.monotonic() - started,
                                    checkpoint=str((run_dir / filename).relative_to(ROOT))))
                                save_checkpoint(run_dir, agent, config, filename)

                    collection_seconds = time.monotonic() - collection_started

                    if config["target_episode"] is not None:
                        # Keep the policy which generated the qualifying scores, without another update.
                        config["stop_reason"] = "solved" if config["target_score"] == 0.5 else "target_score"
                        break
                    update_started = time.monotonic()
                    metrics = agent.update(rollout)
                    update_seconds = time.monotonic() - update_started
                    config["iterations_completed"] = iteration
                    update_writer.writerow(dict(
                        iteration=iteration, environment_steps=config["environment_steps"],
                        transitions=config["transitions"], elapsed_seconds=time.monotonic() - started,
                        collection_seconds=collection_seconds, update_seconds=update_seconds,
                        environment_steps_per_second=(rollout.position * num_envs) / collection_seconds,
                        **metrics, latent_std_mean=float(agent.model.log_std.detach().exp().mean())))
                    update_file.flush()
                    if iteration % config["checkpoint_every"] == 0:
                        save_checkpoint(run_dir, agent, config)
                    config["elapsed_seconds"] = time.monotonic() - started
                    write_json(path, config)
                    rolling = config["final_rolling100"]
                    rolling_text = "pending" if rolling is None else f"{rolling:.3f}"
                    print(f"Iteration {iteration}/{config['iterations']} | episodes={config['episodes']} | "
                          f"steps={config['environment_steps']} | rolling100={rolling_text} | "
                          f"std={float(agent.model.log_std.detach().exp().mean()):.3f} | "
                          f"policy_loss={metrics['policy_loss']:.4f}", flush=True)
                else:
                    config["stop_reason"] = "iteration_budget"
            config["status"] = "complete"
        except KeyboardInterrupt:
            config["status"] = "interrupted"
            raise
        except BaseException:
            config["status"] = "failed"
            raise
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                config["finished_at"] = datetime.now(timezone.utc).isoformat()
                config["elapsed_seconds"] = time.monotonic() - started
                save_checkpoint(run_dir, agent, config)
                write_json(path, config)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        train_from_config(args.config)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
