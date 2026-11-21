"""Explore Tennis with independent uniform random actions for both players."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re

import numpy as np

from environment import make_env

ROOT = Path(__file__).resolve().parent
FIELDS = ("episode", "steps", "return_agent0", "return_agent1", "score",
          "mean_100", "terminated")


def write_config(path, config):
    """Replace metadata atomically so interrupted writes cannot corrupt JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path)


def validate_config(path, config):
    """Accept only a fresh config stored in its named run directory."""
    if config.get("status") != "prepared":
        raise ValueError("Only prepared runs can be started; create a new run folder.")
    required = {"name", "status", "policy", "seed", "worker_id", "episodes",
                "max_steps", "graphics", "environment"}
    if set(config) != required:
        raise ValueError("Config has missing or unknown fields.")
    name = config["name"]
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
            or path != ROOT / "runs" / name / "config.json"):
        raise ValueError("Use runs/<name>/config.json with a matching simple run name.")
    if config["policy"] != "random" or type(config["graphics"]) is not bool:
        raise ValueError("Policy must be random and graphics must be a JSON boolean.")
    for key, low, high in (("seed", 0, 2**31 - 1), ("worker_id", 0, 65535 - 5005),
                           ("episodes", 1, 10**7), ("max_steps", 1, 10**7)):
        value = config[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}.")
    value = config["environment"]
    if not isinstance(value, str) or not value:
        raise ValueError("environment must be a relative executable path.")
    executable = Path(value)
    if executable.is_absolute() or ".." in executable.parts:
        raise ValueError("environment must be relative to the repository, without '..'.")


def run_config(config_path):
    """Claim one prepared run, record episode returns, and always close Unity."""
    path = Path(config_path).resolve()
    # Lock the directory: atomic config replacement must not invalidate the lock.
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This run is already in use.") from None
        config = json.loads(path.read_text())
        validate_config(path, config)
        if {entry.name for entry in path.parent.iterdir()} != {"config.json"}:
            raise ValueError("Prepared run must contain only config.json; refusing to overwrite data.")

        config.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
        write_config(path, config)
        env = None
        try:
            env = make_env(config["environment"], seed=config["seed"],
                           worker_id=config["worker_id"], graphics=config["graphics"])
            brain = env.brain_names[0]
            action_size = env.brains[brain].vector_action_space_size
            rng = np.random.default_rng(config["seed"])
            recent = deque(maxlen=100)
            config.update(episodes_completed=0, episodes_truncated=0, solved_episode=None)
            with (path.parent / "episodes.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                stream.flush()
                for episode in range(1, config["episodes"] + 1):
                    info = env.reset(train_mode=not config["graphics"])[brain]
                    shape = np.asarray(info.vector_observations).shape
                    if shape != (2, 24) or action_size != 2:
                        raise RuntimeError("Expected Tennis: two players, 24 stacked observations, two actions.")
                    if episode == 1:
                        config["environment_details"] = dict(
                            agents=2, observation_size=24, action_size=2
                        )
                        write_config(path, config)
                        print("Tennis: 2 players, 24 observations/player, 2 actions/player", flush=True)
                    returns = np.zeros(2, dtype=np.float64)
                    terminated = False
                    for steps in range(1, config["max_steps"] + 1):
                        actions = rng.uniform(-1.0, 1.0, size=(2, action_size))
                        info = env.step(actions)[brain]
                        returns += info.rewards
                        if np.any(info.local_done):
                            terminated = True
                            break
                    score = float(returns.max())
                    if terminated:
                        recent.append(score)
                    else:
                        recent.clear()
                        config["episodes_truncated"] += 1
                    mean = float(np.mean(recent)) if len(recent) == 100 else None
                    if mean is not None and mean >= 0.5 and config["solved_episode"] is None:
                        config["solved_episode"] = episode
                    writer.writerow(dict(episode=episode, steps=steps,
                                         return_agent0=float(returns[0]), return_agent1=float(returns[1]),
                                         score=score, mean_100=mean, terminated=terminated))
                    stream.flush()
                    config["episodes_completed"] = episode
                    write_config(path, config)
                    if episode <= 5 or episode % 10 == 0 or episode == config["episodes"]:
                        window = "pending" if mean is None else f"{mean:.3f}"
                        print(f"Episode {episode}: score={score:.3f}, steps={steps}, "
                              f"terminated={terminated}, mean_100={window}", flush=True)
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
            except BaseException:
                config["status"] = "failed"
                raise
            finally:
                config["finished_at"] = datetime.now(timezone.utc).isoformat()
                write_config(path, config)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Prepared runs/<name>/config.json; edit JSON to change settings")
    args = parser.parse_args()
    try:
        run_config(args.config)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
