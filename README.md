# Multi-agent Tennis

A minimal workspace for exploring Udacity's two-player Tennis environment and
comparing reinforcement learning experiments. The current baseline runs random
actions; training algorithms, successful weights, and the project report will be
added after the learning approach is chosen.

## Environment and scoring

Two players control rackets to keep a ball in play. Each player receives its own
local observations of the ball and racket. A successful return earns `+0.1`;
letting the ball fall or sending it out of bounds incurs `-0.01`.

Each observation has **8 values**, stacked across **3 frames** by this executable,
so the Python API returns **24 values per player**. Each player chooses **2
continuous actions** in `[-1, 1]`: movement toward/away from the net and jumping.
The joint action array therefore has shape `(2, 2)`.

An episode's score is the **maximum of the two players' total undiscounted
returns**. The environment is solved when the mean of those scores over **100
consecutive episodes is at least 0.5**. These rules follow the
[Udacity project description](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

## Setup

Requirements: Linux x86-64, Python 3.10, `curl`, and `unzip`. From the repository
root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p env
curl -fL --retry 2 \
  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip \
  -o env/Tennis_Linux.zip
unzip env/Tennis_Linux.zip -d env
chmod +x env/Tennis_Linux/Tennis.x86_64
rm env/Tennis_Linux.zip
```

The executable should be at `env/Tennis_Linux/Tennis.x86_64`, alongside its
`Tennis_Data/` directory. Unity itself is not required. The simulator and Python
virtual environment stay local and are ignored by Git.

The small legacy Unity API-4 client is bundled in `vendor/` with its third-party
licenses. Current `mlagents_envs` packages cannot replace it for this executable.
Only NumPy, Pillow, protobuf, and gRPC are needed for the random baseline.

## Run a random policy

```bash
source .venv/bin/activate
python random_policy.py --config runs/random_seed0/config.json
```

This starts five episodes, sampling each player's actions independently and
uniformly from `[-1, 1]`. The config uses `graphics: false` for fast execution.
To watch the simulator, set `graphics` to `true` in a fresh config and run from a
graphical desktop. If the standard Linux build cannot initialize on a server,
try `xvfb-run -a python random_policy.py --config ...` with Xvfb installed.

The script prints the observed dimensions and scores, closes Unity after the
run, and saves:

- `config.json`: settings, observed dimensions, progress, timestamps, and status.
- `episodes.csv`: episode number, steps, both player returns, maximum score,
  100-episode mean, and whether the episode terminated naturally.

`max_steps` limits each episode. Episodes stopped at that limit are logged as
truncated and reset the solve window. The mean stays empty until 100 consecutive
complete episodes are available. Completing a random-policy run does not mean
the environment has been solved; `solved_episode` records the first qualifying
window, or remains `null`.

`runs/random_seed0/` is a fresh configuration ready to launch.

## Separate experiments

Each experiment gets its own folder and JSON settings. To prepare another run,
copy only the input settings, including when the source run has already finished:

```bash
python - <<'PY'
import json
from pathlib import Path

source = json.loads(Path("runs/random_seed0/config.json").read_text())
fields = ["policy", "seed", "worker_id", "episodes", "max_steps", "graphics", "environment"]
config = {key: source[key] for key in fields}
config.update(name="random_seed1", status="prepared", seed=1, worker_id=2, episodes=100)
folder = Path("runs") / config["name"]
folder.mkdir()
(folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
python random_policy.py --config runs/random_seed1/config.json
```

Edit the prepared JSON to change settings. Run names must match their folder and
use letters, digits, underscores, or hyphens. Executable paths are relative to the
repository. Concurrent simulator processes require distinct `worker_id` values,
which select the communication port `5005 + worker_id`.

Only a `prepared` folder containing just `config.json` can start. Its status
changes to `running`, then `complete`, `failed`, or `interrupted`. Existing runs
are never restarted or overwritten; create a new folder instead. Episode CSVs
are flushed as they are recorded. Automatic resume is not implemented.

## Repository structure

```text
environment.py          Tennis launcher and legacy client adapter
random_policy.py        Config validation, random actions, scoring, and logging
requirements.txt        Minimal exploration dependencies
runs/<name>/            Per-experiment configs and episode CSVs
vendor/                 Compatible Unity client and third-party licenses
env/                    Local simulator files (ignored)
.venv/                  Local Python environment (ignored)
```

The eventual submission will add Python/PyTorch training code, successful model
weights, and a root-level `Report.md` documenting the algorithm, architecture,
hyperparameters, reward plot, solve episode, and concrete future improvements.
