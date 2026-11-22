"""Evaluate a shared teacher or local student without updating its weights."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from environment import make_env, teacher_observations
from checkpoints import load_policy
from run_utils import file_hash, write_json

ROOT = Path(__file__).resolve().parent


def evaluate(checkpoint_path, episodes=100, seed=10002, worker_id=10,
             mode="auto", graphics=False, max_steps=10000):
    """Return per-episode scores and a summary from a fixed, independently seeded policy."""
    if episodes <= 0 or max_steps <= 0 or mode not in ("auto", "stochastic", "deterministic"):
        raise ValueError("Choose positive episode/step limits and a valid evaluation mode.")
    torch.set_num_threads(1)
    model, config, kind = load_policy(checkpoint_path)
    mode = ("deterministic" if kind == "student" else "stochastic") if mode == "auto" else mode
    if kind == "student" and mode != "deterministic":
        raise ValueError("The distilled student is a deterministic actor.")
    torch.manual_seed(seed)
    env = make_env(config["environment"], seed=seed, worker_id=worker_id, graphics=graphics)
    rows = []
    try:
        brain = env.brain_names[0]
        for episode in range(1, episodes + 1):
            info = env.reset(train_mode=not graphics)[brain]
            local_states = np.asarray(info.vector_observations, dtype=np.float32)
            states = teacher_observations(local_states) if kind == "teacher" else local_states
            returns = np.zeros(2, dtype=np.float64)
            terminated = False
            episode_complete = False
            for steps in range(1, max_steps + 1):
                with torch.no_grad():
                    tensor = torch.as_tensor(states)
                    if kind == "student":
                        actions = model(tensor)
                    elif mode == "deterministic":
                        means, _ = model(tensor)
                        actions = torch.tanh(means)
                    else:
                        _, actions, _, _ = model.sample(tensor)
                info = env.step(actions.numpy())[brain]
                returns += info.rewards
                local_states = np.asarray(info.vector_observations, dtype=np.float32)
                states = teacher_observations(local_states) if kind == "teacher" else local_states
                if np.any(info.local_done):
                    episode_complete = True
                    terminated = not bool(np.any(info.max_reached))
                    break
            rows.append(dict(episode=episode, steps=steps,
                             return_agent0=float(returns[0]), return_agent1=float(returns[1]),
                             score=float(returns.max()), terminated=terminated,
                             episode_complete=episode_complete))
            if episode % 10 == 0 or episode == episodes:
                print(f"{mode}: {episode}/{episodes} episodes, mean score="
                      f"{np.mean([row['score'] for row in rows]):.3f}", flush=True)
    finally:
        env.close()
    scores = [row["score"] for row in rows]
    qualified_windows = [float(np.mean(scores[end-100:end]))
                         for end in range(100, len(rows) + 1)
                         if all(row["episode_complete"] for row in rows[end-100:end])]
    summary = dict(policy_type=kind, mode=mode, seed=seed, episodes=episodes, max_steps=max_steps,
                   mean_score=float(np.mean(scores)),
                   mean_agent_returns=[float(np.mean([row[f"return_agent{i}"] for row in rows])) for i in range(2)],
                   truncated_episodes=sum(not row["episode_complete"] for row in rows),
                   time_limit_episodes=sum(row["episode_complete"] and not row["terminated"] for row in rows),
                   final_rolling100=float(np.mean(scores[-100:])) if len(rows) >= 100
                   and all(row["episode_complete"] for row in rows[-100:]) else None,
                   solved=any(mean >= 0.5 for mean in qualified_windows))
    return rows, summary


def summarize_seeds(cases):
    """Report variability between independent seed means, not between episodes."""
    means = np.asarray([case["mean_score"] for case in cases], dtype=np.float64)
    if not len(means):
        raise ValueError("No evaluation cases to summarize.")
    return dict(mean_score=float(means.mean()), seed_mean_std=float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                minimum_seed_mean=float(means.min()), maximum_seed_mean=float(means.max()),
                passed_seeds=sum(bool(case["solved"]) for case in cases), seed_count=len(cases),
                episodes=sum(case["episodes"] for case in cases),
                truncated_episodes=sum(case["truncated_episodes"] for case in cases),
                time_limit_episodes=sum(case["time_limit_episodes"] for case in cases))


def evaluate_seeds(checkpoint_path, seeds, output_dir, episodes=100, worker_id=200,
                   mode="auto", workers=4, max_steps=10000):
    """Evaluate frozen weights in independent Unity processes and merge auditable results."""
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
            or episodes <= 0 or workers <= 0 or max_steps <= 0 or not 0 <= worker_id <= 60530 - len(seeds)):
        raise ValueError("Invalid seeds, worker count, episode limit, or communication ports.")
    checkpoint_path = Path(checkpoint_path).resolve()
    reference = str(checkpoint_path.relative_to(ROOT))
    _, _, kind = load_policy(checkpoint_path)
    mode = ("deterministic" if kind == "student" else "stochastic") if mode == "auto" else mode
    if mode not in ("stochastic", "deterministic") or (kind == "student" and mode != "deterministic"):
        raise ValueError("Invalid evaluation mode for the saved policy.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(status="running", policy_type=kind, mode=mode, checkpoint=reference,
                   checkpoint_sha256=file_hash(checkpoint_path), seeds=seeds, episodes_per_seed=episodes,
                   workers=min(workers, len(seeds)), max_steps=max_steps,
                   seed_timeout_seconds=max(120, episodes * 10),
                   started_at=datetime.now(timezone.utc).isoformat(), per_seed=[])
    write_json(output_dir / "summary.json", summary)
    began = time.perf_counter()
    processes, running, cases, all_rows = [], {}, {}, {}
    try:
        with tempfile.TemporaryDirectory(prefix="tennis-evaluation-") as temporary:
            pending = iter(enumerate(seeds))
            exhausted = False
            while running or not exhausted:
                while len(running) < workers and not exhausted:
                    try:
                        index, seed = next(pending)
                    except StopIteration:
                        exhausted = True
                        break
                    destination = Path(temporary) / f"seed_{seed}"
                    command = [sys.executable, str(ROOT / "evaluate.py"), "--checkpoint", reference,
                               "--seed", str(seed), "--episodes", str(episodes), "--worker-id", str(worker_id + index),
                               "--mode", mode, "--max-steps", str(max_steps), "--output-dir", str(destination)]
                    console_path = Path(temporary) / f"seed_{seed}.log"
                    with console_path.open("w") as console:
                        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                                   stdout=console, stderr=subprocess.STDOUT, text=True)
                    processes.append(process)
                    running[seed] = (process, destination, console_path, time.perf_counter())
                for seed, (process, destination, console_path, launched) in list(running.items()):
                    if process.poll() is None:
                        if time.perf_counter() - launched > summary["seed_timeout_seconds"]:
                            raise TimeoutError(f"Evaluation seed {seed} timed out.")
                        continue
                    if process.returncode:
                        raise RuntimeError(f"Evaluation seed {seed} failed:\n{console_path.read_text()[-2000:]}")
                    cases[seed] = json.loads((destination / f"evaluation_{mode}.json").read_text())
                    with (destination / f"evaluation_{mode}.csv").open() as stream:
                        all_rows[seed] = [dict(seed=seed, **row) for row in csv.DictReader(stream)]
                    del running[seed]
                    summary["per_seed"] = [cases[value] for value in seeds if value in cases]
                    write_json(output_dir / "summary.json", summary)
                    print(f"{kind}/{mode}: seed {seed}, mean={cases[seed]['mean_score']:.3f}, "
                          f"passed={cases[seed]['solved']} ({len(cases)}/{len(seeds)})", flush=True)
                if running:
                    time.sleep(0.05)
            if file_hash(checkpoint_path) != summary["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during evaluation.")
            rows = [row for seed in seeds for row in all_rows[seed]]
            with (output_dir / "episodes.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            summary.update(status="complete", **summarize_seeds(summary["per_seed"]))
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    finally:
        # Attempt group cleanup even after the Python child has died.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        summary.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - began)
        write_json(output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int, default=10002)
    seed_options.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent processes for --seeds")
    parser.add_argument("--worker-id", type=int, default=10)
    parser.add_argument("--mode", choices=("auto", "stochastic", "deterministic"), default="auto")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--graphics", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        checkpoint_reference = str(args.checkpoint.resolve().relative_to(ROOT))
    except ValueError:
        parser.error("Use a checkpoint inside this repository so its saved path stays portable.")
    if args.mode == "auto":
        _, _, kind = load_policy(args.checkpoint)
        args.mode = "deterministic" if kind == "student" else "stochastic"
    if args.seeds:
        if args.graphics:
            parser.error("Use a single seed to watch the simulator.")
        destination = args.output_dir or args.checkpoint.parent / f"evaluation_{args.mode}_seeds"
        result = evaluate_seeds(args.checkpoint, args.seeds, destination, args.episodes,
                                args.worker_id, args.mode, args.workers, args.max_steps)
        print(json.dumps(result, indent=2))
        parser.exit()
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"evaluation_{args.mode}.json"
    scores_path = output_dir / f"evaluation_{args.mode}.csv"
    if summary_path.exists() or scores_path.exists():
        parser.error("Evaluation outputs already exist; choose a fresh --output-dir.")
    rows, summary = evaluate(args.checkpoint, args.episodes, args.seed, args.worker_id,
                             args.mode, args.graphics, args.max_steps)
    # Store portable paths only, even if the command received an absolute path.
    summary["checkpoint"] = checkpoint_reference
    with scores_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
