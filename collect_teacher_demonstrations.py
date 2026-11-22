"""Collect episode-grouped classical distillation data from parallel frozen teachers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time

# Spawned collectors must not each start a multithreaded BLAS pool.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import torch

from environment import make_env, teacher_observations
from checkpoints import load_teacher
from run_utils import ROOT, file_hash, prepared_run, relative_path, write_json


def validation_episode(seed, episode_id, fraction):
    """Assign an entire match to a split, independently of process scheduling."""
    digest = hashlib.blake2b(f"{seed}:{episode_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < fraction


def cpu_order():
    """Use separate physical cores before siblings where Linux topology is available."""
    allowed = sorted(os.sched_getaffinity(0))
    seen, first, siblings = set(), [], []
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((topology / "physical_package_id").read_text(), (topology / "core_id").read_text())
        except OSError:
            key = (cpu,)
        if key in seen:
            siblings.append(cpu)
        else:
            seen.add(key)
            first.append(cpu)
    return first + siblings


def write_shard(directory, worker_index, chunk_index, buffers):
    arrays = {key: np.concatenate(value) for key, value in buffers.items()}
    name = f"worker_{worker_index:02d}_{chunk_index:04d}.npz"
    path = directory / name
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    report = dict(file=name, samples=len(arrays["observations"]),
                  validation_samples=int(arrays["validation"].sum()), sha256=file_hash(path))
    for value in buffers.values():
        value.clear()
    return report


def collect_worker(task, ready, results, start, stop):
    """Own one teacher/Unity pair; store pre-action inputs and their matching targets."""
    os.setsid()  # Lets the parent clean up Unity too if a worker becomes unresponsive.
    os.sched_setaffinity(0, {task["cpu"]})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.getLogger("unityagents").setLevel(logging.WARNING)
    env = None
    try:
        checkpoint = relative_path(task["teacher_checkpoint"])
        if file_hash(checkpoint) != task["teacher_sha256"]:
            raise ValueError("Teacher checkpoint changed.")
        model, _ = load_teacher(checkpoint)
        torch.manual_seed(task["seed"])
        env = make_env(task["environment"], task["seed"], task["worker_id"])
        brain = env.brain_names[0]
        local = np.asarray(env.reset(train_mode=True)[brain].vector_observations, dtype=np.float32)
        with torch.inference_mode():
            model.mean_action(torch.as_tensor(teacher_observations(local)))
        ready.put(task["index"])
        if not start.wait(timeout=120):
            raise TimeoutError("Collector start barrier timed out.")
        began = time.perf_counter()
        steps = episodes = buffered = chunk = 0
        shards = []
        buffers = {key: [] for key in ("observations", "target_actions", "episode_ids", "players", "validation")}
        benchmark = task["benchmark_seconds"] > 0
        directory = relative_path(task["dataset"]) if not benchmark else None
        with torch.inference_mode():
            while not stop.is_set():
                episode_id = (task["index"] << 32) + episodes + 1
                actions = model.mean_action(torch.as_tensor(teacher_observations(local))).numpy()
                if not benchmark:
                    buffers["observations"].append(local.copy())
                    buffers["target_actions"].append(actions.copy())
                    buffers["episode_ids"].append(np.full(2, episode_id, dtype=np.int64))
                    buffers["players"].append(np.array([0, 1], dtype=np.uint8))
                    held_out = validation_episode(task["split_seed"], episode_id, task["validation_fraction"])
                    buffers["validation"].append(np.full(2, held_out, dtype=bool))
                    buffered += 2
                info = env.step(actions)[brain]
                steps += 1
                local = np.asarray(info.vector_observations, dtype=np.float32)
                if not benchmark and buffered >= task["shard_samples"]:
                    shards.append(write_shard(directory, task["index"], chunk, buffers))
                    buffered = 0
                    chunk += 1
                if np.any(info.local_done):
                    episodes += 1
                    if not benchmark and 2 * steps >= task["quota"]:
                        break  # Finish the match rather than cutting off a trajectory.
                    local = np.asarray(env.reset(train_mode=True)[brain].vector_observations, dtype=np.float32)
                if benchmark and time.perf_counter() - began >= task["benchmark_seconds"]:
                    break
        if stop.is_set():
            raise InterruptedError("Collection cancelled.")
        if buffered:
            shards.append(write_shard(directory, task["index"], chunk, buffers))
        report = dict(index=task["index"], seed=task["seed"], worker_id=task["worker_id"],
                      samples=2 * steps, episodes=episodes, environment_steps=steps,
                      elapsed_seconds=time.perf_counter() - began, shards=shards)
        results.put(report)
    except BaseException as error:
        results.put(dict(index=task["index"], error=type(error).__name__))
        if not isinstance(error, InterruptedError):
            import traceback
            traceback.print_exc()
    finally:
        if env is not None:
            env.close()


def cleanup_workers(processes):
    """Clean up a worker's Unity process group even when the worker already died."""
    for process in processes:
        process.join(timeout=2)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join(timeout=2)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.is_alive():
            process.kill()
        process.join()


def run_workers(config, count, benchmark_seconds=0):
    """Synchronize warm workers and terminate their process groups on failure."""
    context = mp.get_context("spawn")
    ready, results = context.Queue(), context.Queue()
    start, stop = context.Event(), context.Event()
    cpus = cpu_order()
    processes = []
    reports = []
    began = time.perf_counter()
    try:
        quota = 2 * math.ceil(config["samples"] / (2 * count))
        for index in range(count):
            task = dict(config, index=index, cpu=cpus[index], seed=config["seed"] + index,
                        split_seed=config["seed"], worker_id=config["base_worker_id"] + index,
                        quota=quota, benchmark_seconds=benchmark_seconds)
            process = context.Process(target=collect_worker, args=(task, ready, results, start, stop))
            process.start()
            processes.append(process)
        warmed = set()
        while len(warmed) < count:
            try:
                warmed.add(ready.get(timeout=0.5))
            except queue.Empty:
                if not results.empty() or any(p.exitcode is not None for p in processes):
                    raise RuntimeError("A collector failed during startup; inspect its console output.")
                if time.perf_counter() - began > 120:
                    raise TimeoutError("Collector startup timed out.")
        startup = time.perf_counter() - began
        start.set()
        deadline = time.perf_counter() + config["worker_timeout_seconds"]
        while len(reports) < count:
            try:
                report = results.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes):
                    raise RuntimeError("A collector process exited unexpectedly.")
                if time.perf_counter() > deadline:
                    raise TimeoutError("Collector workload timed out.")
                continue
            if "error" in report:
                raise RuntimeError(f"Collector {report['index']} failed: {report['error']}.")
            reports.append(report)
            if not benchmark_seconds:
                print(f"Collector {len(reports)}/{count} finished; "
                      f"{sum(r['samples'] for r in reports):,} examples saved", flush=True)
        return sorted(reports, key=lambda item: item["index"]), startup
    finally:
        stop.set()
        cleanup_workers(processes)
        for channel in (ready, results):
            channel.close()
            channel.join_thread()


def select_workers(config, run_dir):
    maximum = len(cpu_order())
    if config["workers"] != "auto":
        return config["workers"]
    levels = sorted({n for n in (1, 2, 4, 8, 12, maximum) if n <= maximum})
    measurements = []
    for count in levels:
        print(f"Benchmarking {count} collector(s)...", flush=True)
        reports, startup = run_workers(config, count, config["benchmark_seconds"])
        rate = sum(r["samples"] for r in reports) / max(r["elapsed_seconds"] for r in reports)
        measurements.append(dict(workers=count, examples_per_second=rate, startup_seconds=startup))
        print(f"{count} collector(s): {rate:,.0f} examples/s", flush=True)
        write_json(run_dir / "collection_benchmark.json", dict(measurements=measurements))
    selected = max(measurements, key=lambda item: item["examples_per_second"])["workers"]
    write_json(run_dir / "collection_benchmark.json", dict(selected_workers=selected, measurements=measurements))
    return selected


def collect_from_config(config_path):
    with prepared_run(config_path) as (path, config):
        required = {"name", "status", "teacher_checkpoint", "dataset", "workers", "samples",
                    "seed", "base_worker_id", "validation_fraction", "shard_samples",
                    "benchmark_seconds", "worker_timeout_seconds"}
        if set(config) != required:
            raise ValueError("Invalid demonstration config fields.")
        for key in ("samples", "shard_samples", "benchmark_seconds", "worker_timeout_seconds"):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if config["shard_samples"] % 2:
            raise ValueError("Shard sizes must preserve two-player pairs.")
        maximum = len(cpu_order())
        if config["workers"] != "auto" and (type(config["workers"]) is not int or not 1 <= config["workers"] <= maximum):
            raise ValueError("workers must be auto or fit the available CPU allocation.")
        if (type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**31 - maximum
                or type(config["base_worker_id"]) is not int
                or not 0 <= config["base_worker_id"] <= 60530 - maximum):
            raise ValueError("Seeds or worker IDs exceed Unity's supported range.")
        if type(config["validation_fraction"]) not in (int, float) or not 0 < config["validation_fraction"] < 1:
            raise ValueError("validation_fraction must lie strictly between zero and one.")
        checkpoint = relative_path(config["teacher_checkpoint"])
        _, teacher_config = load_teacher(checkpoint)
        if not relative_path(teacher_config["environment"]).is_file():
            raise FileNotFoundError("The configured Tennis executable is missing.")
        directory = relative_path(config["dataset"])
        if not directory.is_relative_to(ROOT / "datasets") or directory == ROOT / "datasets":
            raise ValueError("Choose a fresh dataset directory under datasets/.")
        directory.mkdir(parents=True, exist_ok=False)
        config.update(status="running", teacher_sha256=file_hash(checkpoint),
                      environment=teacher_config["environment"],
                      started_at=datetime.now(timezone.utc).isoformat())
        write_json(path, config)
        manifest = dict(schema_version=1, status="running", teacher_checkpoint=config["teacher_checkpoint"],
                        teacher_sha256=config["teacher_sha256"], environment=config["environment"],
                        observation_size=24, action_size=2, target_mode="deterministic",
                        validation_fraction=config["validation_fraction"], split_seed=config["seed"],
                        split_rule="blake2b(seed:global_episode_id); whole match, both players",
                        samples=0, episodes=0, shards=[])
        write_json(directory / "manifest.json", manifest)
        began = time.perf_counter()
        try:
            count = select_workers(config, path.parent)
            config["selected_workers"] = count
            write_json(path, config)
            print(f"Collecting at least {config['samples']:,} player examples using {count} workers", flush=True)
            reports, startup = run_workers(config, count)
            if file_hash(checkpoint) != config["teacher_sha256"]:
                raise ValueError("Teacher checkpoint changed during collection.")
            manifest.update(status="complete", workers=reports,
                            samples=sum(r["samples"] for r in reports), episodes=sum(r["episodes"] for r in reports),
                            shards=[shard for r in reports for shard in r["shards"]])
            manifest["validation_samples"] = sum(s["validation_samples"] for s in manifest["shards"])
            manifest["training_samples"] = manifest["samples"] - manifest["validation_samples"]
            config.update(status="complete", collected_samples=manifest["samples"], episodes=manifest["episodes"],
                          collection_startup_seconds=startup,
                          collection_examples_per_second=manifest["samples"] / max(r["elapsed_seconds"] for r in reports))
        except BaseException as error:
            config["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["status"] = config["status"]
            raise
        finally:
            config["elapsed_seconds"] = time.perf_counter() - began
            config["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(directory / "manifest.json", manifest)
            write_json(path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        collect_from_config(args.config)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
