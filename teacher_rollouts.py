"""Collect PPO rollouts with independent frozen teacher copies in Unity workers."""
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time

import numpy as np
import torch

from environment import make_env, teacher_observations
from teacher_model import TeacherActorCritic
from teacher_ppo import TeacherRolloutBuffer

ROLLOUT_FIELDS = ("states", "latent_actions", "rewards", "values", "log_probs",
                  "next_values", "terminated", "ended")


class TeacherEpisodeTracker:
    """Accumulate player returns separately for every game, including across rollouts."""
    def __init__(self, num_envs):
        self.returns = np.zeros((num_envs, 2), dtype=np.float64)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.episodes = np.zeros(num_envs, dtype=np.int64)

    def add(self, rewards, ended, time_limits):
        self.returns += rewards
        self.steps += 1
        completed = []
        for index in np.flatnonzero(ended):
            self.episodes[index] += 1
            completed.append(dict(
                environment_index=int(index), environment_episode=int(self.episodes[index]),
                steps=int(self.steps[index]), return_agent0=float(self.returns[index, 0]),
                return_agent1=float(self.returns[index, 1]), score=float(self.returns[index].max()),
                terminated=not bool(time_limits[index]), time_limit=bool(time_limits[index])))
            self.returns[index].fill(0)
            self.steps[index] = 0
        return completed


class LocalTeacherCollector:
    """Own one game and collect a complete rollout without parent communication."""
    def __init__(self, environment, seed, worker_id):
        self.env = make_env(environment, seed, worker_id)
        try:
            self.brain = self.env.brain_names[0]
            if self.env.brains[self.brain].vector_action_space_size != 2:
                raise RuntimeError("Expected two Tennis actions per player.")
            self.states = self._states(self.env.reset(train_mode=True)[self.brain])
            self.tracker = TeacherEpisodeTracker(1)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _states(info):
        return teacher_observations(info.vector_observations)

    def collect(self, model, rollout_steps):
        rollout = TeacherRolloutBuffer(rollout_steps, 2, 48, 2)
        episodes = []
        with torch.inference_mode():
            for step in range(rollout_steps):
                latent, actions, log_probs, values = model.sample(torch.from_numpy(self.states))
                info = self.env.step(actions.numpy())[self.brain]
                final_states = self._states(info)
                rewards = np.asarray(info.rewards, dtype=np.float32)
                if rewards.shape != (2,) or not np.isfinite(rewards).all():
                    raise RuntimeError("Expected two finite Tennis rewards.")
                ended = bool(np.any(info.local_done))
                time_limit = ended and bool(np.any(info.max_reached))
                _, next_values = model(torch.from_numpy(final_states))
                rollout.add(self.states, latent.numpy(), rewards, values.numpy(), log_probs.numpy(),
                            next_values.numpy(), np.full(2, ended and not time_limit, dtype=bool),
                            np.full(2, ended, dtype=bool))
                for episode in self.tracker.add(rewards[None], [ended], [time_limit]):
                    episodes.append(dict(episode, rollout_step=step + 1))
                # Retain terminal observations above for bootstrapping; reset only ended games.
                self.states = self._states(self.env.reset(train_mode=True)[self.brain]) if ended else final_states
        return dict(arrays={field: getattr(rollout, field) for field in ROLLOUT_FIELDS}, episodes=episodes)

    def close(self):
        self.env.close()


def _copy_parameters(model, shared_weights):
    flat = np.frombuffer(shared_weights, dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(torch.from_numpy(flat[offset:offset + count]).view_as(parameter))
            offset += count
    if offset != flat.size:
        raise RuntimeError("Teacher snapshot does not match the worker architecture.")


def _worker(connection, shared_weights, hidden_size, environment, seed, worker_id):
    os.setsid()  # Unity belongs to this group even if its Python worker crashes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logging.getLogger("unityagents").setLevel(logging.WARNING)
    collector = None
    try:
        model = TeacherActorCritic(48, 2, hidden_size=hidden_size).eval().requires_grad_(False)
        torch.manual_seed(seed)
        collector = LocalTeacherCollector(environment, seed, worker_id)
        connection.send(("ok", None))
        previous_version = 0
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command != "collect":
                raise RuntimeError("Unknown teacher worker command.")
            rollout_steps, version = payload
            if version <= previous_version:
                raise RuntimeError("Teacher policy versions must increase between rollouts.")
            _copy_parameters(model, shared_weights)
            previous_version = version
            result = collector.collect(model, rollout_steps)
            connection.send(("ok", dict(result, policy_version=version)))
    except EOFError:
        pass
    except BaseException as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            if collector is not None:
                collector.close()
        finally:
            connection.close()


class TeacherRolloutPool:
    """One changing parent policy; one frozen copy per worker during collection.

    Parallel workers exchange a parameter snapshot and complete trajectories once
    per rollout. Shared parameters stay unchanged until all workers finish. Games
    persist between calls. A single environment collects directly in the parent.
    Startup and each parallel rollout have a whole-batch timeout.
    """
    def __init__(self, model, environment, seed=0, worker_id=1, num_envs=1, timeout_seconds=120):
        if type(num_envs) is not int or not 1 <= num_envs <= len(os.sched_getaffinity(0)):
            raise ValueError("num_envs must fit the available CPU allocation.")
        if (type(seed) is not int or not 0 <= seed <= 2**31 - num_envs
                or type(worker_id) is not int or not 0 <= worker_id <= 60531 - num_envs):
            raise ValueError("Environment seeds or worker IDs exceed supported ranges.")
        if type(timeout_seconds) not in (int, float) or not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite.")
        self.model = model
        self.num_envs = num_envs
        self.timeout_seconds = timeout_seconds
        self.processes, self.connections, self.readers = [], [], []
        self.results = queue.Queue()
        self.local = None
        self.closed = False
        self.policy_version = 0
        try:
            if num_envs == 1:
                self.local = LocalTeacherCollector(environment, seed, worker_id)
            else:
                context = mp.get_context("spawn")
                self.shared_weights = context.RawArray("f", sum(p.numel() for p in model.parameters()))
                hidden_size = model.actor[0].out_features
                for index in range(num_envs):
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker, args=(
                        child, self.shared_weights, hidden_size, environment, seed + index, worker_id + index))
                    self.connections.append(parent)
                    self.processes.append(process)
                    try:
                        process.start()
                    finally:
                        child.close()
                    reader = threading.Thread(target=self._read, args=(index, parent), daemon=True)
                    reader.start()
                    self.readers.append(reader)
                self._receive()
        except BaseException:
            self.close()
            raise

    def _read(self, index, connection):
        # A large partial pipe message must not block the parent's timeout handling.
        while not self.closed:
            try:
                status, payload = connection.recv()
            except (EOFError, OSError) as error:
                if not self.closed:
                    self.results.put((index, "error", f"Worker disconnected: {type(error).__name__}"))
                return
            self.results.put((index, status, payload))
            if status != "ok":
                return

    def _receive(self):
        deadline = time.monotonic() + self.timeout_seconds
        results = [None] * self.num_envs
        pending = set(range(self.num_envs))
        while pending:
            try:
                index, status, payload = self.results.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError("Parallel teacher workers timed out.") from None
            if index not in pending or status != "ok":
                raise RuntimeError(f"Teacher worker {index} failed: {payload}")
            pending.remove(index)
            results[index] = payload
        return results

    def collect(self, rollout_steps):
        if self.closed:
            raise RuntimeError("The teacher rollout pool is closed.")
        if type(rollout_steps) is not int or rollout_steps <= 0:
            raise ValueError("rollout_steps must be a positive integer.")
        self.policy_version += 1
        try:
            if self.local is not None:
                results = [dict(self.local.collect(self.model, rollout_steps), policy_version=self.policy_version)]
            else:
                # Only the parent writes this snapshot, before issuing collect commands.
                flat = np.frombuffer(self.shared_weights, dtype=np.float32)
                offset = 0
                for parameter in self.model.parameters():
                    count = parameter.numel()
                    flat[offset:offset + count] = parameter.detach().cpu().numpy().reshape(-1)
                    offset += count
                for connection in self.connections:
                    connection.send(("collect", (rollout_steps, self.policy_version)))
                results = self._receive()
            rollout = TeacherRolloutBuffer(rollout_steps, 2 * self.num_envs, 48, 2)
            episodes = []
            for index, result in enumerate(results):
                if result["policy_version"] != self.policy_version:
                    raise RuntimeError("Workers returned different teacher policy versions.")
                for field in ROLLOUT_FIELDS:
                    destination = getattr(rollout, field)[:, 2 * index:2 * index + 2]
                    source = result["arrays"][field]
                    if source.shape != destination.shape or source.dtype != destination.dtype:
                        raise RuntimeError("Worker returned malformed teacher trajectory data.")
                    if not np.isfinite(source).all():
                        raise RuntimeError("Worker returned nonfinite teacher trajectory data.")
                    destination[:] = source
                episodes.extend(dict(episode, environment_index=index) for episode in result["episodes"])
            rollout.position = rollout_steps
            episodes.sort(key=lambda row: (row["rollout_step"], row["environment_index"]))
            return rollout, episodes
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.local is not None:
                self.local.close()
        finally:
            for connection in self.connections:
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            deadline = time.monotonic() + 1
            started = [process for process in self.processes if process.pid is not None]
            for process in started:
                process.join(timeout=max(0, deadline - time.monotonic()))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for process in started:
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        if process.is_alive():
                            process.kill() if sig == signal.SIGKILL else process.terminate()
                deadline = time.monotonic() + 1
                for process in started:
                    process.join(timeout=max(0, deadline - time.monotonic()))
            for connection in self.connections:
                connection.close()
            for reader in self.readers:
                reader.join(timeout=.1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
