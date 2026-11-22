"""Train a local actor offline to imitate frozen deterministic teacher actions."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from student_model import StudentActor
from run_utils import file_hash, prepared_run, relative_path, write_json


class EarlyStopping:
    """Save exact minima while measuring patience against cumulative progress."""

    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best = self.significant_best = float("inf")
        self.stale = 0

    def observe(self, loss):
        if not np.isfinite(loss):
            raise FloatingPointError("Validation MSE became non-finite.")
        improved = loss < self.best
        self.best = min(self.best, loss)
        if loss < self.significant_best - self.min_delta:
            self.significant_best, self.stale = loss, 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience


def validate_partition(episode_ids, validation):
    """Reject frame/player leakage between training and validation matches."""
    if episode_ids.ndim != 1 or episode_ids.shape != validation.shape or not len(episode_ids):
        raise ValueError("Invalid episode partition arrays.")
    order = np.argsort(episode_ids, kind="stable")
    ids, held_out = episode_ids[order], validation[order]
    if np.any((ids[1:] == ids[:-1]) & (held_out[1:] != held_out[:-1])):
        raise ValueError("A match appears in both training and validation.")


def load_dataset(directory):
    """Validate and load small numeric shards; Unity and the teacher are not needed."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("status"), manifest.get("schema_version"), manifest.get("observation_size"),
            manifest.get("action_size"), manifest.get("target_mode")) != ("complete", 1, 24, 2, "deterministic"):
        raise ValueError("Expected a complete, local-observation teacher demonstration dataset.")
    relative_path(manifest["environment"])
    fields = ("observations", "target_actions", "episode_ids", "players", "validation")
    pieces = {key: [] for key in fields}
    names = set()
    for shard in manifest["shards"]:
        name = shard["file"]
        if Path(name).name != name or not name.endswith(".npz") or name in names:
            raise ValueError("Invalid or repeated shard filename.")
        names.add(name)
        path = directory / name
        if file_hash(path) != shard["sha256"]:
            raise ValueError(f"Dataset shard checksum failed: {name}.")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(fields):
                raise ValueError("Dataset shard has unexpected fields.")
            arrays = {key: archive[key] for key in fields}
        count = len(arrays["observations"])
        if (count != shard["samples"] or not count or count % 2
                or arrays["observations"].shape != (count, 24)
                or arrays["target_actions"].shape != (count, 2)
                or any(arrays[key].shape != (count,) for key in ("episode_ids", "players", "validation"))):
            raise ValueError("Invalid shard shapes or example count.")
        if (arrays["observations"].dtype != np.float32 or arrays["target_actions"].dtype != np.float32
                or arrays["episode_ids"].dtype != np.int64 or arrays["players"].dtype != np.uint8
                or arrays["validation"].dtype != bool):
            raise ValueError("Invalid shard dtypes.")
        if (not np.isfinite(arrays["observations"]).all()
                or not np.isfinite(arrays["target_actions"]).all()
                or np.any(np.abs(arrays["target_actions"]) > 1)):
            raise ValueError("Invalid observations or bounded teacher actions.")
        if (not np.all(arrays["players"].reshape(-1, 2) == [0, 1])
                or not np.all(arrays["episode_ids"][::2] == arrays["episode_ids"][1::2])
                or not np.all(arrays["validation"][::2] == arrays["validation"][1::2])):
            raise ValueError("Player pairs lost their match/split alignment.")
        if int(arrays["validation"].sum()) != shard["validation_samples"]:
            raise ValueError("Shard validation count does not match its manifest.")
        for key in fields:
            pieces[key].append(arrays[key])
    if not names:
        raise ValueError("Dataset is empty.")
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    mask = arrays["validation"]
    if (len(mask) != manifest["samples"] or int(mask.sum()) != manifest["validation_samples"]
            or int((~mask).sum()) != manifest["training_samples"] or not mask.any() or mask.all()):
        raise ValueError("Dataset split counts are invalid or a split is empty.")
    validate_partition(arrays["episode_ids"], mask)
    if int(np.unique(arrays["episode_ids"]).size) != manifest["episodes"]:
        raise ValueError("Match count does not agree with the dataset manifest.")
    return arrays, manifest


def measure(model, observations, targets, batch_size):
    squared = torch.zeros(2, device=observations.device, dtype=torch.float64)
    agreement = 0
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            prediction = model(observations[start:start + batch_size])
            expected = targets[start:start + batch_size]
            squared += (prediction - expected).square().sum(dim=0).double()
            agreement += int(((prediction[:, 1] > 0.5) == (expected[:, 1] > 0.5)).sum())
    errors = (squared / len(observations)).cpu().tolist()
    return dict(mse=sum(errors) / 2, movement_mse=errors[0], jump_mse=errors[1],
                jump_agreement=agreement / len(observations))


def choose_cpu_threads(observations, targets, config):
    """Benchmark actual student optimizer steps, without modifying the trained student."""
    import os
    levels = [n for n in (1, 2, 4, 8, 16) if n <= len(os.sched_getaffinity(0))]
    measurements = []
    size = min(config["batch_size"], len(observations))
    inputs, labels = observations[:size], targets[:size]
    for threads in levels:
        torch.set_num_threads(threads)
        probe = StudentActor(config["hidden_size"])
        optimizer = torch.optim.Adam(probe.parameters(), lr=config["learning_rate"])
        began = None
        for step in range(24):
            if step == 4:
                began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            (probe(inputs) - labels).square().mean().backward()
            optimizer.step()
        elapsed = time.perf_counter() - began
        measurements.append(dict(threads=threads, optimizer_steps_per_second=20 / elapsed))
    selected = max(measurements, key=lambda row: row["optimizer_steps_per_second"])["threads"]
    torch.set_num_threads(selected)
    return selected, measurements


def save_student(path, model, config, epoch, metrics):
    checkpoint = dict(policy_type="student", state_size=24, action_size=2,
                      hidden_size=config["hidden_size"], epoch=epoch, validation=metrics,
                      model_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                      config=config.copy())
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def distill_from_config(config_path):
    with prepared_run(config_path) as (path, config):
        required = {"name", "status", "dataset", "seed", "hidden_size", "learning_rate", "batch_size",
                    "epochs", "patience", "min_delta", "device", "cpu_threads"}
        if set(config) != required:
            raise ValueError("Invalid student config fields.")
        for key in ("hidden_size", "batch_size", "epochs", "patience"):
            if type(config[key]) is not int or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**31:
            raise ValueError("seed must be a nonnegative signed 32-bit integer.")
        for key in ("learning_rate", "min_delta"):
            if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or config[key] < 0:
                raise ValueError(f"Invalid {key}.")
        if config["learning_rate"] == 0:
            raise ValueError("learning_rate must be positive.")
        if config["device"] not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this Python environment; use CPU or a CUDA-enabled installation.")
        import os
        if config["cpu_threads"] != "auto" and (type(config["cpu_threads"]) is not int
                or not 1 <= config["cpu_threads"] <= len(os.sched_getaffinity(0))):
            raise ValueError("cpu_threads must be auto or fit the available CPU allocation.")
        directory = relative_path(config["dataset"])
        arrays, manifest = load_dataset(directory)
        mask = arrays["validation"]
        train_x = torch.from_numpy(arrays["observations"][~mask])
        train_y = torch.from_numpy(arrays["target_actions"][~mask])
        valid_x = torch.from_numpy(arrays["observations"][mask])
        valid_y = torch.from_numpy(arrays["target_actions"][mask])
        config.update(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                      environment=manifest["environment"], dataset_manifest_sha256=file_hash(directory / "manifest.json"),
                      teacher_sha256=manifest["teacher_sha256"], training_samples=len(train_x),
                      validation_samples=len(valid_x), training_episodes=int(np.unique(arrays["episode_ids"][~mask]).size),
                      validation_episodes=int(np.unique(arrays["episode_ids"][mask]).size),
                      epochs_completed=0, best_epoch=None, best_validation_mse=None,
                      software={"torch": str(torch.__version__), "numpy": np.__version__, "device": config["device"]})
        write_json(path, config)
        began = time.perf_counter()
        model = None
        try:
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            if config["device"] == "cpu" and config["cpu_threads"] == "auto":
                threads, measurements = choose_cpu_threads(train_x, train_y, config)
                write_json(path.parent / "cpu_benchmark.json", dict(selected_threads=threads, measurements=measurements))
            else:
                threads = config["cpu_threads"] if config["cpu_threads"] != "auto" else 1
                torch.set_num_threads(threads)
            config["selected_cpu_threads"] = threads
            torch.manual_seed(config["seed"])
            device = torch.device(config["device"])
            train_x, train_y = train_x.to(device), train_y.to(device)
            valid_x, valid_y = valid_x.to(device), valid_y.to(device)
            model = StudentActor(config["hidden_size"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            stopping = EarlyStopping(config["patience"], config["min_delta"])
            last_metrics = {}
            initial = measure(model, valid_x, valid_y, config["batch_size"])
            config["initial_validation_mse"] = initial["mse"]
            print(f"CPU threads={threads}; training={len(train_x):,}, validation={len(valid_x):,}; "
                  f"initial validation MSE={initial['mse']:.6f}", flush=True)
            fields = ("epoch", "train_mse", "validation_mse", "movement_mse", "jump_mse", "jump_agreement", "elapsed_seconds")
            with (path.parent / "losses.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for epoch in range(1, config["epochs"] + 1):
                    model.train()
                    order = torch.randperm(len(train_x), device=device)
                    loss_sum = 0.0
                    for start in range(0, len(order), config["batch_size"]):
                        indices = order[start:start + config["batch_size"]]
                        loss = (model(train_x[indices]) - train_y[indices]).square().mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Student MSE became non-finite.")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(loss.detach()) * len(indices)
                    model.eval()
                    last_metrics = measure(model, valid_x, valid_y, config["batch_size"])
                    validation = last_metrics["mse"]
                    improved, stop = stopping.observe(validation)
                    config["epochs_completed"] = epoch
                    if improved:
                        config.update(best_epoch=epoch, best_validation_mse=stopping.best)
                        save_student(path.parent / "best.pt", model, config, epoch, last_metrics)
                    writer.writerow(dict(epoch=epoch, train_mse=loss_sum / len(train_x), validation_mse=validation,
                                         movement_mse=last_metrics["movement_mse"], jump_mse=last_metrics["jump_mse"],
                                         jump_agreement=last_metrics["jump_agreement"], elapsed_seconds=time.perf_counter() - began))
                    stream.flush()
                    write_json(path, config)
                    print(f"Epoch {epoch}/{config['epochs']}: train MSE={loss_sum / len(train_x):.6f}, "
                          f"validation MSE={validation:.6f}, jump agreement={last_metrics['jump_agreement']:.3%}", flush=True)
                    if stop:
                        config["stop_reason"] = "validation_patience"
                        break
                else:
                    config["stop_reason"] = "epoch_budget"
            config.update(status="complete", final_validation=last_metrics)
        except BaseException as error:
            config["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            raise
        finally:
            config.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - began)
            if model is not None:
                save_student(path.parent / "checkpoint.pt", model, config, config["epochs_completed"],
                             locals().get("last_metrics", {}))
            write_json(path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        distill_from_config(args.config)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
