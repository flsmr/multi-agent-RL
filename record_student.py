"""Record actual Unity gameplay from a frozen student on a private X display."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import ImageGrab
import torch

from checkpoints import load_policy
from environment import make_env
from run_utils import ROOT, file_hash, write_json


@contextmanager
def virtual_display():
    """Ask Xvfb for a free display; restore DISPLAY and reap it on every exit."""
    read_fd, write_fd = os.pipe()
    process = None
    previous = os.environ.get("DISPLAY")
    try:
        process = subprocess.Popen(
            ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "1280x720x24", "-nolisten", "tcp"],
            pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(write_fd)
        write_fd = None
        if not select.select([read_fd], [], [], 15)[0]:
            raise TimeoutError("Xvfb did not start within 15 seconds.")
        number = os.read(read_fd, 32).decode().strip()
        if not number.isdecimal():
            raise RuntimeError("Xvfb did not return a display number.")
        display = ":" + number
        os.environ["DISPLAY"] = display
        yield display
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        if previous is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = previous
        if process is not None:
            process.terminate()
            process.wait(timeout=5)


def record_student(checkpoint, output, seconds=12, seed=10002, worker_id=7900):
    """Capture real-time rendered play; save a GIF and checkpoint provenance."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite.")
    for command in ("Xvfb", "ffmpeg"):
        if shutil.which(command) is None:
            raise FileNotFoundError(f"Install {command} to record Unity.")
    if output.suffix != ".gif" or output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Choose a fresh .gif output path.")
    model, config, kind = load_policy(checkpoint)
    if kind != "student":
        raise ValueError("Recording requires a local student checkpoint.")
    fingerprint = file_hash(checkpoint)
    torch.set_num_threads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with virtual_display() as display, tempfile.TemporaryDirectory(prefix="tennis-record-") as temporary:
        env = None
        recorder = None
        try:
            env = make_env(config["environment"], seed=seed, worker_id=worker_id, graphics=True)
            brain = env.brain_names[0]
            info = env.reset(train_mode=False)[brain]

            def step():
                nonlocal info
                with torch.inference_mode():
                    actions = model(torch.as_tensor(np.asarray(info.vector_observations, dtype=np.float32))).numpy()
                info = env.step(actions)[brain]
                ended = bool(np.any(info.local_done))
                if ended:
                    info = env.reset(train_mode=False)[brain]
                return ended

            # Wait for a real scene instead of recording the startup's black window.
            for _ in range(100):
                step()
                screen = ImageGrab.grab(xdisplay=display)
                if np.asarray(screen).std() > 15:
                    break
            else:
                raise RuntimeError("Unity did not render a visible scene.")
            video = Path(temporary) / "gameplay.mp4"
            recorder = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-draw_mouse", "0", "-framerate", "15",
                 "-video_size", "1280x720", "-i", display, "-t", str(seconds),
                 "-an", "-c:v", "libx264", "-threads", "1", "-preset", "ultrafast",
                 "-pix_fmt", "yuv420p", str(video)], stdin=subprocess.DEVNULL)
            began = time.monotonic()
            steps = episodes = 0
            while recorder.poll() is None:
                episodes += int(step())
                steps += 1
                if time.monotonic() - began > seconds + 30:
                    raise TimeoutError("Screen recorder exceeded its time budget.")
            if recorder.returncode != 0:
                raise RuntimeError("ffmpeg screen recording failed.")
            elapsed = time.monotonic() - began
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-i", str(video),
                 "-filter_complex_threads", "1", "-filter_complex",
                 "fps=15,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                 "-loop", "0", str(output)], check=True)
            if file_hash(checkpoint) != fingerprint:
                raise ValueError("Student checkpoint changed during recording.")
            write_json(output.with_suffix(".json"), dict(
                checkpoint=str(checkpoint.resolve().relative_to(ROOT)), checkpoint_sha256=fingerprint,
                seed=seed, policy_type="student", mode="deterministic", seconds=seconds, fps=15,
                dimensions=[800, 450], environment_steps=steps, completed_episodes=episodes,
                capture_elapsed_seconds=elapsed, source="actual Unity rendering on isolated Xvfb display",
                gif_sha256=file_hash(output)))
        finally:
            if recorder is not None and recorder.poll() is None:
                recorder.terminate()
                try:
                    recorder.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    recorder.kill()
                    recorder.wait()
            if env is not None:
                env.close()
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/student_playing.gif"))
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--seed", type=int, default=10002)
    parser.add_argument("--worker-id", type=int, default=7900)
    args = parser.parse_args()
    record_student(args.checkpoint, args.output, args.seconds, args.seed, args.worker_id)
