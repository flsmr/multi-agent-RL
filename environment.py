"""Launch the Udacity Tennis executable using the legacy Unity API-4 client."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def make_env(environment, seed=0, worker_id=1, graphics=False):
    """Return a Unity environment; callers must close it after use."""
    executable = ROOT / environment
    if not executable.is_file():
        raise FileNotFoundError("Tennis executable is missing; follow the README setup instructions.")
    sys.path.insert(0, str(ROOT / "vendor" / "unityagents_v04"))
    from unityagents import UnityEnvironment

    class TennisEnvironment(UnityEnvironment):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except BaseException:
                try:
                    self._close()
                except Exception:
                    pass  # Preserve the original startup error after cleanup.
                raise

        def executable_launcher(self, file_name, docker_training, no_graphics):
            log = ROOT / "env" / f"unity-player-{worker_id}.log"
            log.parent.mkdir(exist_ok=True)
            command = [file_name, "--port", str(self.port), "-logFile", str(log)]
            if no_graphics:
                command += ["-nographics", "-batchmode"]
            self.proc1 = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )

        def _close(self):
            self._loaded = False
            communicator = getattr(self, "communicator", None)
            try:
                if communicator is not None:
                    communicator.close()
            finally:
                try:
                    # The old client leaves its gRPC server open on startup timeout.
                    server = getattr(communicator, "server", None)
                    if server is not None:
                        server.stop(0)
                finally:
                    process = getattr(self, "proc1", None)
                    if process is not None:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=5)

    return TennisEnvironment(str(executable), seed=seed, worker_id=worker_id,
                             no_graphics=not graphics)
