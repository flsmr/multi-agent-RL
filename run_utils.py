"""Small shared helpers for portable configs and exclusive experiment runs."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def relative_path(value):
    """Resolve a repository-relative path without allowing escapes or personal paths."""
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("Artifact paths must be relative to the repository, without '..'.")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Artifact path escapes the repository.")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def prepared_run(config_path):
    path = Path(config_path).resolve()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This run is already in use.") from None
        config = json.loads(path.read_text())
        name = config.get("name") if isinstance(config, dict) else None
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
                or path != ROOT / "runs" / name / "config.json"):
            raise ValueError("Use runs/<name>/config.json with a matching simple run name.")
        if config.get("status") != "prepared":
            raise ValueError("Only prepared runs can start; create a fresh run folder.")
        if {p.name for p in path.parent.iterdir()} != {"config.json"}:
            raise ValueError("Prepared run must contain only config.json.")
        yield path, config
    finally:
        os.close(descriptor)
