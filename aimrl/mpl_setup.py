"""Set a writable Matplotlib cache before importing Matplotlib."""
import os
import tempfile
from pathlib import Path


def configure_mpl():
    project_cache = Path(__file__).resolve().parent.parent / ".mplconfig"
    candidates = ([Path(os.environ["MPLCONFIGDIR"]).expanduser()] if os.environ.get("MPLCONFIGDIR") else []) + [project_cache]
    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path):
                pass
            os.environ["MPLCONFIGDIR"] = str(path.resolve())
            return os.environ["MPLCONFIGDIR"]
        except OSError:
            continue
    path = tempfile.mkdtemp(prefix="aimrl-mpl-")
    os.environ["MPLCONFIGDIR"] = path
    return path
