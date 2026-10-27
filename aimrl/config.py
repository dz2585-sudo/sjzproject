"""Load the project configuration independently of the working directory."""
import json
from pathlib import Path


def load_config():
    path = Path(__file__).resolve().parent.parent / "configs" / "base.json"
    with path.open(encoding="utf-8") as file:
        return json.load(file)
