"""On-disk cache for per-expert preprocessed inputs.

Cache key is (expert_name, mesh source path or id). Values are pickled numpy
payloads. Kept intentionally small — swap for a real KV store if you need to.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional


class PreprocessCache:
    def __init__(self, root) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, expert_name: str, key: str) -> Path:
        h = hashlib.sha1(f"{expert_name}::{key}".encode()).hexdigest()
        return self.root / f"{h}.pkl"

    def get(self, expert_name: str, key: str) -> Optional[Any]:
        p = self._path(expert_name, key)
        if not p.exists():
            return None
        with p.open("rb") as fh:
            return pickle.load(fh)

    def put(self, expert_name: str, key: str, value: Any) -> None:
        p = self._path(expert_name, key)
        with p.open("wb") as fh:
            pickle.dump(value, fh)
