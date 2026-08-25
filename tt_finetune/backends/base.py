"""Backend contract kept independent from the notebook-facing trainer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TrainingBackend(ABC):
    """Execution boundary for preflight, smoke, and full training."""

    name: str

    @abstractmethod
    def preflight(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def smoke(self, steps: int = 2) -> Any:
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        resume: bool | str = False,
        stop_after: int | None = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def save_config(self, path: str | Path) -> Path:
        raise NotImplementedError

    def compile_profile(self) -> dict[str, Any]:
        """Compile/profile the current semantic geometry without training it."""

        raise NotImplementedError(
            f"{type(self).__name__} does not expose profile compilation"
        )

    def profile_cache_inventory(self) -> dict[str, Any]:
        """Inspect measured execution profiles visible to this backend."""

        raise NotImplementedError(
            f"{type(self).__name__} does not expose a profile cache"
        )

    def data_plan(self) -> dict[str, Any]:
        """Inspect deterministic data geometry without initializing a device."""

        raise NotImplementedError(
            f"{type(self).__name__} does not expose data planning"
        )
