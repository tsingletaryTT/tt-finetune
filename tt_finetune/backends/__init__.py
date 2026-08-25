"""Execution backends."""

from .blackhole import BlackholeXlaBackend
from .direct import DirectBlackholeBackend

__all__ = ["BlackholeXlaBackend", "DirectBlackholeBackend"]
