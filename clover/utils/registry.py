"""Generic name -> object registry shared by every plugin package.

Each plugin package (``datasets``, ``scenarios``, ``methods``, ``backbones``)
owns one :class:`Registry` instance in its ``__init__.py`` and re-exports a
``register_x`` decorator plus ``get_x`` / ``list_x`` helpers built on it. Core
code looks plugins up through these helpers and never imports a plugin module
by name.
"""

from __future__ import annotations

import difflib
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of plugins of one kind (e.g. "dataset", "method")."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers its target under ``name``."""

        def decorator(obj: T) -> T:
            if name in self._entries:
                raise ValueError(
                    f"{self._kind} '{name}' is already registered "
                    f"(registered to {self._entries[name]!r})"
                )
            self._entries[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Look up a registered entry, raising an actionable error if absent."""
        try:
            return self._entries[name]
        except KeyError:
            suggestions = difflib.get_close_matches(name, self._entries, n=1)
            hint = f" did you mean '{suggestions[0]}'?" if suggestions else ""
            raise KeyError(
                f"unknown {self._kind} '{name}'.{hint} "
                f"available: {sorted(self._entries)}"
            ) from None

    def list(self) -> list[str]:
        """Return all registered names, sorted."""
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
