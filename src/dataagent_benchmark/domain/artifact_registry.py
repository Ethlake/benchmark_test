"""Artifact registry — version counter + event log.

Tracks monotonically increasing version numbers per (kind, id) pair
and maintains an append-only event log for debugging/replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from dataagent_benchmark.domain.artifacts import ArtifactRef


@dataclass
class _Entry:
    """Internal bookkeeping for a registered artifact."""

    ref: ArtifactRef
    metadata: dict[str, Any]
    registered_at: str


class ArtifactRegistry:
    """Version counter and event log for artifacts.

    Each ``register()`` call bumps the version for the given
    ``(kind, id)`` pair and returns a new :class:`ArtifactRef`
    with the assigned version number.
    """

    def __init__(self) -> None:
        self._versions: dict[tuple[str, str], int] = {}
        self._entries: dict[tuple[str, str, int], _Entry] = {}
        self._log: list[dict[str, Any]] = []

    def register(
        self,
        kind: str,
        id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Register a new version of an artifact and return its ref."""
        key = (kind, id)
        version = self._versions.get(key, 0) + 1
        self._versions[key] = version

        ref = ArtifactRef(kind=kind, id=id, version=version)
        entry = _Entry(
            ref=ref,
            metadata=metadata or {},
            registered_at=datetime.now(tz=UTC).isoformat(),
        )
        self._entries[(kind, id, version)] = entry

        self._log.append(
            {
                "event": "register",
                "ref": str(ref),
                "metadata": entry.metadata,
                "at": entry.registered_at,
            }
        )
        return ref

    def exists(self, ref: ArtifactRef) -> bool:
        """Check whether a specific versioned artifact has been registered."""
        return (ref.kind, ref.id, ref.version) in self._entries

    def latest(self, kind: str, id: str) -> ArtifactRef | None:
        """Return the latest version of an artifact, or None."""
        key = (kind, id)
        version = self._versions.get(key)
        if version is None:
            return None
        return ArtifactRef(kind=kind, id=id, version=version)

    def get_log(self) -> list[dict[str, Any]]:
        """Return a copy of the event log."""
        return list(self._log)

    def get_metadata(self, kind: str, id: str, version: int | None = None) -> dict:
        """Return metadata for an artifact. Latest version if version is None."""
        if version is None:
            version = self._versions.get((kind, id), 0)
        entry = self._entries.get((kind, id, version))
        return dict(entry.metadata) if entry else {}

    def clear(self) -> None:
        """Reset all state (for episode boundaries)."""
        self._versions.clear()
        self._entries.clear()
        self._log.clear()
