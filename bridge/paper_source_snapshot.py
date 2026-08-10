"""Reproducible identities for locally patched, pinned upstream owners.

The patch files are archival snapshots of the tracked changes already applied
inside each submodule.  They are not imported or reimplemented here.  Formal
matrix generation and execution both require the live submodule HEAD and its
exact binary diff to match the committed snapshot.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

UPSTREAM_PATCH_FILES: Final[Mapping[str, str]] = {
    "upstreams/dspy": "patches/dspy-working-tree.patch",
    "upstreams/gepa": "patches/gepa-working-tree.patch",
    "upstreams/gepa-artifact": "patches/gepa-artifact-working-tree.patch",
}
UPSTREAM_LOCK_KEYS: Final[Mapping[str, str]] = {
    "upstreams/dspy": "dspy",
    "upstreams/gepa": "gepa",
    "upstreams/gepa-artifact": "gepa_artifact",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
    ).stdout


def build_frozen_upstream_snapshot(project_root: Path) -> dict[str, dict[str, str]]:
    """Return HEAD/diff identities only after matching committed patch files."""

    root = project_root.resolve()
    lock = json.loads((root / "upstreams.lock.json").read_text(encoding="utf-8"))
    lock_sources = lock.get("sources")
    if lock.get("schema_version") != 1 or not isinstance(lock_sources, Mapping):
        raise RuntimeError("canonical upstream lock is invalid")
    snapshot: dict[str, dict[str, str]] = {}
    for relative, patch_relative in UPSTREAM_PATCH_FILES.items():
        upstream = (root / relative).resolve()
        patch = (root / patch_relative).resolve()
        if not patch.is_file():
            raise FileNotFoundError(f"missing committed upstream patch: {patch}")
        diff = _git_output(
            upstream,
            "-c",
            "core.abbrev=7",
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
        )
        patch_payload = patch.read_bytes()
        if diff != patch_payload:
            raise RuntimeError(
                f"live upstream diff differs from committed patch: {relative}"
            )
        head = _git_output(upstream, "rev-parse", "HEAD").decode().strip()
        patch_sha256 = _sha256(patch_payload)
        lock_record = lock_sources.get(UPSTREAM_LOCK_KEYS[relative])
        if not isinstance(lock_record, Mapping):
            raise TypeError(f"canonical upstream lock is missing: {relative}")
        expected_lock = {
            "path": relative,
            "commit": head,
            "patch": patch_relative,
            "patch_sha256": patch_sha256,
        }
        mismatches = {
            name: {"expected": value, "actual": lock_record.get(name)}
            for name, value in expected_lock.items()
            if lock_record.get(name) != value
        }
        if mismatches:
            raise RuntimeError(
                f"canonical upstream lock differs from live source: "
                f"{relative}: {mismatches}"
            )
        snapshot[relative] = {
            "head": head,
            "tracked_diff_sha256": _sha256(diff),
            "patch_file": patch_relative,
            "patch_sha256": patch_sha256,
        }
    return snapshot


def verify_frozen_upstream_snapshot(
    project_root: Path,
    expected: Mapping[str, Any],
) -> None:
    """Fail closed if a formal run no longer matches its upstream snapshot."""

    if set(expected) != set(UPSTREAM_PATCH_FILES):
        raise RuntimeError("formal upstream snapshot set changed")
    actual = build_frozen_upstream_snapshot(project_root)
    if actual != {name: dict(value) for name, value in expected.items()}:
        raise RuntimeError("formal upstream HEAD or tracked patch changed")


def verify_project_source_snapshot(
    project_root: Path,
    expected: Mapping[str, Any],
) -> None:
    """Verify one project revision, semantic-file set, and optional upstreams."""

    root = project_root.resolve()
    expected_head = expected.get("root_head")
    if expected_head is not None:
        if not isinstance(expected_head, str) or len(expected_head) != 40:
            raise TypeError("source_snapshot.root_head must be a Git commit")
        actual_head = _git_output(root, "rev-parse", "HEAD").decode().strip()
        if actual_head != expected_head:
            raise RuntimeError(
                "source revision differs from the frozen matrix: "
                f"expected={expected_head}, actual={actual_head}"
            )
    file_hashes = expected.get("file_sha256")
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise ValueError("source_snapshot.file_sha256 must be non-empty")
    for relative_text, expected_hash in file_hashes.items():
        if not isinstance(relative_text, str) or not relative_text:
            raise TypeError("source snapshot paths must be non-empty text")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise TypeError(
                f"source snapshot hash must be SHA256 text: {relative_text!r}"
            )
        path = (root / relative_text).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"source snapshot path escapes the project root: {relative_text!r}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"source snapshot file is missing: {relative_text}")
        actual_hash = _sha256(path.read_bytes())
        if actual_hash != expected_hash:
            raise RuntimeError(
                "source differs from the frozen matrix: "
                f"path={relative_text}, expected={expected_hash}, "
                f"actual={actual_hash}"
            )
    submodules = expected.get("submodules")
    if submodules is not None:
        if not isinstance(submodules, Mapping):
            raise TypeError("source_snapshot.submodules must be a JSON object")
        verify_frozen_upstream_snapshot(root, submodules)


__all__ = [
    "UPSTREAM_LOCK_KEYS",
    "UPSTREAM_PATCH_FILES",
    "build_frozen_upstream_snapshot",
    "verify_frozen_upstream_snapshot",
    "verify_project_source_snapshot",
]
