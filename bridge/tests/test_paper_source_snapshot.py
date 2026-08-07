from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from bridge.paper_source_snapshot import (
    UPSTREAM_PATCH_FILES,
    build_frozen_upstream_snapshot,
    verify_frozen_upstream_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_committed_patch_snapshots_match_live_upstream_changes() -> None:
    snapshot = build_frozen_upstream_snapshot(PROJECT_ROOT)

    assert set(snapshot) == set(UPSTREAM_PATCH_FILES)
    verify_frozen_upstream_snapshot(PROJECT_ROOT, snapshot)
    assert all(len(item["head"]) == 40 for item in snapshot.values())
    assert all(len(item["patch_sha256"]) == 64 for item in snapshot.values())


def test_upstream_snapshot_rejects_tampered_expected_identity() -> None:
    snapshot = build_frozen_upstream_snapshot(PROJECT_ROOT)
    tampered = deepcopy(snapshot)
    tampered["upstreams/gepa"]["patch_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="HEAD or tracked patch changed"):
        verify_frozen_upstream_snapshot(PROJECT_ROOT, tampered)
