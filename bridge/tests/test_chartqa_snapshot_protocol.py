from __future__ import annotations

from collections import Counter
from pathlib import Path

from bridge.chartqa_protocol import (
    CHARTQA_OWNER_COMMIT,
    EXPECTED_COMPOSITION,
    EXPECTED_SPLIT_COUNTS,
    SOURCE_SPECS,
)
from experiments.paper.prepare_chartqa_balanced_snapshot import (
    _canonical_identity,
    _git_bytes,
    _git_split_image_blobs,
    select_balanced_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHARTQA_REPO = PROJECT_ROOT / "upstreams" / "chartqa"


def _owner_payloads() -> dict[tuple[str, str], bytes]:
    return {
        (spec.split, spec.source): _git_bytes(
            CHARTQA_REPO,
            CHARTQA_OWNER_COMMIT,
            spec.owner_path,
        )
        for spec in SOURCE_SPECS
    }


def test_balanced_selection_is_deterministic_on_the_pinned_owner_data() -> None:
    payloads = _owner_payloads()

    image_blobs = _git_split_image_blobs(CHARTQA_REPO, CHARTQA_OWNER_COMMIT)
    first, first_hygiene = select_balanced_rows(payloads, image_blobs)
    second, second_hygiene = select_balanced_rows(payloads, image_blobs)

    assert first == second
    assert first_hygiene == second_hygiene
    assert first_hygiene["cross_split_image_blob_count"] > 0
    for split, rows in first.items():
        assert len(rows) == EXPECTED_SPLIT_COUNTS[split]
        assert Counter(row["source"] for row in rows) == Counter(
            EXPECTED_COMPOSITION[split]
        )
        assert len({row["id"] for row in rows}) == len(rows)
        assert all(row["split"] == split for row in rows)

    blob_splits: dict[str, set[str]] = {}
    for split, entries in image_blobs.items():
        for blob_id in entries.values():
            blob_splits.setdefault(blob_id, set()).add(split)
    contaminated = {
        blob_id for blob_id, splits in blob_splits.items() if len(splits) > 1
    }
    assert all(
        image_blobs[split][row["imgname"]] not in contaminated
        for split, rows in first.items()
        for row in rows
    )
    for rows in first.values():
        exact_records = {
            (row["imgname"], row["question"], row["answer"]) for row in rows
        }
        assert len(exact_records) == len(rows)


def test_selection_identity_does_not_depend_on_the_gold_answer() -> None:
    common = {
        "imgname": "chart.png",
        "query": "What value is shown?",
    }

    left = _canonical_identity(
        split="train",
        source="human",
        source_position=3,
        row={**common, "label": "1"},
    )
    right = _canonical_identity(
        split="train",
        source="human",
        source_position=3,
        row={**common, "label": "999"},
    )

    assert left == right
