"""Frozen study protocol for the reproducible ChartQA optimization view."""

from __future__ import annotations

from dataclasses import dataclass


CHARTQA_OWNER_COMMIT = "044eabfc306abfe9340c5741f0093aefc5973d06"
CHARTQA_DATASET_REVISION = "9e63b7df1592a1c2158e735cc1725454aef0d6d9"
LMMS_EVAL_COMMIT = "cb45ac4d4a667ea5ef89c7a148bff69b3489b981"
SELECTION_RULE_ID = "sha256_rank_after_owner_blob_hygiene_v2"
SELECTION_SALT = "compass-chartqa-balanced-optimization-view-v1"
SPLIT_HYGIENE_RULE_ID = "exclude_cross_owner_split_git_blob_v1"
EXACT_DUPLICATE_RULE_ID = "retain_earliest_exact_owner_record_per_source_v1"
TEST_PARQUET_SHA256 = (
    "165263505f2998aba65d819b44be832edecd92d676fee2c030645f784cd55d06"
)
TEST_ROW_COUNT = 2500
TEST_TYPE_COUNTS = {"human_test": 1250, "augmented_test": 1250}
FROZEN_SNAPSHOT_MANIFEST_SHA256 = (
    "ace63d0a25e587898cbf281764fe5d0aedd332585f1841efeea3df57407f326a"
)
FROZEN_SNAPSHOT_BUILDER_SHA256 = (
    "0342107da25105e9f7a49e1c7c0ebf43063f6a958633d21a2f06b0fe6f74a168"
)
FROZEN_SPLIT_FINGERPRINTS = {
    "train": "410792d8bc493b0e4b97d55d6c2dffca34e33f647e2aeb587036bc30a06301de",
    "validation": "3637d073a0fddd8e73d6d841dd6b9687baaef955ae694c939702bcf1f722d907",
    "test": "341561aa8e7ad27c769599e007fdacf96505ad79ec4df9f6a5374ab77bdebbed",
}
FROZEN_VIEW_SHA256 = {
    "views/train_lite_inputs.jsonl": (
        "14b503a5d0545a44f4a0ce29741262e3bf8c81264b9f0f5313a7e72a4cc9d5e3"
    ),
    "views/train_lite_labels.jsonl": (
        "bbe58facb6a7244c8420d732cabc4fe40bc6b370eac2e8b7fd7549dddf405b6d"
    ),
    "views/val_lite_inputs.jsonl": (
        "9e267711f4a392ff59212a9301a8ed85f8bb42c2dbb548278623fae8888931c7"
    ),
    "views/val_lite_labels.jsonl": (
        "548d8edf3596936c876c3e82623014d1b1bbfa4d41740f87cf36be1592fd4774"
    ),
}


@dataclass(frozen=True, slots=True)
class ChartQASourceSpec:
    split: str
    source: str
    owner_path: str
    owner_count: int
    owner_sha256: str
    quota: int


SOURCE_SPECS = (
    ChartQASourceSpec(
        split="train",
        source="human",
        owner_path="ChartQA Dataset/train/train_human.json",
        owner_count=7398,
        owner_sha256=(
            "4eaaa03e406dbbbe43caff925dbec9930b87fadba960f1c2ab30a5def287c384"
        ),
        quota=75,
    ),
    ChartQASourceSpec(
        split="train",
        source="augmented",
        owner_path="ChartQA Dataset/train/train_augmented.json",
        owner_count=20901,
        owner_sha256=(
            "77342d8527d2a194a6011ce3f4389ea32d9c50cfe60b10c74dbe1cd2e49724da"
        ),
        quota=75,
    ),
    ChartQASourceSpec(
        split="val",
        source="human",
        owner_path="ChartQA Dataset/val/val_human.json",
        owner_count=960,
        owner_sha256=(
            "e297ce5b38b4ab79bd0e22a93eb19a65cb89f9bd07ce886cb78f2e333fb0ba14"
        ),
        quota=150,
    ),
    ChartQASourceSpec(
        split="val",
        source="augmented",
        owner_path="ChartQA Dataset/val/val_augmented.json",
        owner_count=960,
        owner_sha256=(
            "cafc675781fba27282b78cab40eb49b77a5bf890ef1e89294bb67c994bdfde47"
        ),
        quota=150,
    ),
)


EXPECTED_COMPOSITION = {
    "train": {"human": 75, "augmented": 75},
    "val": {"human": 150, "augmented": 150},
}
EXPECTED_SPLIT_COUNTS = {
    "train": 150,
    "val": 300,
    "test": TEST_ROW_COUNT,
}
