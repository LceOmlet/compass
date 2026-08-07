from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridge.chartqa_protocol import (  # noqa: E402
    CHARTQA_DATASET_REVISION,
    CHARTQA_OWNER_COMMIT,
    EXACT_DUPLICATE_RULE_ID,
    EXPECTED_COMPOSITION,
    EXPECTED_SPLIT_COUNTS,
    LMMS_EVAL_COMMIT,
    SELECTION_RULE_ID,
    SELECTION_SALT,
    SPLIT_HYGIENE_RULE_ID,
    SOURCE_SPECS,
    TEST_PARQUET_SHA256,
    TEST_ROW_COUNT,
    TEST_TYPE_COUNTS,
    ChartQASourceSpec,
)


LMMS_OWNER_FILES = {
    "lmms_eval/tasks/chartqa/chartqa.yaml": (
        "lmms_owner/chartqa.yaml",
        "12f0a1329d31605b2e0e343a9e637d9962b45616de57318a5731467006f572d4",
    ),
    "lmms_eval/tasks/chartqa/utils.py": (
        "lmms_owner/utils.py",
        "5e336957fada294cc7cc493136e0177103efcf04e1d7bc7b4900fa7ab8c75e0e",
    ),
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(repo: Path, revision: str, owner_path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{owner_path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cannot read pinned owner artifact {revision}:{owner_path}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _require_commit(repo: Path, revision: str, owner: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{revision}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0 or result.stdout.strip() != revision:
        raise RuntimeError(
            f"cannot resolve pinned {owner} commit {revision} in {repo}"
        )


def _git_split_image_blobs(
    repo: Path,
    revision: str,
) -> dict[str, dict[str, str]]:
    """Read the owner's split/image-to-blob map without using the worktree."""

    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            revision,
            "--",
            "ChartQA Dataset/train/png",
            "ChartQA Dataset/val/png",
            "ChartQA Dataset/test/png",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "cannot enumerate pinned ChartQA image blobs: "
            + result.stderr.strip()
        )
    image_blobs: dict[str, dict[str, str]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    for line in result.stdout.splitlines():
        metadata, owner_path = line.split("\t", 1)
        _mode, object_type, blob_id = metadata.split()
        parts = Path(owner_path).parts
        if object_type != "blob" or len(parts) != 4 or parts[2] != "png":
            raise RuntimeError(f"unexpected ChartQA image tree entry: {line}")
        split = parts[1]
        imgname = parts[3]
        if split not in image_blobs or imgname in image_blobs[split]:
            raise RuntimeError(f"unexpected ChartQA image identity: {owner_path}")
        image_blobs[split][imgname] = blob_id
    if any(not entries for entries in image_blobs.values()):
        raise RuntimeError("pinned ChartQA owner has an empty image split")
    return image_blobs


def _canonical_identity(
    *,
    split: str,
    source: str,
    source_position: int,
    row: dict[str, Any],
) -> bytes:
    identity = {
        "owner_commit": CHARTQA_OWNER_COMMIT,
        "split": split,
        "source": source,
        "source_position": source_position,
        "imgname": row["imgname"],
        "question": row["query"],
    }
    return json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _ranked_rows(
    rows: list[dict[str, Any]],
    spec: ChartQASourceSpec,
    *,
    split_image_blobs: dict[str, dict[str, str]],
    cross_split_blob_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ranked: list[dict[str, Any]] = []
    seen_exact_records: set[tuple[str, str, str]] = set()
    hygiene_counts = {
        "excluded_cross_split_image_blob": 0,
        "excluded_later_exact_duplicate": 0,
    }
    for source_position, row in enumerate(rows):
        if set(row) != {"imgname", "query", "label"}:
            raise ValueError(
                f"unexpected owner fields in {spec.owner_path}:{source_position}: "
                f"{sorted(row)}"
            )
        if not all(isinstance(row[field], str) and row[field] for field in row):
            raise ValueError(
                f"owner row must contain non-empty text in "
                f"{spec.owner_path}:{source_position}"
            )
        imgname = row["imgname"]
        if Path(imgname).name != imgname or not imgname.lower().endswith(".png"):
            raise ValueError(
                f"unsafe ChartQA image name in {spec.owner_path}:{source_position}"
            )
        exact_record = (imgname, row["query"], row["label"])
        if exact_record in seen_exact_records:
            hygiene_counts["excluded_later_exact_duplicate"] += 1
            continue
        seen_exact_records.add(exact_record)
        try:
            image_blob_id = split_image_blobs[spec.split][imgname]
        except KeyError as error:
            raise RuntimeError(
                f"owner annotation references an unpinned image: "
                f"{spec.owner_path}:{source_position}:{imgname}"
            ) from error
        if image_blob_id in cross_split_blob_ids:
            hygiene_counts["excluded_cross_split_image_blob"] += 1
            continue
        identity = _canonical_identity(
            split=spec.split,
            source=spec.source,
            source_position=source_position,
            row=row,
        )
        rank = hashlib.sha256(
            SELECTION_SALT.encode("utf-8") + b"\0rank\0" + identity
        ).hexdigest()
        example_id = hashlib.sha256(
            SELECTION_SALT.encode("utf-8") + b"\0id\0" + identity
        ).hexdigest()
        ranked.append(
            {
                "rank": rank,
                "id": example_id,
                "split": spec.split,
                "source": spec.source,
                "source_position": source_position,
                "imgname": imgname,
                "question": row["query"],
                "answer": row["label"],
            }
        )
    ranked.sort(key=lambda item: (item["rank"], item["source_position"]))
    return ranked, hygiene_counts


def select_balanced_rows(
    owner_payloads: dict[tuple[str, str], bytes],
    split_image_blobs: dict[str, dict[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    blob_splits: dict[str, set[str]] = defaultdict(set)
    for split, images in split_image_blobs.items():
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unexpected ChartQA owner split: {split}")
        for blob_id in images.values():
            blob_splits[blob_id].add(split)
    cross_split_blob_ids = {
        blob_id for blob_id, splits in blob_splits.items() if len(splits) > 1
    }

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    source_hygiene: dict[str, dict[str, int]] = {}
    for spec in SOURCE_SPECS:
        payload = owner_payloads[(spec.split, spec.source)]
        if _sha256_bytes(payload) != spec.owner_sha256:
            raise RuntimeError(f"owner annotation checksum mismatch: {spec.owner_path}")
        rows = json.loads(payload)
        if not isinstance(rows, list) or len(rows) != spec.owner_count:
            raise RuntimeError(
                f"owner annotation count mismatch for {spec.owner_path}: "
                f"expected={spec.owner_count}, actual={len(rows) if isinstance(rows, list) else 'non-list'}"
            )
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"owner annotations must be objects: {spec.owner_path}")
        ranked, hygiene_counts = _ranked_rows(
            rows,
            spec,
            split_image_blobs=split_image_blobs,
            cross_split_blob_ids=cross_split_blob_ids,
        )
        if len(ranked) < spec.quota:
            raise RuntimeError(
                f"not enough ChartQA rows after frozen hygiene for {spec.owner_path}: "
                f"needed={spec.quota}, available={len(ranked)}"
            )
        selected[spec.split].extend(ranked[: spec.quota])
        source_hygiene[f"{spec.split}_{spec.source}"] = hygiene_counts

    for split, rows in selected.items():
        rows.sort(key=lambda item: (item["rank"], item["source"], item["source_position"]))
        composition = Counter(row["source"] for row in rows)
        if len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise RuntimeError(f"balanced ChartQA {split} count mismatch")
        if dict(composition) != EXPECTED_COMPOSITION[split]:
            raise RuntimeError(f"balanced ChartQA {split} composition mismatch")
    return selected, {
        "split_hygiene_rule_id": SPLIT_HYGIENE_RULE_ID,
        "exact_duplicate_rule_id": EXACT_DUPLICATE_RULE_ID,
        "cross_split_image_blob_count": len(cross_split_blob_ids),
        "source_exclusions": source_hygiene,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _verify_test_parquet(path: Path) -> None:
    if _sha256_file(path) != TEST_PARQUET_SHA256:
        raise RuntimeError(f"ChartQA test parquet checksum mismatch: {path}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to verify the ChartQA test parquet") from error
    parquet_file = parquet.ParquetFile(path)
    if parquet_file.metadata.num_rows != TEST_ROW_COUNT:
        raise RuntimeError(
            f"ChartQA test row count mismatch: {parquet_file.metadata.num_rows}"
        )
    required_columns = {"type", "question", "answer", "image"}
    if set(parquet_file.schema_arrow.names) != required_columns:
        raise RuntimeError(
            f"ChartQA test schema mismatch: {parquet_file.schema_arrow.names}"
        )
    type_counts = Counter(
        parquet.read_table(path, columns=["type"]).column("type").to_pylist()
    )
    if dict(type_counts) != TEST_TYPE_COUNTS:
        raise RuntimeError(f"ChartQA test type counts mismatch: {dict(type_counts)}")


def _file_entry(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "manifest.json.sha256"}:
            continue
        files[path.relative_to(root).as_posix()] = _file_entry(path)
    manifest["files"] = files
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "manifest.json.sha256").write_text(
        f"{_sha256_file(manifest_path)}  manifest.json\n",
        encoding="utf-8",
        newline="\n",
    )


def build_snapshot(
    *,
    chartqa_repo: Path,
    lmms_eval_repo: Path,
    test_parquet: Path,
    output_dir: Path,
) -> dict[str, Any]:
    chartqa_repo = chartqa_repo.resolve(strict=True)
    lmms_eval_repo = lmms_eval_repo.resolve(strict=True)
    test_parquet = test_parquet.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ChartQA snapshot: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=False)

    _require_commit(chartqa_repo, CHARTQA_OWNER_COMMIT, "vis-nlp/ChartQA")
    _require_commit(lmms_eval_repo, LMMS_EVAL_COMMIT, "LMMS-Eval")
    _verify_test_parquet(test_parquet)

    owner_payloads: dict[tuple[str, str], bytes] = {}
    source_integrity: dict[str, dict[str, Any]] = {}
    for spec in SOURCE_SPECS:
        payload = _git_bytes(chartqa_repo, CHARTQA_OWNER_COMMIT, spec.owner_path)
        owner_payloads[(spec.split, spec.source)] = payload
        output_relative = Path("owner") / Path(spec.owner_path).relative_to(
            "ChartQA Dataset"
        )
        output_path = staging / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        source_integrity[spec.owner_path] = {
            "count": spec.owner_count,
            "sha256": spec.owner_sha256,
        }

    split_image_blobs = _git_split_image_blobs(chartqa_repo, CHARTQA_OWNER_COMMIT)
    selected, hygiene_report = select_balanced_rows(
        owner_payloads,
        split_image_blobs,
    )
    selected_positions: dict[str, dict[str, list[int]]] = {}
    image_pairs: set[tuple[str, str]] = set()
    for split, rows in selected.items():
        inputs: list[dict[str, Any]] = []
        labels: list[dict[str, Any]] = []
        selected_positions[split] = {"human": [], "augmented": []}
        for split_position, row in enumerate(rows):
            image_relative = f"images/{split}/{row['imgname']}"
            inputs.append(
                {
                    "id": row["id"],
                    "split_position": split_position,
                    "source": row["source"],
                    "source_position": row["source_position"],
                    "question": row["question"],
                    "image": image_relative,
                }
            )
            labels.append(
                {
                    "id": row["id"],
                    "split_position": split_position,
                    "answer": row["answer"],
                }
            )
            selected_positions[split][row["source"]].append(row["source_position"])
            image_pairs.add((split, row["imgname"]))
        _write_jsonl(staging / "views" / f"{split}_lite_inputs.jsonl", inputs)
        _write_jsonl(staging / "views" / f"{split}_lite_labels.jsonl", labels)

    download_lines: list[str] = []
    for split, imgname in sorted(image_pairs):
        owner_path = f"ChartQA Dataset/{split}/png/{imgname}"
        payload = _git_bytes(chartqa_repo, CHARTQA_OWNER_COMMIT, owner_path)
        if not payload.startswith(PNG_SIGNATURE):
            raise RuntimeError(f"pinned ChartQA image is not PNG: {owner_path}")
        image_path = staging / "images" / split / imgname
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(payload)
        quoted_owner_path = quote(owner_path, safe="/")
        download_lines.append(
            "https://raw.githubusercontent.com/vis-nlp/ChartQA/"
            f"{CHARTQA_OWNER_COMMIT}/{quoted_owner_path}\t"
            f"images/{split}/{imgname}\n"
        )
    (staging / "download_images.tsv").write_text(
        "".join(download_lines),
        encoding="utf-8",
        newline="\n",
    )

    for owner_path, (output_relative, expected_sha256) in LMMS_OWNER_FILES.items():
        payload = _git_bytes(lmms_eval_repo, LMMS_EVAL_COMMIT, owner_path)
        if _sha256_bytes(payload) != expected_sha256:
            raise RuntimeError(f"LMMS owner checksum mismatch: {owner_path}")
        output_path = staging / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)

    local_test = staging / "lmms_test" / "test-00000-of-00001.parquet"
    local_test.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_parquet, local_test)
    _verify_test_parquet(local_test)

    manifest = {
        "benchmark": "ChartQA",
        "status": "frozen",
        "build": {
            "builder": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "builder_sha256": _sha256_file(Path(__file__)),
            "selection_rule_id": SELECTION_RULE_ID,
            "selection_salt": SELECTION_SALT,
        },
        "optimization_owner": {
            "repo": "https://github.com/vis-nlp/ChartQA.git",
            "commit": CHARTQA_OWNER_COMMIT,
        },
        "owner_annotation_counts": {
            f"{spec.split}_{spec.source}": spec.owner_count for spec in SOURCE_SPECS
        },
        "source_integrity": source_integrity,
        "hygiene": hygiene_report,
        "selection": {
            f"{split}_lite": {
                "count": EXPECTED_SPLIT_COUNTS[split],
                "composition": EXPECTED_COMPOSITION[split],
                "rule_id": SELECTION_RULE_ID,
                "selected_owner_positions": selected_positions[split],
                "source_split": split,
                "unique_images": len(
                    {row["imgname"] for row in selected[split]}
                ),
            }
            for split in ("train", "val")
        },
        "image_source": {
            "count": len(image_pairs),
            "download_list": "download_images.tsv",
            "all_png_signature_valid": True,
            "url_template": (
                "https://raw.githubusercontent.com/vis-nlp/ChartQA/"
                f"{CHARTQA_OWNER_COMMIT}/ChartQA%20Dataset/{{split}}/png/{{imgname}}"
            ),
        },
        "formal_test": {
            "dataset_repo": "https://huggingface.co/datasets/lmms-lab/ChartQA",
            "dataset_revision": CHARTQA_DATASET_REVISION,
            "parquet": "lmms_test/test-00000-of-00001.parquet",
            "parquet_sha256": TEST_PARQUET_SHA256,
            "row_count": TEST_ROW_COUNT,
            "type_counts": TEST_TYPE_COUNTS,
            "selection_policy": "full owner test in parquet order after candidate freeze",
            "lmms_eval_repo": "https://github.com/EvolvingLMMs-Lab/lmms-eval.git",
            "lmms_eval_commit": LMMS_EVAL_COMMIT,
            "task_yaml": "lmms_owner/chartqa.yaml",
            "metric_source": "lmms_owner/utils.py",
            "metric_policy": (
                "reuse chartqa_process_results and relaxed_correctness; no local metric"
            ),
        },
        "model_visible_contract": {
            "fields": ["image", "question"],
            "gold_files": "views/*_labels.jsonl only; never model input",
            "metadata_not_model_visible": [
                "id",
                "split_position",
                "source",
                "source_position",
            ],
        },
    }
    _write_manifest(staging, manifest)
    staging.replace(output_dir)
    return {
        "output_dir": str(output_dir),
        "manifest_sha256": _sha256_file(output_dir / "manifest.json"),
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "composition": EXPECTED_COMPOSITION,
        "unique_images": {
            split: len({row["imgname"] for row in rows})
            for split, rows in selected.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chartqa-repo",
        type=Path,
        default=PROJECT_ROOT / "upstreams" / "chartqa",
    )
    parser.add_argument(
        "--lmms-eval-repo",
        type=Path,
        default=(
            PROJECT_ROOT
            / "upstreams"
            / "skill-factory"
            / "upstream"
            / "lmms-eval"
        ),
    )
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_snapshot(
        chartqa_repo=args.chartqa_repo,
        lmms_eval_repo=args.lmms_eval_repo,
        test_parquet=args.test_parquet,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
