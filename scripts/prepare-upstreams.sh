#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export GIT_LFS_SKIP_SMUDGE=1

git -C "$repo_root" submodule update --init --recursive

apply_source() {
    source_path=$1
    expected_commit=$2
    patch_path=${3-}

    actual_commit=$(git -C "$repo_root/$source_path" rev-parse HEAD)
    if [ "$actual_commit" != "$expected_commit" ]; then
        echo "$source_path is at $actual_commit; expected $expected_commit" >&2
        exit 1
    fi

    if [ -z "$patch_path" ]; then
        return
    fi

    if git -C "$repo_root/$source_path" apply --reverse --check "$repo_root/$patch_path" 2>/dev/null; then
        echo "$source_path: patch already applied"
    elif git -C "$repo_root/$source_path" apply --check "$repo_root/$patch_path"; then
        git -C "$repo_root/$source_path" apply --index "$repo_root/$patch_path"
        echo "$source_path: patch applied"
    else
        echo "$source_path: patch conflicts with the working tree" >&2
        exit 1
    fi
}

apply_source upstreams/gepa 665cbc368313b7223f2b0638d2ef47da7a130dad patches/gepa-working-tree.patch
apply_source upstreams/dspy 96bae53d458d300b2cab49a5ddf30087498df952 patches/dspy-working-tree.patch
apply_source upstreams/gepa-artifact cbefbc1aa0f43dd39874ec4bf42211365dbda42e patches/gepa-artifact-working-tree.patch
apply_source upstreams/flashtrace 9935467b628bbd7c7083bb84389271e16a4b1740
apply_source upstreams/dspy-gepa 62dc3b634d7dc0c4889abcf905cb4c391ea6b396
apply_source upstreams/dci-agent-lite 271f37e71f053bf0c99c05ce6d2fb53b841d922e
apply_source upstreams/pi-mono a6be5eb4cce278de31ac05792af3dfc0883215dc
apply_source upstreams/hitab d179602662b490249baf068a76fbe4137029126e
apply_source upstreams/chartqa 044eabfc306abfe9340c5741f0093aefc5973d06
apply_source upstreams/clutrr d045fae289d3746503677ceed7631c999202501e
apply_source upstreams/clutrr-hf-data e5b496941e91abb7c319d2618a3ce96752bc4ab7
apply_source upstreams/clutrr-baselines 303ed9a48f82a59b4eb34ac5bd5866f0d82c5552
apply_source upstreams/skill-factory e23f82f9c7ed2eacfb9124b92143358a9953263c
apply_source upstreams/skill-factory/upstream/lmms-eval cb45ac4d4a667ea5ef89c7a148bff69b3489b981

echo "Pinned upstream sources are ready."
