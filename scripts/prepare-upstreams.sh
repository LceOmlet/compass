#!/usr/bin/env sh
set -eu

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
        git -C "$repo_root/$source_path" apply "$repo_root/$patch_path"
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

echo "Pinned upstream sources are ready."
