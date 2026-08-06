#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
matrix_dir=${1:-"$repo_root/experiments/mechanism/generated/hitab_chartqa_clutrr_mechanism_v1"}
lane_filter=${2:-all}
queue_file="$matrix_dir/launch_queue_all.jsonl"
manifest_file="$matrix_dir/matrix_manifest.json"
control_python=${COMPASS_CONTROL_PYTHON:-"$repo_root/.venv/Scripts/python.exe"}

case "$lane_filter" in
    all|hitab|chartqa|clutrr) ;;
    *)
        echo "lane must be one of: all, hitab, chartqa, clutrr" >&2
        exit 2
        ;;
esac

if [ "$lane_filter" != "clutrr" ] && [ -z "${COMPASS_LITELLM_PROXY_KEY:-}" ]; then
    echo "COMPASS_LITELLM_PROXY_KEY must be explicitly set to a non-empty local client token" >&2
    exit 2
fi

if { [ "$lane_filter" = "all" ] || [ "$lane_filter" = "clutrr" ]; } && \
        [ -z "${COMPASS_VLLM_API_KEY:-}" ]; then
    COMPASS_VLLM_API_KEY=$(
        ssh \
            -p "${COMPASS_VLLM_SSH_PORT:-31906}" \
            -o BatchMode=yes \
            -o ConnectTimeout=8 \
            root@ssh.v5000-prod-gw.nhss.zhejianglab.com \
            'cat /root/.config/vllm-qwen3-8b/api_key'
    )
    export COMPASS_VLLM_API_KEY
fi
if { [ "$lane_filter" = "all" ] || [ "$lane_filter" = "clutrr" ]; } && \
        [ -z "${COMPASS_VLLM_API_KEY:-}" ]; then
    echo "COMPASS_VLLM_API_KEY is unavailable" >&2
    exit 2
fi

for required in "$manifest_file" "$queue_file"; do
    if [ ! -f "$required" ]; then
        echo "required frozen matrix artifact is missing: $required" >&2
        exit 2
    fi
done
if [ ! -x "$control_python" ]; then
    echo "fixed queue-validation runtime is unavailable: $control_python" >&2
    exit 2
fi
for command_name in xargs curl; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "required command is unavailable: $command_name" >&2
        exit 2
    fi
done

if [ "$lane_filter" = "all" ] || [ "$lane_filter" = "hitab" ]; then
    curl --fail --silent --show-error --max-time 8 \
        http://127.0.0.1:40038/v1/models >/dev/null
fi
if [ "$lane_filter" = "all" ] || [ "$lane_filter" = "chartqa" ]; then
    curl --fail --silent --show-error --max-time 8 \
        http://127.0.0.1:40039/v1/models >/dev/null
fi
if [ "$lane_filter" = "all" ] || [ "$lane_filter" = "clutrr" ]; then
    printf 'header = "Authorization: Bearer %s"\n' "$COMPASS_VLLM_API_KEY" | \
        curl --config - --fail --silent --show-error --max-time 8 \
            http://127.0.0.1:18000/v1/models >/dev/null
fi

export HITAB_PREPARED_ROOT=${HITAB_PREPARED_ROOT:-F:/compass-hitab-local/datasets/hitab_d179602662b490249baf068a76fbe4137029126e}
export CHARTQA_PREPARED_ROOT=${CHARTQA_PREPARED_ROOT:-F:/compass-chartqa-local/datasets/chartqa_044eabfc306abfe9340c5741f0093aefc5973d06}
export CHARTQA_ROOT=${CHARTQA_ROOT:-"$repo_root/upstreams/chartqa"}
export SKILL_FACTORY_ROOT=${SKILL_FACTORY_ROOT:-"$repo_root/upstreams/skill-factory"}
export CLUTRR_HF_DATA_ROOT=${CLUTRR_HF_DATA_ROOT:-"$repo_root/upstreams/clutrr-hf-data"}
export CLUTRR_BASELINE_ROOT=${CLUTRR_BASELINE_ROOT:-"$repo_root/upstreams/clutrr-baselines"}
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

emit_lane() {
    lane=$1
    "$control_python" - "$manifest_file" "$queue_file" "$lane" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1]).resolve()
queue_path = pathlib.Path(sys.argv[2]).resolve()
lane = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
queue_meta = manifest.get("launch_queues", {}).get("all", {})
if queue_path.name != queue_meta.get("file"):
    raise SystemExit("frozen all-queue filename does not match the manifest")
payload = queue_path.read_bytes()
if hashlib.sha256(payload).hexdigest() != queue_meta.get("sha256"):
    raise SystemExit("frozen all-queue hash mismatch")
rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
if len(rows) != queue_meta.get("count") or len(rows) != manifest.get("run_count"):
    raise SystemExit("frozen all-queue count mismatch")

run_meta = {entry["slug"]: entry for entry in manifest.get("runs", [])}
project_root = pathlib.Path(rows[0]["cwd"]).resolve() if rows else None
for relative, expected in manifest.get("source_snapshot", {}).get(
    "file_sha256", {}
).items():
    source = (project_root / relative).resolve()
    try:
        source.relative_to(project_root)
    except ValueError as error:
        raise SystemExit(f"source snapshot path escapes project root: {relative}") from error
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"source snapshot hash mismatch: {relative}")

def row_lane(task_id: str) -> str:
    if task_id == "hitab":
        return "hitab"
    if task_id == "chartqa":
        return "chartqa"
    if task_id.startswith("clutrr_"):
        return "clutrr"
    raise SystemExit(f"unknown task in frozen queue: {task_id}")

selected = []
for row in rows:
    slug = row["slug"]
    meta = run_meta.get(slug)
    if meta is None or meta.get("config_sha256") != row.get("config_sha256"):
        raise SystemExit(f"manifest/queue config hash mismatch: {slug}")
    config_path = (manifest_path.parent / row["config_file"]).resolve()
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != row["config_sha256"]:
        raise SystemExit(f"config content hash mismatch: {slug}")
    argv = row.get("argv")
    if not isinstance(argv, list) or len(argv) != 4 or argv[2] != "--config":
        raise SystemExit(f"unsupported frozen argv: {slug}")
    if pathlib.Path(argv[3]).resolve() != config_path:
        raise SystemExit(f"queue config path mismatch: {slug}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("run_dir") != row.get("run_dir")
        or config.get("cache_dir") != row.get("cache_dir")
        or config.get("task_id") != row.get("task_id")
        or config.get("model", {}).get("api_base") != row.get("api_base")
        or config.get("model", {}).get("api_key_env") != row.get("api_key_env")
    ):
        raise SystemExit(f"queue/config semantic mismatch: {slug}")
    required_env = row.get("required_env")
    if not isinstance(required_env, list) or not all(
        isinstance(name, str) and name and name.replace("_", "A").isalnum()
        for name in required_env
    ):
        raise SystemExit(f"invalid required_env representation: {slug}")
    pythonpath = row.get("pythonpath")
    if not isinstance(pythonpath, list) or not pythonpath:
        raise SystemExit(f"invalid pythonpath representation: {slug}")
    if row_lane(row["task_id"]) == lane:
        selected.append(row)

expected = sum(row_lane(row["task_id"]) == lane for row in rows)
if expected == 0 or len(selected) != expected:
    raise SystemExit(f"frozen lane count mismatch for {lane}: {len(selected)}")

fields = (
    "slug",
    "cwd",
    "stdout_log",
    "stderr_log",
    "pid_file",
    "run_dir",
    "cache_dir",
)
for row in selected:
    values = [str(row[name]) for name in fields]
    values.extend(str(value) for value in row["argv"])
    values.append(";".join(str(value) for value in row["pythonpath"]))
    values.append(",".join(row["required_env"]))
    if any("\0" in value for value in values):
        raise SystemExit(f"NUL in queue record: {row['slug']}")
    sys.stdout.buffer.write("\0".join(values).encode("utf-8") + b"\0")
PY
}

launch_record='set -euo pipefail
slug=$1
cwd=$2
stdout_log=$3
stderr_log=$4
pid_file=$5
run_dir=$6
cache_dir=$7
python_executable=$8
runner=$9
config_flag=${10}
config_path=${11}
pythonpath=${12}
required_env=${13}
for target in "$stdout_log" "$stderr_log" "$pid_file" "$run_dir" "$cache_dir"; do
    if [ -e "$target" ]; then
        echo "$slug: refusing existing target: $target" >&2
        exit 73
    fi
done
if [ ! -x "$python_executable" ] || [ ! -f "$runner" ] || [ "$config_flag" != "--config" ]; then
    echo "$slug: frozen runtime or argv is unavailable" >&2
    exit 2
fi
old_ifs=$IFS
IFS=,
for name in $required_env; do
    if [ -z "${!name:-}" ]; then
        echo "$slug: required environment variable is empty: $name" >&2
        exit 2
    fi
done
IFS=$old_ifs
mkdir -p "$(dirname "$stdout_log")" "$(dirname "$run_dir")" "$(dirname "$cache_dir")"
set -o noclobber
if ! printf "%s\n" "$BASHPID" >"$pid_file"; then
    echo "$slug: another launcher already claimed the frozen row" >&2
    exit 73
fi
if ! exec >"$stdout_log"; then
    echo "$slug: stdout log could not be claimed" >&2
    exit 73
fi
if ! exec 2>"$stderr_log"; then
    echo "$slug: stderr log could not be claimed" >&2
    exit 73
fi
set +o noclobber
printf "%s START %s\n" "$(date -Iseconds)" "$slug"
cd "$cwd"
exec env PYTHONPATH="$pythonpath" "$python_executable" "$runner" "$config_flag" "$config_path"'

run_lane() {
    lane=$1
    parallel=$2
    emit_lane "$lane" | xargs -0 -r -n 13 -P "$parallel" \
        bash -c "$launch_record" _
}

failures=0
if [ "$lane_filter" = "all" ]; then
    run_lane hitab 1 &
    hitab_pid=$!
    run_lane chartqa 1 &
    chartqa_pid=$!
    run_lane clutrr 4 &
    clutrr_pid=$!
    for lane_pid in "$hitab_pid" "$chartqa_pid" "$clutrr_pid"; do
        if ! wait "$lane_pid"; then
            failures=$((failures + 1))
        fi
    done
else
    parallel=1
    if [ "$lane_filter" = "clutrr" ]; then
        parallel=4
    fi
    if ! run_lane "$lane_filter" "$parallel"; then
        failures=1
    fi
fi
if [ "$failures" -ne 0 ]; then
    echo "mechanism matrix ended with $failures failed lane(s)" >&2
    exit 1
fi
printf "%s COMPLETE mechanism matrix lane=%s\n" \
    "$(date -Iseconds)" "$lane_filter"
