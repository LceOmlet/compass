#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
generator="$repo_root/experiments/paper/generate_formal_experiment_matrix.py"
control_python=${PAPER_MATRIX_CONTROL_PYTHON:-"$repo_root/.venv/Scripts/python.exe"}

usage() {
    cat >&2 <<'EOF'
usage: run-paper-formal-matrix.sh MANIFEST PHASE CAPACITY_GROUP [CELL_ID ...]

The manifest and its queue must already be frozen.  This launcher never reads
or creates credentials, changes worker counts, retries work, monitors resource
usage, or modifies an in-flight run.  It only claims create-only output paths
and directly execs each bound Python runner.
EOF
    exit 2
}

[ "$#" -ge 3 ] || usage
manifest=$1
phase=$2
capacity_group=$3
shift 3

if [ "$phase" != preflight ] && [ "$phase" != formal ]; then
    usage
fi
if [ ! -f "$manifest" ] || [ ! -f "$generator" ]; then
    echo "frozen manifest or queue generator is unavailable" >&2
    exit 2
fi
if [ ! -x "$control_python" ]; then
    echo "fixed queue-validation Python is unavailable: $control_python" >&2
    exit 2
fi
# The control process imports only this repository's frozen orchestration
# modules.  Bound experiment runners replace PYTHONPATH with their own exact
# dependency list below.
export PYTHONPATH="$repo_root"
for command_name in bash xargs; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "required Bash tool is unavailable: $command_name" >&2
        exit 2
    fi
done

parallel=$(
    "$control_python" "$generator" capacity-limit \
        --manifest "$manifest" \
        --capacity-group "$capacity_group"
)
case "$parallel" in
    ''|*[!0-9]*|0)
        echo "invalid frozen capacity limit: $parallel" >&2
        exit 2
        ;;
esac

emit_args=(
    "$control_python" "$generator" emit
    --manifest "$manifest"
    --phase "$phase"
    --capacity-group "$capacity_group"
)
for cell_id in "$@"; do
    emit_args+=(--cell-id "$cell_id")
done

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
        echo "$slug: refusing existing create-only target: $target" >&2
        exit 73
    fi
done
if [ ! -x "$python_executable" ] || [ ! -f "$runner" ]; then
    echo "$slug: frozen Python runtime or runner is unavailable" >&2
    exit 2
fi
if [ "$config_flag" != "--config" ] || [ ! -f "$config_path" ]; then
    echo "$slug: frozen config binding is unavailable" >&2
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
printf "%s\n" "$BASHPID" >"$pid_file"
exec >"$stdout_log" 2>"$stderr_log"
set +o noclobber
printf "%s START %s\n" "$(date -Iseconds)" "$slug"
cd "$cwd"
exec env \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH="$pythonpath" \
    "$python_executable" "$runner" "$config_flag" "$config_path"'

"${emit_args[@]}" | xargs -0 -r -n 13 -P "$parallel" \
    bash -c "$launch_record" _
