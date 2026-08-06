#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/home /agent

for config_name in models.json settings.json; do
    seed_path="/agent-seed/${config_name}"
    if [[ -f "${seed_path}" ]]; then
        cp -- "${seed_path}" "/agent/${config_name}"
    fi
done

export HOME=/tmp/home
export PI_CODING_AGENT_DIR=/agent
export SHELL=/bin/bash

exec python3 -m dci.benchmark.pi_rpc_runner "$@"
