# DCI filesystem isolation

This image runs the pinned upstream DCI-Agent-Lite and Pi process chain inside a
Linux container. The image build, container entrypoint, and Pi tool shell all use
`/bin/bash`; no `cmd.exe` or PowerShell entrypoint is present.

Build the image from the repository root:

```bash
docker build \
  --progress=plain \
  --file docker/dci-sandbox/Dockerfile \
  --tag compass-dci-sandbox:local \
  .

docker image inspect compass-dci-sandbox:local --format '{{.Id}}'
```

Use the returned immutable `sha256:...` ID in
`optimizer.dci.runner_command`:

```json
[
  "python",
  "-m",
  "bridge.dci_docker_isolation",
  "--image",
  "sha256:REPLACE_WITH_THE_IMAGE_ID",
  "--pass-env",
  "OPENAI_API_KEY",
  "--pass-env",
  "OPENAI_BASE_URL",
  "--"
]
```

The existing DCI adapter appends the official runner arguments after `--`. The
launcher mounts only the exact corpus snapshot and system prompt read-only, a
unique empty artifact directory read-write, and fresh `/tmp` and `/agent` tmpfs
filesystems. It does not mount the repository, host home, held-out/test data,
other runs, or the Docker socket. Linux capabilities are dropped and privilege
escalation is disabled. Only environment variable names explicitly listed with
`--pass-env` are forwarded.

If present, only `models.json` and `settings.json` are copied from the configured
agent directory into the fresh `/agent` tmpfs. Keep credentials out of those
files and pass them through explicit environment names instead.

This is a filesystem and process boundary. It deliberately does not duplicate
the upstream agent loop, Pi tools, retries, parsing, or scheduling.
