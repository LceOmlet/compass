FROM node@sha256:f32b81066cde10a75dbac96646099533316d94bac4150c55da1636e1f0ffdc46 AS pi-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

ENV SHELL=/bin/bash \
    npm_config_script_shell=/bin/bash

WORKDIR /opt/pi-mono
COPY upstreams/pi-mono/ /opt/pi-mono/
RUN --mount=type=cache,target=/root/.npm \
    HUSKY=0 npm ci --foreground-scripts --loglevel=verbose
RUN cd packages/tui \
    && npm run build \
    && cd ../ai \
    && npx --no-install tsgo -p tsconfig.build.json \
    && cd ../agent \
    && npm run build \
    && cd ../coding-agent \
    && npm run build


FROM node@sha256:f32b81066cde10a75dbac96646099533316d94bac4150c55da1636e1f0ffdc46

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        findutils \
        grep \
        python3 \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

ENV SHELL=/bin/bash \
    npm_config_script_shell=/bin/bash \
    PYTHONPATH=/opt/dci-agent-lite/src

COPY --from=pi-builder /opt/pi-mono /opt/pi-mono
COPY upstreams/dci-agent-lite/ /opt/dci-agent-lite/
COPY docker/dci-sandbox/entrypoint.sh /usr/local/bin/compass-dci-agent-lite

RUN chmod 0555 /usr/local/bin/compass-dci-agent-lite \
    && mkdir -p /agent /agent-seed /corpus /input /output

ENTRYPOINT ["/usr/local/bin/compass-dci-agent-lite"]
