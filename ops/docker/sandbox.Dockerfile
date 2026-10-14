# AutoTrade Agent sandbox image (docs/environment-design.md 3.1/3.3).
# Build (context is the repo root so the trusted runtime modules can be copied in):
#   docker build --network=host --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
#     --build-arg ALL_PROXY --build-arg NO_PROXY \
#     -t autotrade-sandbox:latest -f ops/docker/sandbox.Dockerfile .
# Package and release downloads default to mirrors. Proxy build args without an
# explicit value forward the current process environment without baking proxy
# credentials into the image; --network=host is required for loopback proxies.
# Project-level build identity is assigned after a successful offline smoke
# test by the host lifecycle; this build consumes the declared Python tag.
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com
ARG DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian
ARG DEBIAN_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security
ARG DUCKDB_CLI_URL=https://ghfast.top/https://github.com/duckdb/duckdb/releases/download/v1.1.3/duckdb_cli-linux-amd64.zip

# The strategy container runs read-only with a /tmp tmpfs, so every cache the
# interpreter and its libraries may touch is redirected there. PYTHONPATH makes
# the baked trusted strategy runtime importable as `autotrade.environment.*`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/autotrade \
    XDG_CACHE_HOME=/tmp/cache \
    PIP_CACHE_DIR=/tmp/cache/pip \
    MPLCONFIGDIR=/tmp/cache/mpl

# Pre-bake the C/C++/Fortran build toolchain so source-only packages build in
# this image. Without this, the base python:3.11-slim has no compiler and such
# installs fail at build time (the root cause of an early GNN-transfer run's
# image-build error).
# No Debian python3-dev: source builds compile against the base image's own
# /usr/local/include/python3.11 headers; the distro package would only drag in
# an unusable second interpreter's (3.13) headers and runtime.
RUN sed -i \
        -e "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
        -e "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        npm \
        ripgrep \
        build-essential \
        g++ \
        gfortran \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Complete, pinned research/DL stack. Data layer pins are the environment's
# byte-format contract (parquet caches) and stay put; the DL family follows
# "widely adopted, relatively recent": torch 2.10.0 is the last and most
# mature release of the CUDA 12.8 line, and the boosting picks are the current
# stable releases the ecosystem has settled on. The stack is what a formal
# strategy may import (strategy_loader.ALLOWED_MODULES) plus the data tools the
# Agent's shell uses; torch wheels carry their own CUDA runtime, so the image
# needs no CUDA toolkit and is not tied to one GPU architecture.
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} \
        pandas==2.2.3 \
        numpy==2.1.3 \
        pyarrow==18.1.0 \
        duckdb==1.1.3 \
        scipy==1.17.1 \
        scikit-learn==1.5.2 \
        statsmodels==0.14.4 \
        torch==2.10.0 \
        torchvision==0.25.0 \
        safetensors==0.8.0 \
        lightgbm==4.7.0 \
        xgboost==3.2.0

# DuckDB CLI binary, pinned to the Python package version. The Agent's data-probe
# guidance uses `duckdb -c "..."`; without the CLI it fails with exit 127 and the
# Agent wastes turns falling back. (curl is already installed; release zip extracted
# with the bundled Python to avoid an extra apt dependency. Host is x86_64.)
RUN curl -fL --retry 8 --retry-all-errors --retry-delay 3 --connect-timeout 30 --max-time 600 \
        "${DUCKDB_CLI_URL}" \
        -o /tmp/duckdb_cli.zip \
    && python -c "import zipfile; zipfile.ZipFile('/tmp/duckdb_cli.zip').extractall('/usr/local/bin')" \
    && chmod +x /usr/local/bin/duckdb \
    && rm /tmp/duckdb_cli.zip \
    && duckdb -c "select 1"

# Trusted host-side runtime baked in: the strategy worker, the loader and the
# strategy context, loaded as `python -m autotrade.environment.strategy_worker`
# in the strategy container. Standard-library plus the pinned data stack only;
# the Broker stays on the host, so the worker never needs broker_core. The
# formal strategy allowlist (strategy_loader.ALLOWED_MODULES) names numpy,
# pandas, scipy, scikit-learn, LightGBM, XGBoost, statsmodels and torch from
# the pip layer above; the strategy container gets --gpus only when the
# experiment asked for a device (executor.DockerStrategyExecutor), else
# CPU-only.
WORKDIR /opt/autotrade
COPY src/autotrade/__init__.py /opt/autotrade/autotrade/__init__.py
COPY src/autotrade/environment/__init__.py /opt/autotrade/autotrade/environment/__init__.py
COPY src/autotrade/environment/strategy.py /opt/autotrade/autotrade/environment/strategy.py
COPY src/autotrade/environment/strategy_loader.py /opt/autotrade/autotrade/environment/strategy_loader.py
COPY src/autotrade/environment/strategy_worker.py /opt/autotrade/autotrade/environment/strategy_worker.py
COPY src/autotrade/environment/contract_fingerprint.py /opt/autotrade/autotrade/environment/contract_fingerprint.py
COPY ops/docker/pyrightconfig.json /opt/autotrade/pyrightconfig.json

# Build-time fingerprint of the full Agent-facing strategy contract: the three
# modules baked above plus the output template README that states the same
# rules to the Agent. A research session recomputes it from the same repository
# files before every strategy container start and refuses a drifted image, so
# an unrebuilt image can no longer enforce a rule the Agent was never told.
# Only the README needs this record: what the container would really enforce is
# digested from the modules baked above, which stay in the image. The sources
# are staged in repository layout only so both sides run the same function, and
# are deleted again — the image keeps the digest, not the copies.
COPY src/autotrade/environment/strategy.py src/autotrade/environment/strategy_loader.py src/autotrade/environment/strategy_worker.py /opt/autotrade/contract/src/autotrade/environment/
COPY configs/agent_output_template/README.md /opt/autotrade/contract/configs/agent_output_template/README.md
RUN python -c 'from pathlib import Path; from autotrade.environment.contract_fingerprint import IMAGE_FINGERPRINT_PATH, compute_contract_fingerprint; Path(IMAGE_FINGERPRINT_PATH).write_text(compute_contract_fingerprint("/opt/autotrade/contract"), encoding="utf-8")' \
    && rm -rf /opt/autotrade/contract

# COPY preserves the source mode (0600 on the host), so make the trusted modules
# world-readable for the non-root `agent` user that runs them.
RUN chmod -R a+rX /opt/autotrade

# Non-root agent user; Runner/root stays root for frozen execution and binds.
RUN useradd --create-home --uid 61000 agent

# Fixed mount points (populated by docker run -v / --mount). /strategy receives
# the read-only bind of the frozen strategy package (the directory holding
# main.py and its sibling modules); the Agent session binds its decision view
# at /mnt/snapshot.
RUN mkdir -p /mnt/snapshot \
        /mnt/artifacts /mnt/agent/workspace \
        /strategy /strategy-data \
    && chown root:root /mnt

# Image default user stays root (the build never switches away); the executor
# selects the non-root agent user per-process at docker run time.
WORKDIR /mnt/agent

# Static-check advisor for the Agent session. Runtime is offline, so pin
# globally here.
# Same layer verifies the binary; do not install via pip or at session start.
RUN npm install -g --prefix /usr/local --no-fund --no-audit --registry "${NPM_CONFIG_REGISTRY}" pyright@1.1.411 \
    && /usr/local/bin/pyright --version
