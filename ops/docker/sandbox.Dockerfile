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

# Pre-bake the C/C++/Fortran build toolchain so a fold that pins a source-only
# wheel (e.g. torch_scatter/torch_sparse) builds without declaring apt_packages.
# Without this, the base python:3.11-slim has no compiler and such installs fail
# at build time (the root cause of an early GNN-transfer run's image-build error).
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
# mature release of the CUDA 12.8 line, and the transformers/boosting/PyG
# picks are the current stable releases the ecosystem has settled on.
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
        torch_geometric==2.8.0.post1 \
        transformers==5.14.1 \
        accelerate==1.14.0 \
        safetensors==0.8.0 \
        einops==0.8.2 \
        lightgbm==4.7.0 \
        xgboost==3.2.0 \
        ninja==1.13.0 \
        "huggingface_hub[cli]==1.24.0"

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

# CUDA build toolchain (nvcc + headers/dev libs), completing the pre-baked
# compiler policy above for CUDA-extension source builds (torch_scatter,
# torch_sparse, pyg_lib, ...) declared via sandbox_environment.json. Version
# matches the torch==2.10.0 wheel's CUDA 12.8. Installed from the fixed-version
# TLS runfile because the NVIDIA apt repo key is rejected by Debian trixie's
# Sequoia apt policy. Nsight profilers are dropped to keep the layer lean. Placed
# after the pip/CLI layers so adding it kept their cache.
# TORCH_CUDA_ARCH_LIST targets the host L20s (sm_89): extension builds compile
# one arch instead of all, cutting derived-image build time several-fold.
ARG CUDA_RUNFILE_URL=https://developer.download.nvidia.cn/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
# libxml2 is required by the runfile's cuda-installer (and by some CUDA tools
# at runtime); installed here rather than in the first apt layer so adding the
# toolchain did not invalidate the pip layer cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libxml2 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fL --retry 8 --retry-delay 3 --connect-timeout 30 --max-time 7200 \
        "${CUDA_RUNFILE_URL}" \
        -o /tmp/cuda.run \
    && sh /tmp/cuda.run --silent --toolkit --override --no-man-page \
    && rm -f /tmp/cuda.run \
    && rm -rf /usr/local/cuda-12.8/nsight* /usr/local/cuda-12.8/gds \
        /usr/local/cuda-12.8/libnvvp /usr/local/cuda-12.8/extras/demo_suite \
        /var/log/cuda-installer.log /tmp/cuda-installer.log \
    && /usr/local/cuda-12.8/bin/nvcc --version
# glibc 2.41 (Debian trixie) declares sinpi/cospi/sinpif/cospif with noexcept;
# CUDA 12.8's crt/math_functions.h re-declarations lack it, so any host-side
# nvcc compilation fails (NVIDIA fixed the headers in later toolkits). Mirror
# the distro-standard fix by annotating the four declarations; the count check
# fails the build if a toolkit bump changes the header shape.
RUN sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ +(double|float) +(sinpif|cospif|sinpi|cospi)\(.*\));$/\1 noexcept (true);/' \
        /usr/local/cuda-12.8/include/crt/math_functions.h \
    && test "$(grep -c 'noexcept (true);' /usr/local/cuda-12.8/include/crt/math_functions.h)" = 4

# FORCE_CUDA: docker build has no GPU, so torch-extension setup scripts would
# silently produce CPU-only kernels; the flag makes every source build here
# AND in derived images compile real CUDA kernels for the arch below.
ENV CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=8.9 \
    FORCE_CUDA=1

# Compiled PyG companions, source-built against the baked torch + nvcc (no
# prebuilt cu128/2.10 wheels on PyPI). --no-build-isolation so the builds see
# the installed torch; ninja + the single-arch TORCH_CUDA_ARCH_LIST keep the
# compile bounded. The cuda_version() assertions enforce at build time that
# the extensions really contain CUDA kernels (importable-but-CPU-only would
# pass a plain import check).
RUN pip install --no-cache-dir --no-build-isolation -i ${PIP_INDEX_URL} \
        torch_scatter==2.1.2 \
        torch_sparse==0.6.18 \
        torch_cluster==1.6.3 \
    && python -c "import torch, torch_scatter, torch_sparse, torch_cluster, torch_geometric; \
assert torch.ops.torch_scatter.cuda_version() > 0, 'torch_scatter built without CUDA'; \
assert torch.ops.torch_sparse.cuda_version() > 0, 'torch_sparse built without CUDA'; \
assert torch.ops.torch_cluster.cuda_version() > 0, 'torch_cluster built without CUDA'"

# Trusted host-side runtime baked in: the strategy worker, the loader and the
# strategy context, loaded as `python -m autotrade.environment.strategy_worker`
# in the strategy container. Standard-library plus the pinned data stack only;
# the Broker stays on the host, so the worker never needs broker_core.
WORKDIR /opt/autotrade
COPY src/autotrade/__init__.py /opt/autotrade/autotrade/__init__.py
COPY src/autotrade/environment/__init__.py /opt/autotrade/autotrade/environment/__init__.py
COPY src/autotrade/environment/strategy.py /opt/autotrade/autotrade/environment/strategy.py
COPY src/autotrade/environment/strategy_loader.py /opt/autotrade/autotrade/environment/strategy_loader.py
COPY src/autotrade/environment/strategy_worker.py /opt/autotrade/autotrade/environment/strategy_worker.py
COPY ops/docker/pyrightconfig.json /opt/autotrade/pyrightconfig.json
# COPY preserves the source mode (0600 on the host), so make the trusted modules
# world-readable for the non-root `agent` user that runs them.
RUN chmod -R a+rX /opt/autotrade

# Non-root agent user; Runner/root stays root for frozen execution and binds.
RUN useradd --create-home --uid 61000 agent

# Fixed mount points (populated by docker run -v / --mount). /strategy/main.py
# must exist as a file for the read-only bind of the frozen strategy;
# /opt/autotrade_runtime receives trusted host modules for the Agent session.
RUN mkdir -p /mnt/snapshots/train /mnt/snapshots/valid /mnt/snapshot \
        /mnt/artifacts /mnt/agent/workspace /mnt/agent/output /mnt/agent/models \
        /mnt/runtime /opt/autotrade_runtime /strategy /strategy-data \
    && chown root:root /mnt \
    && touch /strategy/main.py \
    && chmod 0444 /strategy/main.py

# Image default user stays root (the build never switches away); the executor
# selects the non-root agent user per-process at docker run time.
WORKDIR /mnt/agent

# Fold/Explore static-check advisor. Runtime is offline, so pin globally here.
# Same layer verifies the binary; do not install via pip or at session start.
RUN npm install -g --prefix /usr/local --no-fund --no-audit --registry "${NPM_CONFIG_REGISTRY}" pyright@1.1.411 \
    && /usr/local/bin/pyright --version
