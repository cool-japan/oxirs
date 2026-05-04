# syntax=docker/dockerfile:1.7
#
# OxiRS Embed — production multi-stage build.
#
# Stage 1 (builder): cargo-chef cooks dependencies, then we compile the
# embedding-server binary with `--release --features api-server,basic-models`.
# Stage 2 (runtime): scratch base + the static musl binary; absolute minimum
# attack surface and ~30 MB image size.
#
# Build:
#   docker build -f deploy/Dockerfile -t oxirs-embed:0.3.0 .
# Run (single-node):
#   docker compose -f deploy/docker-compose.yml up
#
# IMPORTANT: cargo-chef is **not** required — we use a plain two-stage build
# that relies on Docker layer cache.  If you want incremental builds add
# `cargo-chef` and split dependency compilation into its own layer.

ARG RUST_VERSION=1.85
ARG MUSL_TARGET=x86_64-unknown-linux-musl

# ── builder ─────────────────────────────────────────────────────────────────
FROM rust:${RUST_VERSION}-slim-bookworm AS builder

ARG MUSL_TARGET
ENV CARGO_TERM_COLOR=always

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        pkg-config \
        musl-tools \
        clang \
        cmake \
        git \
        make \
        libssl-dev \
 && rm -rf /var/lib/apt/lists/*

RUN rustup target add ${MUSL_TARGET}

# Workspace layout: copy only what oxirs-embed actually needs.  We use a
# coarse copy of the workspace because oxirs-embed depends on several
# in-workspace crates (oxirs-vec, oxirs-core, ...).  The `.dockerignore`
# (kept alongside this Dockerfile, opt-in) trims target/ and tests/.
WORKDIR /workspace
COPY . .

# Build the embedding server.  The api-server feature pulls in axum;
# basic-models keeps the image small (no neural-models / GPU deps).
RUN --mount=type=cache,target=/workspace/target \
    --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --release \
        --target ${MUSL_TARGET} \
        --manifest-path ai/oxirs-embed/Cargo.toml \
        --features "api-server basic-models" \
        --example advanced_embedding_server \
 && cp /workspace/target/${MUSL_TARGET}/release/examples/advanced_embedding_server \
       /workspace/oxirs-embed-server

# ── runtime ──────────────────────────────────────────────────────────────────
FROM scratch AS runtime

# A truly minimal runtime: no shell, no package manager, no /etc/passwd.
# The musl binary is fully statically linked, so this is sufficient for the
# embedding HTTP service.  CA certs are bundled in case downstream code
# needs to fetch a remote model.
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /workspace/oxirs-embed-server /usr/local/bin/oxirs-embed-server

# Default config lives outside the image; mount via a ConfigMap or
# `docker run -v` to override.
ENV OXIRS_EMBED_BIND="0.0.0.0:8080"
ENV OXIRS_EMBED_LOG="info"
ENV OXIRS_EMBED_METRICS="0.0.0.0:9090"

EXPOSE 8080 9090

# scratch has no /usr/bin/test or healthcheck shell, so we let docker-compose
# / k8s use httpGet probes against /health and /metrics instead.

ENTRYPOINT ["/usr/local/bin/oxirs-embed-server"]
