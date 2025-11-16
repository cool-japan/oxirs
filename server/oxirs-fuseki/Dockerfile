# OxiRS Fuseki - Production-Ready SPARQL Server
# Multi-stage build for minimal image size

# Build stage
FROM rust:1.82-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libgmp-dev \
    libmpfr-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace manifests
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo

# Copy all workspace crates (oxirs-fuseki depends on other workspace crates)
COPY core ../core
COPY engine ../engine
COPY storage ../storage
COPY stream ../stream
COPY ai ../ai
COPY tools ../tools
COPY server ./server

# Build in release mode with optimizations
WORKDIR /app/server/oxirs-fuseki
RUN cargo build --release --bin oxirs-fuseki \
    --features production,metrics,auth,tls,ldap \
    && strip target/release/oxirs-fuseki

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libgmp10 \
    libmpfr6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r oxirs && useradd -r -g oxirs oxirs

# Create data directories
RUN mkdir -p /data/oxirs /var/log/oxirs /etc/oxirs && \
    chown -R oxirs:oxirs /data/oxirs /var/log/oxirs /etc/oxirs

# Copy binary from builder
COPY --from=builder /app/server/oxirs-fuseki/target/release/oxirs-fuseki /usr/local/bin/

# Copy default configuration
COPY --chown=oxirs:oxirs server/oxirs-fuseki/oxirs.toml /etc/oxirs/oxirs.toml

# Switch to non-root user
USER oxirs

# Expose ports
# 3030 - SPARQL HTTP endpoint
# 9090 - Metrics endpoint
# 8080 - Admin UI
EXPOSE 3030 9090 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/usr/local/bin/oxirs-fuseki", "--health-check"]

# Set working directory
WORKDIR /data/oxirs

# Environment variables
ENV RUST_LOG=info \
    OXIRS_CONFIG=/etc/oxirs/oxirs.toml \
    OXIRS_DATA=/data/oxirs \
    OXIRS_LOG_DIR=/var/log/oxirs

# Run the server
ENTRYPOINT ["/usr/local/bin/oxirs-fuseki"]
CMD ["--config", "/etc/oxirs/oxirs.toml"]
