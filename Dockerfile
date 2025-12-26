# Multi-stage build for OxiRS Digital Twin Platform
FROM rust:1.83-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo

# Copy all crates
COPY core core
COPY server server
COPY engine engine
COPY storage storage
COPY stream stream
COPY ai ai
COPY tools tools

# Build with all digital twin features
RUN cargo build --release \
    --bin oxirs-fuseki \
    --features "ngsi-ld,industry40,ids-connector,physics-sim"

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 -s /bin/bash oxirs

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/oxirs-fuseki /usr/local/bin/oxirs-fuseki

# Create data directories
RUN mkdir -p /data/datasets /data/logs /data/config && \
    chown -R oxirs:oxirs /data /app

# Copy default configuration
COPY --chown=oxirs:oxirs server/oxirs-fuseki/oxirs.toml /data/config/oxirs.toml

# Switch to app user
USER oxirs

# Expose ports
EXPOSE 3030

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3030/$/ping || exit 1

# Environment variables
ENV RUST_LOG=info \
    OXIRS_DATA_DIR=/data/datasets \
    OXIRS_LOG_DIR=/data/logs \
    OXIRS_CONFIG=/data/config/oxirs.toml

# Run the server
CMD ["oxirs-fuseki", "--config", "/data/config/oxirs.toml"]
