# OxiRS Chat - Production Docker Image
# Multi-stage build for minimal image size

# Stage 1: Build
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace Cargo files
COPY ../../Cargo.toml ../../Cargo.lock ./
COPY ../../core ./core
COPY ../../engine ./engine
COPY ../../storage ./storage
COPY ../../stream ./stream
COPY ../../ai ./ai
COPY ../../tools ./tools
COPY ../../server ./server

# Build dependencies first (cache layer)
RUN cargo build --release -p oxirs-chat --bin oxirs-chat

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 oxirs && \
    mkdir -p /app/data /app/config && \
    chown -R oxirs:oxirs /app

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/oxirs-chat /usr/local/bin/oxirs-chat

# Copy default configuration
COPY oxirs.toml /app/config/oxirs.toml

# Set ownership
RUN chown -R oxirs:oxirs /app

# Switch to non-root user
USER oxirs

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV OXIRS_CONFIG=/app/config/oxirs.toml

# Run the application
ENTRYPOINT ["/usr/local/bin/oxirs-chat"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--persistence-path", "/app/data/sessions"]
