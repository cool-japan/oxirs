# OxiRS Docker Image
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY core/ ./core/
COPY server/ ./server/
COPY engine/ ./engine/
COPY storage/ ./storage/
COPY stream/ ./stream/
COPY ai/ ./ai/
COPY tools/ ./tools/

# Build the project
RUN cargo build --release --bin oxirs --bin oxirs-fuseki --bin oxirs-gql --bin oxirs-chat

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create oxirs user
RUN useradd -r -s /bin/false oxirs

# Copy binaries
COPY --from=builder /app/target/release/oxirs /usr/local/bin/
COPY --from=builder /app/target/release/oxirs-fuseki /usr/local/bin/
COPY --from=builder /app/target/release/oxirs-gql /usr/local/bin/
COPY --from=builder /app/target/release/oxirs-chat /usr/local/bin/

# Copy default configuration
COPY oxirs.toml /etc/oxirs/oxirs.toml

# Create data directory
RUN mkdir -p /data && chown oxirs:oxirs /data

# Switch to oxirs user
USER oxirs

# Set working directory
WORKDIR /data

# Expose ports
EXPOSE 3030 4000 8080

# Default command (SPARQL server)
CMD ["oxirs-fuseki", "--config", "/etc/oxirs/oxirs.toml", "--port", "3030"]