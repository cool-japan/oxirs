#!/bin/bash
# OxiRS Build Script

set -e

echo "Building OxiRS workspace..."

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Cargo.toml not found. Please run from the workspace root."
    exit 1
fi

# Clean previous builds (optional)
if [ "$1" = "--clean" ]; then
    echo "Cleaning previous builds..."
    cargo clean
fi

# Build the workspace
echo "Building all crates..."
cargo build --workspace --release

# Build with all features if requested
if [ "$1" = "--all-features" ]; then
    echo "Building with all features..."
    cargo build --workspace --release --all-features
fi

# Run tests
echo "Running tests..."
cargo nextest run --no-fail-fast || cargo test --workspace

# Run clippy
echo "Running clippy..."
cargo clippy --workspace --all-targets -- -D warnings

# Check formatting
echo "Checking formatting..."
cargo fmt --all -- --check

echo "Build completed successfully!"
echo ""
echo "Binaries are available in target/release/:"
echo "  - oxirs (CLI tool)"
echo "  - oxirs-fuseki (SPARQL server)"
echo "  - oxirs-gql (GraphQL server)"
echo "  - oxirs-chat (Chat API server)"