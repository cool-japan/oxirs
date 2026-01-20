#!/bin/bash
# OxiRS Development Environment Setup

set -e

echo "Setting up OxiRS development environment..."

# Check for required tools
echo "Checking required tools..."

# Check Rust
if ! command -v rustc &> /dev/null; then
    echo "Error: Rust is not installed. Please install from https://rustup.rs/"
    exit 1
fi

# Check cargo-nextest
if ! command -v cargo-nextest &> /dev/null; then
    echo "Installing cargo-nextest..."
    cargo install cargo-nextest
fi

# Check cargo-watch (for development)
if ! command -v cargo-watch &> /dev/null; then
    echo "Installing cargo-watch..."
    cargo install cargo-watch
fi

# Install useful development tools
echo "Installing development tools..."
cargo install cargo-edit
cargo install cargo-audit

# Set up pre-commit hooks
echo "Setting up git hooks..."
mkdir -p .git/hooks

cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# OxiRS pre-commit hook

set -e

echo "Running pre-commit checks..."

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --workspace --all-targets -- -D warnings

# Run tests
cargo nextest run --no-fail-fast || cargo test --workspace

echo "Pre-commit checks passed!"
EOF

chmod +x .git/hooks/pre-commit

# Create data directories
echo "Creating data directories..."
mkdir -p data/{default,example}
mkdir -p logs
mkdir -p shapes

# Create example SHACL shapes
cat > shapes/person.ttl << 'EOF'
@prefix ex: <http://example.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:PersonShape
    a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:name ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
    ] ;
    sh:property [
        sh:path ex:age ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:maxInclusive 150 ;
    ] .
EOF

# Setup example data
cat > data/example.ttl << 'EOF'
@prefix ex: <http://example.org/> .

ex:alice a ex:Person ;
    ex:name "Alice Smith" ;
    ex:age 30 ;
    ex:knows ex:bob .

ex:bob a ex:Person ;
    ex:name "Bob Jones" ;
    ex:age 25 .
EOF

echo "Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Run 'cargo build' to build the project"
echo "2. Run 'cargo nextest run' to run tests"
echo "3. Use 'cargo watch -x check' for continuous compilation"
echo "4. Run './scripts/build.sh' for a full build and test cycle"