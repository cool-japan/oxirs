#!/bin/bash

# OxiRS vs Apache Jena Performance Benchmark Runner
# Usage: ./run_benchmarks.sh [mode] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
JENA_PATH="${JENA_PATH:-$HOME/work/jena}"
DATASETS_PATH="${DATASETS_PATH:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_results}"
MODE="${1:-quick}"
RUNS="${BENCHMARK_RUNS:-10}"
WARMUP="${WARMUP_RUNS:-3}"
TIMEOUT="${TIMEOUT_SECS:-300}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Rust is installed
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo/Rust is not installed. Please install Rust first."
        exit 1
    fi
    
    # Check if Java is installed (for Jena)
    if ! command -v java &> /dev/null; then
        print_error "Java is not installed. Please install Java for Apache Jena."
        exit 1
    fi
    
    # Check if Jena is available
    if [[ ! -d "$JENA_PATH" ]]; then
        print_warning "Jena installation not found at $JENA_PATH"
        print_warning "You can set JENA_PATH environment variable or install Jena"
        print_warning "Benchmarks will still run but Jena comparisons may fail"
    else
        print_success "Found Jena installation at $JENA_PATH"
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        TOTAL_MEM=$(free -g | awk 'NR==2{print $2}')
        if [[ $TOTAL_MEM -lt 4 ]]; then
            print_warning "System has only ${TOTAL_MEM}GB RAM. Benchmarks may be limited."
        fi
    fi
    
    print_success "Prerequisites check completed"
}

# Function to build the benchmark runner
build_benchmark_runner() {
    print_status "Building benchmark runner..."
    
    # Create a temporary Cargo.toml for the benchmark runner
    cat > Cargo.toml.benchmark << EOF
[package]
name = "oxirs-benchmark-runner"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "benchmark_runner"
path = "benchmark_runner.rs"

[dependencies]
anyhow = "1.0"
clap = { version = "4.0", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
tracing = "0.1"
tracing-subscriber = "0.3"
num_cpus = "1.0"

# Add OxiRS dependencies (these would be actual crate dependencies)
# oxirs-arq = { path = "./oxirs-arq" }
# oxirs-shacl = { path = "./oxirs-shacl" }
# oxirs-rule = { path = "./oxirs-rule" }
# oxirs-star = { path = "./oxirs-star" }
# oxirs-vec = { path = "./oxirs-vec" }
EOF

    # Build with cargo
    if cargo build --release --manifest-path Cargo.toml.benchmark; then
        print_success "Benchmark runner built successfully"
    else
        print_error "Failed to build benchmark runner"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "OxiRS vs Apache Jena Performance Benchmark Runner"
    echo "================================================="
    echo ""
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "MODES:"
    echo "  quick       - Run quick benchmark suite (default, ~2-5 minutes)"
    echo "  full        - Run comprehensive benchmark suite (~15-30 minutes)"
    echo "  sparql      - Run only SPARQL query benchmarks"
    echo "  parsing     - Run only RDF parsing benchmarks"
    echo "  reasoning   - Run only reasoning benchmarks"
    echo "  shacl       - Run only SHACL validation benchmarks"
    echo "  vector      - Run only vector search benchmarks"
    echo ""
    echo "ENVIRONMENT VARIABLES:"
    echo "  JENA_PATH        - Path to Apache Jena installation (default: ~/work/jena)"
    echo "  DATASETS_PATH    - Path to test datasets (default: ./data)"
    echo "  OUTPUT_DIR       - Output directory for results (default: ./benchmark_results)"
    echo "  BENCHMARK_RUNS   - Number of benchmark runs (default: 10)"
    echo "  WARMUP_RUNS      - Number of warmup runs (default: 3)"
    echo "  TIMEOUT_SECS     - Timeout per benchmark in seconds (default: 300)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 quick                          # Quick benchmark"
    echo "  $0 full                           # Full benchmark suite"
    echo "  $0 sparql                         # Only SPARQL benchmarks"
    echo "  BENCHMARK_RUNS=20 $0 full         # Full suite with 20 runs per test"
    echo "  JENA_PATH=/opt/jena $0 quick      # Quick suite with custom Jena path"
    echo ""
}

# Function to run benchmarks
run_benchmarks() {
    local mode=$1
    
    print_status "Starting OxiRS vs Apache Jena benchmark suite"
    print_status "Mode: $mode"
    print_status "Runs per test: $RUNS"
    print_status "Warmup runs: $WARMUP"
    print_status "Timeout: ${TIMEOUT}s"
    print_status "Output directory: $OUTPUT_DIR"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Prepare datasets directory
    mkdir -p "$DATASETS_PATH"
    
    # Set up logging
    export RUST_LOG=${RUST_LOG:-info}
    
    # Run the benchmark based on mode
    case $mode in
        "quick"|"full")
            print_status "Running $mode benchmark suite..."
            cargo run --release --manifest-path Cargo.toml.benchmark --bin benchmark_runner -- \
                --mode "$mode" \
                --jena-path "$JENA_PATH" \
                --datasets-path "$DATASETS_PATH" \
                --output-dir "$OUTPUT_DIR" \
                --runs "$RUNS" \
                --warmup "$WARMUP" \
                --timeout "$TIMEOUT" \
                --verbose
            ;;
        "sparql"|"parsing"|"reasoning"|"shacl"|"vector"|"scalability")
            print_status "Running $mode category benchmarks..."
            cargo run --release --manifest-path Cargo.toml.benchmark --bin benchmark_runner -- \
                --mode category \
                --category "$mode" \
                --jena-path "$JENA_PATH" \
                --datasets-path "$DATASETS_PATH" \
                --output-dir "$OUTPUT_DIR" \
                --runs "$RUNS" \
                --warmup "$WARMUP" \
                --timeout "$TIMEOUT" \
                --verbose
            ;;
        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown mode: $mode"
            show_usage
            exit 1
            ;;
    esac
}

# Function to clean up
cleanup() {
    print_status "Cleaning up temporary files..."
    rm -f Cargo.toml.benchmark
    rm -rf target/
}

# Function to generate summary report
generate_summary() {
    if [[ -f "$OUTPUT_DIR/BENCHMARK_SUMMARY.md" ]]; then
        print_success "Benchmark completed! Summary:"
        echo ""
        cat "$OUTPUT_DIR/BENCHMARK_SUMMARY.md"
        echo ""
        print_status "Full results available in:"
        print_status "  HTML Report: $OUTPUT_DIR/benchmark_report.html"
        print_status "  CSV Data: $OUTPUT_DIR/benchmark_results.csv"
        print_status "  JSON Data: $OUTPUT_DIR/benchmark_results.json"
    fi
}

# Main execution
main() {
    echo "🚀 OxiRS vs Apache Jena Performance Benchmark Suite"
    echo "===================================================="
    echo ""
    
    # Check if help was requested
    if [[ "$1" == "help" || "$1" == "-h" || "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Build benchmark runner
    build_benchmark_runner
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run benchmarks
    local start_time=$(date +%s)
    run_benchmarks "$MODE"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Generate summary
    generate_summary
    
    print_success "Benchmark suite completed in ${duration} seconds"
    print_status "Thank you for benchmarking OxiRS!"
}

# Run main function with all arguments
main "$@"