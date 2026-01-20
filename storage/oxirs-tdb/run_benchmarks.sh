#!/bin/bash
# Comprehensive performance benchmarks for oxirs-tdb

set -e  # Exit on any error

echo "🚀 OxiRS TDB Comprehensive Performance Benchmark Suite"
echo "======================================================="
echo ""
echo "This comprehensive benchmark suite tests:"
echo "• Basic operations (insert, query, transactions)"
echo "• Advanced compression algorithms"
echo "• Memory usage patterns"
echo "• Index performance with different data distributions"
echo "• Real-world knowledge graph patterns"
echo "• Recovery and checkpoint operations"
echo "• Large-scale performance (1M+ triples)"
echo ""
echo "⚠️  Note: Full benchmark suite may take 30-60 minutes to complete"
echo ""

# Function to run a benchmark category
run_benchmark() {
    local category=$1
    local description=$2
    echo "📊 Running $description..."
    cargo bench --bench tdb_benchmark -- "$category" --save-baseline "${category}_baseline"
    echo "✅ $description completed"
    echo ""
}

# Parse command line arguments
QUICK_MODE=false
SPECIFIC_BENCHMARK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            echo "🏃 Quick mode enabled - running essential benchmarks only"
            echo ""
            shift
            ;;
        --benchmark)
            SPECIFIC_BENCHMARK="$2"
            echo "🎯 Running specific benchmark: $SPECIFIC_BENCHMARK"
            echo ""
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick         Run essential benchmarks only (~10 minutes)"
            echo "  --benchmark X   Run specific benchmark category"
            echo "  --help          Show this help message"
            echo ""
            echo "Available benchmark categories:"
            echo "  insertion, query, concurrent, mvcc, compression,"
            echo "  memory_usage, index_patterns, real_world_patterns,"
            echo "  recovery, large_scale"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if cargo and criterion are available
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: cargo not found. Please install Rust toolchain."
    exit 1
fi

# Ensure we're in the right directory
if [[ ! -f "Cargo.toml" ]]; then
    echo "❌ Error: Cargo.toml not found. Please run from the project root."
    exit 1
fi

# Create results directory
mkdir -p target/benchmark-results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="target/benchmark-results/run_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR"
echo ""

# Run specific benchmark if requested
if [[ -n "$SPECIFIC_BENCHMARK" ]]; then
    case "$SPECIFIC_BENCHMARK" in
        insertion) run_benchmark "insertion" "Insertion Performance" ;;
        query) run_benchmark "query" "Query Performance" ;;
        concurrent) run_benchmark "concurrent" "Concurrent Access" ;;
        mvcc) run_benchmark "mvcc" "MVCC Performance" ;;
        compression) run_benchmark "compression" "Compression Algorithms" ;;
        memory_usage) run_benchmark "memory_usage" "Memory Usage Patterns" ;;
        index_patterns) run_benchmark "index_patterns" "Index Performance Patterns" ;;
        real_world_patterns) run_benchmark "real_world_patterns" "Real-World Data Patterns" ;;
        recovery) run_benchmark "recovery" "Recovery and Checkpoints" ;;
        large_scale) run_benchmark "large_scale" "Large Scale Performance" ;;
        *)
            echo "❌ Unknown benchmark: $SPECIFIC_BENCHMARK"
            exit 1
            ;;
    esac
    exit 0
fi

# Run benchmark suite
if [[ "$QUICK_MODE" == "true" ]]; then
    echo "🏃 Running essential benchmarks (quick mode)..."
    run_benchmark "insertion" "Basic Insertion Performance"
    run_benchmark "query" "Basic Query Performance"
    run_benchmark "compression" "Compression Performance"
    run_benchmark "concurrent" "Concurrent Access Performance"
else
    echo "🔥 Running comprehensive benchmark suite..."
    echo ""
    
    # Core functionality benchmarks
    run_benchmark "insertion" "Triple Insertion Performance"
    run_benchmark "query" "Query Performance"
    run_benchmark "concurrent" "Concurrent Access Performance"
    run_benchmark "mvcc" "MVCC Transaction Performance"
    
    # Advanced feature benchmarks
    run_benchmark "compression" "Compression Algorithm Performance"
    run_benchmark "memory_usage" "Memory Usage Optimization"
    run_benchmark "index_patterns" "Index Performance with Different Data Patterns"
    run_benchmark "real_world_patterns" "Real-World Knowledge Graph Patterns"
    run_benchmark "recovery" "Recovery and Checkpoint Performance"
    
    # Large scale testing (most time-consuming)
    echo "⏰ Starting large-scale performance testing..."
    echo "   This may take 20-30 minutes..."
    run_benchmark "large_scale" "Large Scale Performance (1M+ triples)"
fi

# Generate summary report
echo "📈 Generating benchmark summary..."
cat > "$RESULTS_DIR/benchmark_summary.md" << EOF
# OxiRS TDB Benchmark Results

**Date:** $(date)
**Mode:** $(if [[ "$QUICK_MODE" == "true" ]]; then echo "Quick Mode"; else echo "Comprehensive"; fi)
**Git Commit:** $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

## System Information
- **OS:** $(uname -s) $(uname -r)
- **CPU:** $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs 2>/dev/null || echo "unknown")
- **Memory:** $(grep MemTotal /proc/meminfo | awk '{print $2 " " $3}' 2>/dev/null || echo "unknown")
- **Rust Version:** $(rustc --version)

## Benchmark Categories Completed

EOF

if [[ "$QUICK_MODE" == "true" ]]; then
    cat >> "$RESULTS_DIR/benchmark_summary.md" << EOF
- ✅ Basic Insertion Performance
- ✅ Basic Query Performance  
- ✅ Compression Performance
- ✅ Concurrent Access Performance

EOF
else
    cat >> "$RESULTS_DIR/benchmark_summary.md" << EOF
- ✅ Triple Insertion Performance
- ✅ Query Performance
- ✅ Concurrent Access Performance
- ✅ MVCC Transaction Performance
- ✅ Compression Algorithm Performance
- ✅ Memory Usage Optimization
- ✅ Index Performance with Different Data Patterns
- ✅ Real-World Knowledge Graph Patterns
- ✅ Recovery and Checkpoint Performance
- ✅ Large Scale Performance (1M+ triples)

EOF
fi

cat >> "$RESULTS_DIR/benchmark_summary.md" << EOF
## Key Metrics to Review

1. **Insertion Rate**: Target >10M triples/minute for bulk loading
2. **Query Response**: Target <1s for complex queries on 100M+ triples
3. **Transaction Throughput**: Target >10K transactions/second
4. **Memory Efficiency**: Target <8GB for 100M triple database
5. **Compression Ratio**: Target >50% space savings with advanced compression

## View Detailed Results

Open the Criterion HTML report: \`target/criterion/report/index.html\`

Or browse individual benchmark results in the \`target/criterion/\` directory.
EOF

echo ""
echo "🎉 Benchmark suite completed successfully!"
echo ""
echo "📊 Summary report: $RESULTS_DIR/benchmark_summary.md"
echo "🌐 Detailed HTML report: target/criterion/report/index.html"
echo ""
echo "🎯 Key Performance Targets:"
echo "   • Insertion: >10M triples/minute"
echo "   • Query: <1s response for 100M+ triples"
echo "   • Transactions: >10K/second"
echo "   • Memory: <8GB for 100M triples"
echo "   • Compression: >50% space savings"
echo ""

# Copy results to timestamped directory
cp -r target/criterion "$RESULTS_DIR/criterion_results" 2>/dev/null || true

echo "✨ All results saved to: $RESULTS_DIR"