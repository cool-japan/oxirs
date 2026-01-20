#!/bin/bash

# OxiRS Rule Engine Benchmark Runner
# This script runs comprehensive benchmarks for the rule engine

set -e

echo "🚀 Starting OxiRS Rule Engine Benchmarks"
echo "========================================"

# Configuration
CARGO_PROFILE="${CARGO_PROFILE:-release}"
BENCH_OUTPUT_DIR="${BENCH_OUTPUT_DIR:-./benchmark_results}"
BENCH_FORMAT="${BENCH_FORMAT:-html}"

# Create output directory
mkdir -p "$BENCH_OUTPUT_DIR"

echo "📊 Configuration:"
echo "  Profile: $CARGO_PROFILE"
echo "  Output Directory: $BENCH_OUTPUT_DIR"
echo "  Output Format: $BENCH_FORMAT"
echo ""

# Function to run benchmark and capture output
run_benchmark() {
    local bench_name="$1"
    local bench_file="$2"
    
    echo "🔍 Running $bench_name benchmarks..."
    
    if [ "$BENCH_FORMAT" = "html" ]; then
        cargo bench --profile="$CARGO_PROFILE" --bench "$bench_file" -- --output-format html
        if [ -d "target/criterion" ]; then
            cp -r target/criterion "$BENCH_OUTPUT_DIR/${bench_name}_criterion"
        fi
    else
        cargo bench --profile="$CARGO_PROFILE" --bench "$bench_file" > "$BENCH_OUTPUT_DIR/${bench_name}_results.txt"
    fi
    
    echo "✅ $bench_name benchmarks completed"
    echo ""
}

# Function to check if benchmarks exist
check_benchmark_exists() {
    local bench_file="$1"
    if [ ! -f "benches/${bench_file}.rs" ]; then
        echo "⚠️  Warning: Benchmark file benches/${bench_file}.rs not found"
        return 1
    fi
    return 0
}

# Run all benchmarks
echo "🏃 Running all benchmark suites..."
echo ""

# Main rule engine benchmarks
if check_benchmark_exists "rule_engine_benchmarks"; then
    run_benchmark "rule_engine" "rule_engine_benchmarks"
fi

# SWRL-specific benchmarks
if check_benchmark_exists "swrl_benchmarks"; then
    run_benchmark "swrl" "swrl_benchmarks"
fi

# Generate summary report
echo "📈 Generating summary report..."

cat > "$BENCH_OUTPUT_DIR/README.md" << EOF
# OxiRS Rule Engine Benchmark Results

Generated on: $(date)
Profile: $CARGO_PROFILE

## Benchmark Suites

### Rule Engine Benchmarks
- **Forward Chaining**: Tests forward reasoning performance with varying fact and rule sizes
- **Backward Chaining**: Tests goal proving performance
- **RETE Network**: Tests RETE pattern matching performance
- **Integration**: Tests integration with oxirs-core performance
- **Memory Usage**: Tests memory consumption patterns
- **Rule Complexity**: Tests performance with increasingly complex rules
- **Concurrent Execution**: Tests parallel rule processing

### SWRL Benchmarks
- **SWRL Execution**: Tests SWRL rule execution performance
- **Complex Rules**: Tests complex SWRL rules with multiple built-ins
- **Built-in Predicates**: Tests individual built-in predicate performance
- **Engine Overhead**: Tests SWRL engine creation and management overhead
- **Variable Binding**: Tests variable unification performance

## Viewing Results

EOF

if [ "$BENCH_FORMAT" = "html" ]; then
    cat >> "$BENCH_OUTPUT_DIR/README.md" << EOF
### HTML Reports
Open the following files in your browser:
- \`rule_engine_criterion/report/index.html\` - Rule engine benchmark report
- \`swrl_criterion/report/index.html\` - SWRL benchmark report

EOF
else
    cat >> "$BENCH_OUTPUT_DIR/README.md" << EOF
### Text Reports
- \`rule_engine_results.txt\` - Rule engine benchmark results
- \`swrl_results.txt\` - SWRL benchmark results

EOF
fi

cat >> "$BENCH_OUTPUT_DIR/README.md" << EOF
## Performance Analysis

To run performance analysis:

\`\`\`bash
cd oxirs-rule
cargo test --release performance_tests
\`\`\`

## Custom Benchmarks

To run specific benchmark groups:

\`\`\`bash
# Run only forward chaining benchmarks
cargo bench forward_chaining

# Run only SWRL built-in benchmarks
cargo bench builtin_predicates

# Run with custom parameters
BENCH_OUTPUT_DIR=./custom_results ./run_benchmarks.sh
\`\`\`

## Interpreting Results

- **Throughput**: Higher is better (operations per second)
- **Latency**: Lower is better (time per operation)
- **Memory**: Lower is better (bytes used)
- **Scalability**: Look for linear or sub-linear growth with input size

Benchmark results include confidence intervals and outlier detection.
Results marked with '⚠️' may indicate performance regressions or bottlenecks.
EOF

echo "✅ Summary report generated: $BENCH_OUTPUT_DIR/README.md"
echo ""

# Performance validation
echo "🔍 Running performance validation tests..."
cargo test --release performance --quiet
echo "✅ Performance validation completed"
echo ""

# Final summary
echo "🎉 All benchmarks completed successfully!"
echo ""
echo "📂 Results available in: $BENCH_OUTPUT_DIR"
echo ""
echo "💡 Next steps:"
echo "   1. Review benchmark results for performance bottlenecks"
echo "   2. Compare with previous results to track performance changes"
echo "   3. Run benchmarks on different hardware for performance profiling"
echo "   4. Use results to guide optimization efforts"
echo ""

if [ "$BENCH_FORMAT" = "html" ]; then
    echo "🌐 Open $BENCH_OUTPUT_DIR/rule_engine_criterion/report/index.html to view detailed results"
fi