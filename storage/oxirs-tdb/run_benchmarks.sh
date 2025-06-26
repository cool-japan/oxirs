#!/bin/bash
# Run performance benchmarks for oxirs-tdb

echo "Running oxirs-tdb performance benchmarks..."
echo "============================================"
echo ""
echo "This will test performance with various dataset sizes:"
echo "- Small datasets (1K - 100K triples)"
echo "- Medium datasets (100K - 1M triples)"
echo "- Large datasets (1M+ triples)"
echo ""
echo "Note: Large scale benchmarks may take significant time to complete"
echo ""

# Run specific benchmarks with different configurations
echo "1. Running insertion benchmarks..."
cargo bench --bench tdb_benchmark -- insertion --save-baseline insertion_baseline

echo ""
echo "2. Running query benchmarks..."
cargo bench --bench tdb_benchmark -- query --save-baseline query_baseline

echo ""
echo "3. Running concurrent access benchmarks..."
cargo bench --bench tdb_benchmark -- concurrent --save-baseline concurrent_baseline

echo ""
echo "4. Running MVCC benchmarks..."
cargo bench --bench tdb_benchmark -- mvcc --save-baseline mvcc_baseline

echo ""
echo "5. Running large scale benchmarks (this may take a while)..."
cargo bench --bench tdb_benchmark -- large_scale --save-baseline large_scale_baseline

echo ""
echo "Benchmark complete! Results saved in target/criterion/"
echo ""
echo "To view the report, open target/criterion/report/index.html"