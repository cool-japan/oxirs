# OxiRS-Star Performance Benchmarks

*Last Updated: November 14, 2025*

## Overview

This document describes the comprehensive benchmarking suite for oxirs-star, including performance tests for v0.1.3 features and beyond.

## Benchmark Suites

### 1. Core Benchmarks (`benches/benchmarks.rs`)

Basic performance tests for fundamental operations:
- Triple parsing and serialization
- Store operations (insert, query, remove)
- Basic graph operations

### 2. Enhanced Benchmarks (`benches/enhanced_benchmarks.rs`)

Advanced performance tests covering:
- **Parsing Complexity**: Tests with different nesting depths (0-3 levels)
- **Serialization Structures**: Flat, quoted-heavy, mixed, and deeply-nested graphs
- **Store Indexing**: Query patterns (subject-only, predicate-only, all-bound, etc.)
- **Memory Performance**: Large graph operations and memory efficiency

### 3. Adaptive Benchmarks (`benches/adaptive_benchmarks.rs`) - NEW in v0.1.3

Comprehensive benchmarks for the latest features:

#### ChunkedIterator Benchmarks

**`benchmark_chunked_iterator`**
- Tests various data sizes: 100, 1K, 10K, 100K elements
- Tests various chunk sizes: 10, 100, 1K elements
- Measures throughput and latency
- Uses logarithmic scale for summary visualization

**`benchmark_chunking_comparison`**
- Compares ChunkedIterator vs manual chunking methods
- Tests against stdlib `.chunks()` method
- 10K elements with 100-element chunks
- Measures relative performance

**`benchmark_triple_batching`**
- Real-world RDF triple batch processing
- Tests batch sizes: 10, 50, 100, 500, 1000 triples
- 10K total triples
- Measures batch processing efficiency

**`benchmark_chunked_memory_efficiency`**
- Large dataset processing: 1M elements
- Various chunk sizes: 100, 1K, 10K, 100K
- Simulates streaming processing
- Measures memory-efficient iteration

#### Adaptive Optimizer Benchmarks

**`benchmark_strategy_selection`**
- Tests optimizer with different query complexities:
  - **Simple**: Basic triple patterns
  - **Medium**: With OPTIONAL and FILTER
  - **Complex**: Quoted triples + OPTIONAL + FILTER + GROUP BY
  - **Very Complex**: Nested quoted triples + UNION + FILTER + GROUP BY + ORDER BY
- Measures full optimization workflow including strategy selection

**`benchmark_adaptive_optimization`**
- Complete optimization workflow
- Tests simple and complex queries
- 10-second measurement time for statistical significance
- Includes ML, Quantum, and Classical strategy selection overhead

**`benchmark_regression_detection`**
- Tests regression detector performance:
  - `update_only`: Baseline update overhead
  - `update_and_detect`: Combined operation
  - `detect_only`: Detection without update
- Established baseline with 40 samples

**`benchmark_multi_objective_setup`**
- Multi-objective optimization configuration
- Tests objective setting overhead
- Measures optimization with multiple objectives:
  - MinimizeLatency (50%)
  - MinimizeMemory (30%)
  - MaximizeAccuracy (20%)

**`benchmark_workload_profiling`**
- Workload pattern analysis overhead
- Tests homogeneous vs heterogeneous workloads
- 100 queries per workload type
- Measures profiling and statistics collection

**`benchmark_auto_tuning_warmup`**
- Auto-tuning warmup phase performance
- Tests warmup sizes: 10, 25, 50, 100 queries
- 15-second measurement time
- Measures statistics collection overhead

## Running Benchmarks

### Run All Benchmarks

```bash
cargo bench
```

### Run Specific Benchmark Suite

```bash
# Core benchmarks
cargo bench --bench benchmarks

# Enhanced benchmarks
cargo bench --bench enhanced_benchmarks

# Adaptive benchmarks (v0.1.3 features)
cargo bench --bench adaptive_benchmarks
```

### Run Specific Benchmark

```bash
# Run only ChunkedIterator benchmarks
cargo bench --bench adaptive_benchmarks chunked_iterator

# Run only adaptive optimizer benchmarks
cargo bench --bench adaptive_benchmarks adaptive_strategy

# Run specific benchmark function
cargo bench --bench adaptive_benchmarks benchmark_regression_detection
```

### Filter by Pattern

```bash
# All benchmarks containing "chunk"
cargo bench chunk

# All benchmarks containing "regression"
cargo bench regression

# All optimization benchmarks
cargo bench optim
```

## Benchmark Configuration

### Measurement Settings

- **Default**: 5 seconds measurement time
- **Extended**: 10-15 seconds for optimizer benchmarks
- **Sampling**: Adaptive based on variance
- **Confidence**: 95% confidence interval

### Data Sizes

| Category | Small | Medium | Large | Very Large |
|----------|-------|---------|-------|------------|
| Elements | 100 | 1K | 10K | 100K-1M |
| Triples | 10 | 100 | 1K | 10K+ |
| Queries | 10 | 25 | 50 | 100 |

### Chunk Sizes

Tested chunk sizes for ChunkedIterator:
- **Small**: 10 elements
- **Medium**: 100 elements
- **Large**: 1,000 elements
- **Very Large**: 10,000-100,000 elements

## Expected Performance Characteristics

### ChunkedIterator

- **Throughput**: Should scale linearly with data size
- **Overhead**: Minimal compared to manual chunking (<5%)
- **Memory**: Constant per-chunk memory usage
- **Best Chunk Size**: 100-1000 elements for most workloads

### Adaptive Optimizer

- **Simple Queries**: <100μs optimization time
- **Complex Queries**: <1ms optimization time
- **Strategy Selection**: <50μs overhead
- **Regression Detection**: <10μs per update
- **Workload Profiling**: <100μs per query

### Regression Detection

- **Update**: O(1) amortized
- **Detection**: O(window_size) - typically 100 samples
- **Memory**: Fixed size rolling window
- **Baseline**: Established after 30 samples

## Optimization Tips

### For Best ChunkedIterator Performance

1. **Match chunk size to processing batch**: Use 100-1000 elements
2. **Avoid tiny chunks**: <10 elements has overhead
3. **Avoid huge chunks**: >10K elements may stress memory
4. **Use size_hint**: Helps pre-allocate vectors

### For Best Adaptive Optimizer Performance

1. **Enable auto-tuning**: After 50+ queries for best accuracy
2. **Provide warmup**: 30-50 queries for baseline establishment
3. **Set appropriate objectives**: Focus on 1-3 most important metrics
4. **Monitor statistics**: Use `optimizer.statistics()` for insights

### For Best Regression Detection Performance

1. **Set appropriate threshold**: 1.3-1.5x for sensitivity
2. **Use adequate window size**: 50-100 samples for stability
3. **Establish baseline early**: First 30-40 samples are critical
4. **Monitor severity levels**: Adjust thresholds based on needs

## Benchmark Output

Benchmarks generate reports in `target/criterion/`:
- HTML reports with graphs
- Statistical analysis
- Historical comparison
- Regression detection (via Criterion)

View reports:
```bash
open target/criterion/report/index.html
```

## Continuous Integration

Benchmarks are designed to run in CI:
- Use `--bench --no-run` for compilation checks
- Use `cargo bench --bench adaptive_benchmarks -- --quick` for fast CI runs
- Full benchmarks run on release branches

## Performance Regression Detection

Criterion automatically detects performance regressions:
- Compares against previous runs
- Uses statistical significance testing
- Flags >5% regressions by default
- Stores baseline in `target/criterion/`

## Contributing

When adding new benchmarks:
1. Follow existing naming conventions
2. Use appropriate measurement times
3. Include throughput measurements
4. Add benchmark to criterion groups
5. Document expected performance
6. Update this BENCHMARKS.md

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Performance Testing Best Practices](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [OxiRS Performance Guide](../../../docs/performance.md)
