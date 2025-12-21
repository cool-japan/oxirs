# OxiRS Fuseki - Benchmark Quick Reference

**Last Updated**: December 9, 2025

## Overview

This document provides a quick reference for running and interpreting oxirs-fuseki benchmarks. All benchmarks use the Criterion.rs framework for statistically rigorous performance measurement.

## Available Benchmark Suites

### 1. Load Testing (`load_testing.rs`)
**Purpose**: Stress testing under various load conditions
**Run**: `cargo bench --bench load_testing`

**Benchmark Groups**:
- `concurrent_queries` - Tests 10, 100, 1000 concurrent requests
- `query_latency` - Response times across dataset sizes
- `throughput` - Sustained queries per second (QPS)
- `batch_queries` - Batch query execution performance
- `result_streaming` - Memory-efficient streaming
- `dataset_operations` - Dataset CRUD operations
- `memory_pressure` - Performance under memory constraints
- `connection_pool` - Connection pool efficiency
- `query_cache` - Cache hit vs miss performance

### 2. Performance Benchmarks (`performance_benchmarks.rs`)
**Purpose**: Detailed analysis of beta.2 optimization features
**Run**: `cargo bench --bench performance_benchmarks`

**Benchmark Groups**:
- **Concurrency** (concurrent.rs features):
  - `work_stealing` - Work-stealing scheduler (2, 4, 8, 16 workers)
  - `priority_queue` - Priority-based request queuing (100, 1K, 10K items)
  - `load_shedding` - Adaptive load shedding (50%, 70%, 85%, 95%)

- **Memory Management** (memory_pool.rs features):
  - `memory_pool` - Pool acquire/release operations (10, 50, 100, 500 ops)
  - `memory_adaptation` - Pressure-based adaptation
  - `chunked_arrays` - Chunked array performance
  - `garbage_collection` - GC cycle efficiency

- **Batching** (batch_execution.rs features):
  - `request_batching` - Automatic batching
  - `adaptive_batching` - Dynamic batch sizing
  - `parallel_execution` - Parallel batch processing

- **Streaming** (streaming_results.rs features):
  - `zero_copy_streaming` - Zero-copy result streaming
  - `compression_streaming` - Compressed streaming (Gzip, Brotli)
  - `backpressure` - Flow control mechanisms

- **Dataset Management** (dataset_management.rs features):
  - `bulk_operations` - Bulk dataset operations
  - `snapshots` - Snapshot creation and management
  - `versioning` - Dataset versioning

## Quick Start Commands

```bash
# Run all benchmarks (WARNING: Takes 30+ minutes)
cargo bench

# Run specific suite
cargo bench --bench load_testing
cargo bench --bench performance_benchmarks

# Run specific benchmark group
cargo bench --bench load_testing -- concurrent_queries
cargo bench --bench performance_benchmarks -- work_stealing

# Run with specific features
cargo bench --features production
cargo bench --all-features

# Reduce measurement time for quick feedback
cargo bench -- --measurement-time 5

# View HTML reports
open target/criterion/report/index.html
```

## Performance Targets (v0.1.0)

| Benchmark Category | Target | Status |
|--------------------|--------|--------|
| **Concurrent Queries (100)** | < 100ms p95 | To be measured |
| **Query Latency (10k triples)** | < 50ms p95 | To be measured |
| **Throughput** | > 10,000 QPS | To be measured |
| **Memory Pool (ops/sec)** | > 1M ops/sec | To be measured |
| **Result Streaming (100k)** | < 500ms | To be measured |
| **Batch Processing (100)** | < 200ms | To be measured |
| **Zero-Copy Streaming** | > 1GB/sec | To be measured |

## Interpreting Results

### Key Metrics

```
concurrent_queries/100  time:   [45.234 ms 47.456 ms 49.678 ms]
                        change: [-5.23% -3.45% -1.67%] (p = 0.02 < 0.05)
                        Performance has improved.
```

- **time**: [lower bound, estimate, upper bound] (95% confidence interval)
- **change**: Performance delta from previous run
- **p-value**: Statistical significance (< 0.05 = significant)
- **R²**: Goodness of fit (closer to 1.0 = more reliable)

### Performance Analysis

**Good Performance Indicators**:
- ✅ Consistent timings (low standard deviation)
- ✅ R² > 0.95 (high confidence)
- ✅ No significant regressions vs baseline
- ✅ Linear scaling with dataset size

**Performance Concerns**:
- ⚠️ High variance between runs
- ⚠️ Unexpected performance regressions
- ⚠️ Non-linear scaling
- ⚠️ High memory usage

## Baseline Comparisons

### vs Apache Jena Fuseki 4.x

| Operation | OxiRS Fuseki | Jena Fuseki | Notes |
|-----------|--------------|-------------|-------|
| Simple SELECT | TBD | ~10ms | 1000 triples |
| JOIN (2-way) | TBD | ~25ms | 10k triples |
| UPDATE (INSERT) | TBD | ~15ms | 100 triples |
| Concurrent (100) | TBD | ~100ms | p95 latency |
| Throughput | TBD | ~8000 QPS | Sustained |

*Note: Actual measurements needed for production validation*

## Best Practices

### Running Benchmarks

1. **Isolate System**: Close unnecessary applications
2. **Consistent Environment**: Use same hardware for comparisons
3. **Multiple Runs**: Run 3-5 times for confidence
4. **Warm-up**: Let caches stabilize before measuring
5. **Documentation**: Record system specs with results

### System Recommendations

```bash
# Optimal benchmark environment
CPU: 8+ cores
RAM: 16GB+
Storage: SSD
OS: Linux (for best performance)
Rust: stable channel

# Disable power-saving
# Disable CPU frequency scaling
# Close browser and IDE
```

### Troubleshooting

**Issue**: Benchmarks taking too long
```bash
# Reduce measurement time
cargo bench -- --measurement-time 5 --sample-size 10
```

**Issue**: Inconsistent results
```bash
# Increase warm-up and sample size
cargo bench -- --warm-up-time 10 --sample-size 100
```

**Issue**: Out of memory
```bash
# Run sequentially
cargo bench -- --test-threads=1
```

## Continuous Performance Monitoring

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run benchmarks
  run: cargo bench --workspace -- --measurement-time 10

- name: Compare with baseline
  run: |
    # Store results in artifact
    tar czf criterion-results.tar.gz target/criterion
```

### Regression Detection

Criterion automatically detects regressions:
```
Performance has regressed (p < 0.05)
Previous: 10.5ms ± 0.5ms
Current:  12.3ms ± 0.6ms
Change: +17.1%
```

## Advanced Usage

### Custom Benchmarks

Add to `benches/custom.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn custom_benchmark(c: &mut Criterion) {
    c.bench_function("my_operation", |b| {
        b.iter(|| {
            // Your code here
            black_box(expensive_operation());
        });
    });
}

criterion_group!(benches, custom_benchmark);
criterion_main!(benches);
```

### Profiling Integration

```bash
# CPU profiling
cargo install flamegraph
cargo flamegraph --bench load_testing -- --bench

# Memory profiling
heaptrack cargo bench --bench load_testing
```

## Benchmark Maintenance

### When to Update Benchmarks

- ✅ After major algorithm changes
- ✅ When adding new features
- ✅ Before release candidates
- ✅ Monthly performance reviews

### Baseline Updates

```bash
# Save new baseline
cargo bench --save-baseline baseline_v0.1.0

# Compare against baseline
cargo bench --baseline baseline_v0.1.0
```

## Additional Resources

- [Criterion.rs Book](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Full Benchmark Documentation](./README.md)
- [OxiRS Performance Guide](../../docs/PERFORMANCE_BASELINE.md)

## Support

For performance issues or benchmark questions:
- **GitHub Issues**: https://github.com/cool-japan/oxirs/issues
- **Documentation**: https://github.com/cool-japan/oxirs/tree/main/docs
- **Performance Baseline**: docs/PERFORMANCE_BASELINE.md

---

*Quick Reference Guide - OxiRS Fuseki Benchmarking*
*Version: 0.1.0-rc.3*
*Last Updated: December 9, 2025*
