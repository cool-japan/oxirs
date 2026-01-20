# Performance Tuning Guide - OxiRS TTL

This guide provides advanced techniques for optimizing RDF parsing and serialization performance.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Profiling and Measurement](#profiling-and-measurement)
- [Parse Performance](#parse-performance)
- [Serialization Performance](#serialization-performance)
- [Memory Optimization](#memory-optimization)
- [I/O Optimization](#i/o-optimization)
- [Parallel Processing](#parallel-processing)
- [Platform-Specific Tuning](#platform-specific-tuning)
- [Benchmarking](#benchmarking)

## Performance Overview

### Current Performance

Measured on Apple M1 with typical RDF datasets:

| Format | Parse Speed | Serialize Speed | Memory Usage |
|--------|-------------|-----------------|--------------|
| **Turtle** | 250-300K triples/s | 180-200K triples/s | ~3x file size |
| **N-Triples** | 400-500K triples/s | 350-400K triples/s | ~2x file size |
| **TriG** | 200-250K triples/s | 160-180K triples/s | ~3.5x file size |
| **N-Quads** | 350-450K triples/s | 300-350K triples/s | ~2.2x file size |

### Performance Features

oxirs-ttl includes several built-in optimizations:

- **SIMD lexing** - Uses `memchr` for 2-4x faster byte scanning
- **Zero-copy parsing** - Minimizes string allocations with `Cow<str>`
- **String interning** - Deduplicates common IRIs (RDF namespaces)
- **Lazy IRI resolution** - Defers IRI normalization until needed
- **Buffer pooling** - Reuses parsing buffers in streaming mode

## Profiling and Measurement

### Built-in Profiler

Use `TtlProfiler` to measure parsing performance:

```rust
use oxirs_ttl::profiling::TtlProfiler;
use oxirs_ttl::turtle::TurtleParser;
use std::fs::File;

let mut profiler = TtlProfiler::new();
let file = File::open("data.ttl")?;
let parser = TurtleParser::new();

profiler.start_parse();
let triples = parser.parse(file)?;
profiler.end_parse();

let stats = profiler.get_stats();
println!("Performance Report:");
println!("  Triples parsed: {}", stats.triples_parsed);
println!("  Parse time: {:.2}s", stats.elapsed_seconds);
println!("  Throughput: {:.0} triples/sec", stats.throughput);
println!("  Memory used: {:.2} MB", stats.memory_mb);
```

### Detailed Profiling Report

Get comprehensive performance metrics:

```rust
let report = profiler.generate_report();
println!("{}", report);

// Output:
// ========================================
// OxiRS TTL Performance Report
// ========================================
// Parsing Statistics:
//   Total triples: 1,234,567
//   Parse time: 4.23 seconds
//   Throughput: 291,847 triples/second
//   Memory used: 245.6 MB
//   Peak memory: 278.9 MB
//
// Breakdown:
//   Lexing: 1.42s (33.6%)
//   Parsing: 1.89s (44.7%)
//   IRI resolution: 0.52s (12.3%)
//   Other: 0.40s (9.4%)
```

### System Profiling Tools

For deeper analysis, use system profiling tools:

**macOS/Linux** - `perf`:
```bash
cargo build --release
perf record --call-graph=dwarf ./target/release/my_parser
perf report
```

**macOS** - Instruments:
```bash
cargo build --release
instruments -t "Time Profiler" ./target/release/my_parser
```

**Linux** - `valgrind` (memory):
```bash
cargo build --release
valgrind --tool=massif ./target/release/my_parser
massif-visualizer massif.out.*
```

### Flamegraphs

Generate flamegraphs for visual profiling:

```bash
cargo install flamegraph
cargo flamegraph --root -- data.ttl
```

## Parse Performance

### Choose the Right Parser

Different formats have different performance characteristics:

```rust
// Fastest - N-Triples (line-based, simple)
let parser = NTriplesParser::new();  // 400-500K triples/s

// Fast - Turtle with optimizations
let parser = TurtleParser::new();  // 250-300K triples/s

// Moderate - TriG (named graphs)
let parser = TriGParser::new();  // 200-250K triples/s
```

### Enable Zero-Copy Mode

Reduce string allocations:

```rust
use oxirs_ttl::toolkit::zero_copy::ZeroCopyParser;

// Standard parsing (copies strings)
let parser = TurtleParser::new();

// Zero-copy parsing (uses Cow<str>)
let parser = TurtleParser::new().with_zero_copy(true);
// ~20-30% faster for large datasets
```

### String Interning

Enable IRI deduplication for datasets with many repeated IRIs:

```rust
use oxirs_ttl::toolkit::string_interner::StringInterner;

let mut interner = StringInterner::new();

// Pre-populate with common namespaces
interner.add_common_rdf_namespaces();

let parser = TurtleParser::new()
    .with_interner(interner);

// 15-25% memory reduction for typical datasets
```

### Lazy IRI Resolution

Defer IRI normalization:

```rust
use oxirs_ttl::toolkit::lazy_iri::LazyIri;

let parser = TurtleParser::new()
    .with_lazy_iri_resolution(true);

// IRIs are only resolved when needed
// ~10-15% faster parsing
```

### SIMD Optimization

oxirs-ttl automatically uses SIMD for lexing via `memchr`:

```rust
// Automatically enabled - no configuration needed
let parser = TurtleParser::new();

// Uses SIMD for:
// - Whitespace scanning
// - Delimiter finding
// - Line counting
// - Comment detection
```

To verify SIMD is being used:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Batch Size Tuning

Optimize batch size for your workload:

```rust
use oxirs_ttl::StreamingConfig;

// Small datasets - small batches
let config = StreamingConfig::default()
    .with_batch_size(5_000);

// Medium datasets - medium batches
let config = StreamingConfig::default()
    .with_batch_size(50_000);

// Large datasets - large batches
let config = StreamingConfig::default()
    .with_batch_size(100_000);
```

**Benchmark to find optimal batch size**:
```rust
fn benchmark_batch_sizes() {
    let sizes = [1_000, 5_000, 10_000, 50_000, 100_000];

    for &size in &sizes {
        let config = StreamingConfig::default().with_batch_size(size);
        let start = Instant::now();

        // Parse with this batch size
        let parser = StreamingParser::with_config(file, config);
        for batch in parser.batches() {
            let _ = batch.unwrap();
        }

        let elapsed = start.elapsed();
        println!("Batch size {}: {:?}", size, elapsed);
    }
}
```

## Serialization Performance

### Optimized Serialization

Use optimized serialization for compact, fast output:

```rust
use oxirs_ttl::turtle::TurtleSerializer;

let serializer = TurtleSerializer::new();

// Optimized mode - 76% more compact
let turtle = serializer.serialize_optimized(&triples)?;

// Features:
// - Predicate grouping (semicolons)
// - Object lists (commas)
// - Blank node inlining
// - Collection syntax
```

### Auto-Generated Prefixes

Let the serializer auto-generate prefixes:

```rust
use oxirs_ttl::toolkit::{Serializer, SerializationConfig};

let config = SerializationConfig::default()
    .with_use_prefixes(true)
    .with_auto_generate_prefixes(true);

let serializer = TurtleSerializer::with_config(config);

// Automatically detects common namespaces
// ~40% size reduction vs. full IRIs
```

### Streaming Serialization

For large datasets, use streaming serialization:

```rust
use oxirs_ttl::turtle::TurtleSerializer;
use std::io::BufWriter;

let file = File::create("output.ttl")?;
let mut writer = BufWriter::new(file);

let serializer = TurtleSerializer::new();

// Write in batches to avoid memory accumulation
for batch in triple_batches {
    serializer.serialize_batch(&batch, &mut writer)?;
}

writer.flush()?;
```

### Parallel Serialization

Serialize batches in parallel:

```rust
use rayon::prelude::*;

let batches: Vec<Vec<Triple>> = /* ... */;

// Serialize batches in parallel
let serialized: Vec<String> = batches
    .par_iter()
    .map(|batch| {
        let serializer = TurtleSerializer::new();
        serializer.serialize_to_string(batch).unwrap()
    })
    .collect();

// Combine results
let combined = serialized.join("\n");
```

## Memory Optimization

### Streaming to Reduce Memory

Always use streaming for large datasets:

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};

// Bad - loads entire file into memory
let parser = TurtleParser::new();
let all_triples = parser.parse_file("huge.ttl")?;  // May OOM

// Good - processes in batches
let config = StreamingConfig::default().with_batch_size(10_000);
let parser = StreamingParser::with_config(file, config);

for batch in parser.batches() {
    let triples = batch?;
    process_batch(&triples)?;
    // Batch is dropped here, freeing memory
}
```

### Buffer Pool Reuse

Enable buffer pooling for streaming:

```rust
use oxirs_ttl::toolkit::buffer_manager::BufferManager;

let mut buffer_manager = BufferManager::new();

// Buffers are reused across batches
let config = StreamingConfig::default()
    .with_buffer_manager(buffer_manager);
```

### Memory-Mapped Files

For very large files, consider memory-mapped I/O:

```rust
use memmap2::Mmap;
use std::fs::File;

let file = File::open("huge.ttl")?;
let mmap = unsafe { Mmap::map(&file)? };

let parser = TurtleParser::new();
let triples = parser.parse(&mmap[..])?;
```

### Monitor Memory Usage

Track memory during parsing:

```rust
use oxirs_ttl::profiling::MemoryTracker;

let mut tracker = MemoryTracker::new();

tracker.start();
let triples = parser.parse_file("data.ttl")?;
tracker.stop();

println!("Peak memory: {:.2} MB", tracker.peak_mb());
println!("Average memory: {:.2} MB", tracker.average_mb());
```

## I/O Optimization

### Buffer Size Tuning

Optimize read buffer size for your storage:

```rust
use std::io::BufReader;

// SSD - default 64KB is optimal
let reader = BufReader::with_capacity(64 * 1024, file);

// HDD - larger buffers reduce seeks
let reader = BufReader::with_capacity(512 * 1024, file);

// Network - very large buffers reduce round-trips
let reader = BufReader::with_capacity(2 * 1024 * 1024, file);
```

### Direct I/O (Linux)

For very large files, use direct I/O to bypass page cache:

```rust
use std::os::unix::fs::OpenOptionsExt;

let file = OpenOptions::new()
    .read(true)
    .custom_flags(libc::O_DIRECT)
    .open("data.ttl")?;
```

### Async I/O

Use async I/O for network sources:

```rust
use oxirs_ttl::async_parser::AsyncStreamingParser;

let response = reqwest::get("https://example.org/data.ttl").await?;
let parser = AsyncStreamingParser::new(response);

// Overlaps network I/O with parsing
```

### Read-Ahead

Enable read-ahead for sequential access:

```bash
# Linux
sudo blockdev --setra 8192 /dev/sda

# Or use
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Parallel Processing

### Parallel Streaming

Process batches in parallel:

```rust
use oxirs_ttl::parallel::ParallelStreamingParser;

let file = File::open("large.ttl")?;
let parser = ParallelStreamingParser::new(file, 8)?;  // 8 threads

let triples = parser.collect_all()?;
// ~4-6x faster on 8-core systems
```

### Manual Parallelization

Fine-grained control over parallelization:

```rust
use rayon::prelude::*;

let file = File::open("data.ttl")?;
let parser = StreamingParser::new(file);

// Collect batches
let batches: Vec<_> = parser.batches()
    .collect::<Result<Vec<_>, _>>()?;

// Process in parallel
batches.par_iter().for_each(|batch| {
    process_batch(batch);
});
```

### Thread Pool Configuration

Optimize thread pool for your workload:

```rust
use rayon::ThreadPoolBuilder;

// Build custom thread pool
let pool = ThreadPoolBuilder::new()
    .num_threads(8)
    .stack_size(4 * 1024 * 1024)  // 4MB stack
    .build()?;

pool.install(|| {
    // Parallel parsing happens here
    let parser = ParallelStreamingParser::new(file, 8)?;
    parser.collect_all()
})?;
```

### Parallel Format Conversion

Convert formats in parallel:

```rust
use rayon::prelude::*;

let files = vec!["file1.ttl", "file2.ttl", "file3.ttl"];

files.par_iter().for_each(|&input| {
    let output = input.replace(".ttl", ".nt");
    convert_turtle_to_ntriples(input, &output).unwrap();
});
```

## Platform-Specific Tuning

### macOS Optimizations

```bash
# Use native CPU features
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Enable LTO
cargo build --release --config 'profile.release.lto=true'
```

### Linux Optimizations

```bash
# Huge pages for large datasets
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages

# Disable ASLR for consistent benchmarking
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Windows Optimizations

```powershell
# Disable Windows Defender for benchmark directory
Add-MpPreference -ExclusionPath "C:\path\to\benchmark"

# High priority
Start-Process -FilePath ".\target\release\parser.exe" -Priority High
```

### SIMD on ARM (M1/M2)

Enable NEON SIMD on Apple Silicon:

```bash
RUSTFLAGS="-C target-feature=+neon" cargo build --release
```

## Benchmarking

### Criterion Benchmarks

Create rigorous benchmarks with Criterion:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxirs_ttl::turtle::TurtleParser;

fn benchmark_turtle_parsing(c: &mut Criterion) {
    let data = std::fs::read_to_string("benchmark_data.ttl").unwrap();

    c.bench_function("turtle_parse_10k", |b| {
        b.iter(|| {
            let parser = TurtleParser::new();
            black_box(parser.parse_document(&data).unwrap())
        });
    });
}

criterion_group!(benches, benchmark_turtle_parsing);
criterion_main!(benches);
```

Run benchmarks:
```bash
cargo bench
```

### Comparative Benchmarking

Compare different configurations:

```rust
fn benchmark_configurations(c: &mut Criterion) {
    let data = load_test_data();

    let mut group = c.benchmark_group("parse_configurations");

    // Baseline
    group.bench_function("default", |b| {
        b.iter(|| {
            let parser = TurtleParser::new();
            parser.parse_document(&data).unwrap()
        });
    });

    // Zero-copy
    group.bench_function("zero_copy", |b| {
        b.iter(|| {
            let parser = TurtleParser::new().with_zero_copy(true);
            parser.parse_document(&data).unwrap()
        });
    });

    // String interning
    group.bench_function("interning", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            let parser = TurtleParser::new().with_interner(interner);
            parser.parse_document(&data).unwrap()
        });
    });

    group.finish();
}
```

### Performance Regression Tests

Track performance over time:

```rust
#[test]
fn test_parse_performance_baseline() {
    let data = generate_test_data(10_000);
    let start = Instant::now();

    let parser = TurtleParser::new();
    let triples = parser.parse_document(&data).unwrap();

    let elapsed = start.elapsed();

    assert_eq!(triples.len(), 10_000);
    assert!(
        elapsed.as_millis() < 100,
        "Performance regression: {}ms",
        elapsed.as_millis()
    );
}
```

## Performance Checklist

### Parsing Optimization

- [ ] Use appropriate parser for format (N-Triples fastest)
- [ ] Enable zero-copy mode for trusted data
- [ ] Configure string interning for repeated IRIs
- [ ] Enable lazy IRI resolution when possible
- [ ] Use streaming for datasets >10MB
- [ ] Optimize batch size (10K-100K triples)
- [ ] Use parallel processing for CPU-bound workloads
- [ ] Configure appropriate buffer sizes
- [ ] Profile with `TtlProfiler` to identify bottlenecks

### Serialization Optimization

- [ ] Use optimized serialization mode
- [ ] Enable auto-generated prefixes
- [ ] Use streaming for large outputs
- [ ] Consider parallel serialization
- [ ] Profile serialization performance

### Memory Optimization

- [ ] Use streaming to avoid loading full dataset
- [ ] Enable buffer pooling
- [ ] Monitor peak memory usage
- [ ] Process and discard batches immediately
- [ ] Use memory-mapped files for very large datasets

### I/O Optimization

- [ ] Use buffered readers (64-512KB)
- [ ] Optimize for storage type (SSD vs HDD)
- [ ] Use async I/O for network sources
- [ ] Enable read-ahead for sequential access

### System Optimization

- [ ] Compile with `--release`
- [ ] Use LTO for maximum optimization
- [ ] Enable native CPU features
- [ ] Disable unnecessary services during benchmarking
- [ ] Use appropriate CPU governor (performance mode)

## Troubleshooting Performance Issues

### Slow Parsing

**Symptoms**: Lower than expected throughput

**Diagnosis**:
1. Profile with `TtlProfiler`
2. Check if I/O-bound (high iowait) or CPU-bound (high CPU usage)
3. Verify SIMD is enabled (check compilation)

**Solutions**:
- I/O-bound: Increase buffer size, use faster storage
- CPU-bound: Enable parallel processing, optimize code path
- Mixed: Use async I/O with parallel processing

### High Memory Usage

**Symptoms**: Process using excessive RAM

**Diagnosis**:
1. Profile with `valgrind massif`
2. Check if accumulating triples in memory
3. Monitor memory during streaming

**Solutions**:
- Use streaming with smaller batch sizes
- Process and discard batches immediately
- Enable buffer pooling
- Check for memory leaks (run memory leak tests)

### Poor Scaling

**Symptoms**: Adding more CPUs doesn't improve performance

**Diagnosis**:
1. Check for lock contention (use profiler)
2. Measure actual parallelism (htop, Activity Monitor)
3. Verify data is being partitioned correctly

**Solutions**:
- Reduce lock contention (use lock-free data structures)
- Increase batch sizes to reduce overhead
- Optimize thread pool configuration
- Check for sequential bottlenecks

## See Also

- [Streaming Tutorial](STREAMING_TUTORIAL.md) - Memory-efficient processing
- [Async Usage Guide](ASYNC_GUIDE.md) - Non-blocking I/O
- [Criterion Documentation](https://bheisler.github.io/criterion.rs/book/) - Benchmarking
- [API Documentation](https://docs.rs/oxirs-ttl) - Full API reference
