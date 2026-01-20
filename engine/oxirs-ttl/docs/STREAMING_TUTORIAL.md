# Streaming Tutorial - OxiRS TTL

This guide covers memory-efficient streaming for parsing and serializing large RDF datasets.

## Table of Contents

- [Why Streaming?](#why-streaming)
- [Basic Streaming](#basic-streaming)
- [Advanced Configuration](#advanced-configuration)
- [Progress Tracking](#progress-tracking)
- [Error Handling in Streams](#error-handling-in-streams)
- [Performance Optimization](#performance-optimization)
- [Real-World Examples](#real-world-examples)

## Why Streaming?

Traditional parsing loads entire RDF documents into memory, which can be problematic for large datasets:

- **Memory limits**: A 10GB Turtle file would require ~30GB RAM for in-memory parsing
- **Latency**: Users wait for the entire file to parse before processing begins
- **Scalability**: Cannot process datasets larger than available RAM

Streaming solves these issues by processing RDF in batches:

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};
use std::fs::File;

// Parse 10GB file with only ~100MB RAM usage
let file = File::open("massive_dataset.ttl")?;
let config = StreamingConfig::default()
    .with_batch_size(10_000);  // Process 10K triples at a time

let parser = StreamingParser::with_config(file, config);

for batch in parser.batches() {
    let triples = batch?;
    // Process batch (e.g., insert into database)
    database.insert_batch(&triples)?;
}
```

## Basic Streaming

### Streaming Turtle Files

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};
use std::fs::File;

fn stream_turtle_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let parser = StreamingParser::new(file);

    let mut total = 0;
    for batch in parser.batches() {
        let triples = batch?;
        total += triples.len();
        println!("Processed batch of {} triples", triples.len());
    }

    println!("Total: {} triples", total);
    Ok(())
}
```

### Streaming N-Triples

N-Triples is ideal for streaming due to its line-based format:

```rust
use oxirs_ttl::ntriples::NTriplesParser;
use std::io::{BufReader, BufRead};
use std::fs::File;

fn stream_ntriples(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let parser = NTriplesParser::new();

    // Process line by line
    for line in reader.lines() {
        let line = line?;
        if let Ok(triple) = parser.parse_line(&line) {
            // Process immediately - zero memory accumulation
            process_triple(&triple)?;
        }
    }

    Ok(())
}
```

### Streaming TriG (Named Graphs)

```rust
use oxirs_ttl::trig::TriGParser;
use std::fs::File;

fn stream_trig(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let parser = TriGParser::new();

    let mut graph_stats = std::collections::HashMap::new();

    for result in parser.parse_streaming(file) {
        let quad = result?;

        // Track statistics per named graph
        let count = graph_stats
            .entry(quad.graph_name.clone())
            .or_insert(0);
        *count += 1;
    }

    for (graph, count) in graph_stats {
        println!("Graph {:?}: {} quads", graph, count);
    }

    Ok(())
}
```

## Advanced Configuration

### Batch Size Tuning

Choose batch size based on your use case:

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};

// Small batches - Lower memory, higher overhead
let config = StreamingConfig::default()
    .with_batch_size(1_000);  // 1K triples/batch

// Medium batches - Balanced (recommended)
let config = StreamingConfig::default()
    .with_batch_size(10_000);  // 10K triples/batch

// Large batches - Better throughput, more memory
let config = StreamingConfig::default()
    .with_batch_size(100_000);  // 100K triples/batch
```

**Guidelines**:
- **Small datasets (<1M triples)**: 10K batch size
- **Medium datasets (1M-100M)**: 50K batch size
- **Large datasets (>100M)**: 100K batch size
- **Memory-constrained**: 1K-5K batch size

### Buffer Size Optimization

Control read buffer size for I/O optimization:

```rust
let config = StreamingConfig::default()
    .with_batch_size(10_000)
    .with_buffer_size(128 * 1024);  // 128KB read buffer (default: 64KB)
```

**Recommendations**:
- **SSD storage**: 64-128KB (default is optimal)
- **HDD storage**: 256-512KB (reduce seek overhead)
- **Network storage**: 1MB+ (reduce network round-trips)

### Memory Limits

Set hard memory limits for streaming:

```rust
let config = StreamingConfig::default()
    .with_batch_size(10_000)
    .with_max_memory_bytes(100 * 1024 * 1024);  // 100MB limit
```

The parser will automatically reduce batch size if memory pressure is detected.

## Progress Tracking

### Built-in Progress Callback

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};
use std::fs::File;

let config = StreamingConfig::default()
    .with_batch_size(10_000)
    .with_progress_callback(Box::new(|stats| {
        println!(
            "Progress: {} triples, {:.2} MB, {:.1} triples/sec",
            stats.triples_parsed,
            stats.bytes_read as f64 / 1_048_576.0,
            stats.throughput
        );
    }));

let file = File::open("large.ttl")?;
let parser = StreamingParser::with_config(file, config);

for batch in parser.batches() {
    let triples = batch?;
    // Process batch...
}
```

### Custom Progress Tracking

```rust
use oxirs_ttl::StreamingParser;
use std::time::Instant;

let file = File::open("dataset.ttl")?;
let parser = StreamingParser::new(file);

let start = Instant::now();
let mut total = 0;
let mut batch_count = 0;

for batch in parser.batches() {
    let triples = batch?;
    total += triples.len();
    batch_count += 1;

    // Custom progress every 10 batches
    if batch_count % 10 == 0 {
        let elapsed = start.elapsed().as_secs_f64();
        let rate = total as f64 / elapsed;
        println!(
            "Batch {}: {} total triples ({:.0} triples/sec)",
            batch_count, total, rate
        );
    }
}
```

### Progress Bar Integration

Using the `indicatif` crate for visual progress:

```rust
use oxirs_ttl::StreamingParser;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;

let file = fs::File::open("dataset.ttl")?;
let file_size = fs::metadata("dataset.ttl")?.len();

// Create progress bar based on file size
let pb = ProgressBar::new(file_size);
pb.set_style(
    ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40} {bytes}/{total_bytes} {msg}")
        .unwrap()
);

let parser = StreamingParser::new(file);
let mut bytes_read = 0;

for batch in parser.batches() {
    let triples = batch?;

    // Estimate bytes read (rough approximation)
    bytes_read += triples.len() * 100;  // ~100 bytes/triple
    pb.set_position(bytes_read.min(file_size));
    pb.set_message(format!("{} triples", triples.len()));
}

pb.finish_with_message("Parsing complete");
```

## Error Handling in Streams

### Fail-Fast Mode (Default)

Stop on first error:

```rust
let parser = StreamingParser::new(file);

for batch in parser.batches() {
    match batch {
        Ok(triples) => {
            // Process valid batch
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
            break;  // Stop processing
        }
    }
}
```

### Lenient Mode (Error Recovery)

Continue parsing despite errors:

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};

let config = StreamingConfig::default()
    .with_error_recovery(true);

let parser = StreamingParser::with_config(file, config);

let mut valid_triples = 0;
let mut errors = 0;

for batch in parser.batches() {
    match batch {
        Ok(triples) => {
            valid_triples += triples.len();
        }
        Err(e) => {
            errors += 1;
            eprintln!("Batch error (continuing): {}", e);
        }
    }
}

println!("Parsed {} triples with {} errors", valid_triples, errors);
```

### Collecting All Errors

```rust
use oxirs_ttl::{StreamingParser, TurtleParseError};

let parser = StreamingParser::new(file);
let mut all_errors = Vec::new();
let mut all_triples = Vec::new();

for batch in parser.batches() {
    match batch {
        Ok(triples) => all_triples.extend(triples),
        Err(e) => all_errors.push(e),
    }
}

if !all_errors.is_empty() {
    eprintln!("Encountered {} errors:", all_errors.len());
    for (i, err) in all_errors.iter().enumerate().take(10) {
        eprintln!("  Error {}: {}", i + 1, err);
    }
}

println!("Successfully parsed {} triples", all_triples.len());
```

## Performance Optimization

### Parallel Batch Processing

Process batches in parallel using rayon (requires `parallel` feature):

```rust
use oxirs_ttl::parallel::ParallelStreamingParser;
use rayon::prelude::*;
use std::fs::File;

let file = File::open("large.ttl")?;
let parser = ParallelStreamingParser::new(file, 4)?;  // 4 threads

// Collect all batches
let batches: Vec<_> = parser.collect_batches()?;

// Process in parallel
batches.par_iter().for_each(|batch| {
    // Thread-safe processing
    database.insert_batch(batch).expect("Insert failed");
});
```

### Zero-Copy Optimization

Enable zero-copy parsing for better performance:

```rust
use oxirs_ttl::StreamingConfig;

let config = StreamingConfig::default()
    .with_batch_size(10_000)
    .with_zero_copy(true);  // Use Cow<str> for IRIs

// 20-30% performance improvement for large datasets
```

### Prefetch and Pipeline

Overlap I/O and processing using channels:

```rust
use oxirs_ttl::StreamingParser;
use std::sync::mpsc;
use std::thread;

let (tx, rx) = mpsc::channel();

// Parser thread
let parser_handle = thread::spawn(move || {
    let file = File::open("dataset.ttl").unwrap();
    let parser = StreamingParser::new(file);

    for batch in parser.batches() {
        if tx.send(batch).is_err() {
            break;  // Receiver dropped
        }
    }
});

// Processing thread
for batch_result in rx {
    let batch = batch_result?;
    // Process while next batch is being parsed
    process_batch(&batch)?;
}

parser_handle.join().unwrap();
```

## Real-World Examples

### Example 1: Database Import

Import large RDF file into PostgreSQL:

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};
use postgres::{Client, NoTls};
use std::fs::File;

fn import_to_postgres(path: &str, db_url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = Client::connect(db_url, NoTls)?;

    // Prepare statement
    let stmt = client.prepare(
        "INSERT INTO triples (subject, predicate, object) VALUES ($1, $2, $3)"
    )?;

    let file = File::open(path)?;
    let config = StreamingConfig::default()
        .with_batch_size(10_000)
        .with_progress_callback(Box::new(|stats| {
            println!("Imported {} triples", stats.triples_parsed);
        }));

    let parser = StreamingParser::with_config(file, config);

    for batch in parser.batches() {
        let triples = batch?;

        // Begin transaction for batch
        let mut transaction = client.transaction()?;

        for triple in triples {
            transaction.execute(
                &stmt,
                &[&triple.subject.to_string(),
                  &triple.predicate.to_string(),
                  &triple.object.to_string()]
            )?;
        }

        transaction.commit()?;
    }

    Ok(())
}
```

### Example 2: RDF Format Conversion

Convert Turtle to N-Quads with streaming:

```rust
use oxirs_ttl::{StreamingParser, nquads::NQuadsSerializer};
use std::fs::File;
use std::io::BufWriter;

fn convert_turtle_to_nquads(
    input: &str,
    output: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let input_file = File::open(input)?;
    let output_file = File::create(output)?;
    let mut writer = BufWriter::new(output_file);

    let parser = StreamingParser::new(input_file);
    let serializer = NQuadsSerializer::new();

    for batch in parser.batches() {
        let triples = batch?;

        // Convert to quads (default graph)
        let quads: Vec<_> = triples
            .into_iter()
            .map(|t| t.into_quad(None))  // None = default graph
            .collect();

        serializer.serialize_batch(&quads, &mut writer)?;
    }

    Ok(())
}
```

### Example 3: Statistical Analysis

Compute statistics on large RDF dataset:

```rust
use oxirs_ttl::StreamingParser;
use std::collections::HashMap;

struct RdfStats {
    total_triples: usize,
    unique_subjects: HashMap<String, usize>,
    unique_predicates: HashMap<String, usize>,
    predicate_usage: HashMap<String, usize>,
}

fn analyze_rdf_dataset(path: &str) -> Result<RdfStats, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let parser = StreamingParser::new(file);

    let mut stats = RdfStats {
        total_triples: 0,
        unique_subjects: HashMap::new(),
        unique_predicates: HashMap::new(),
        predicate_usage: HashMap::new(),
    };

    for batch in parser.batches() {
        let triples = batch?;

        for triple in triples {
            stats.total_triples += 1;

            let subject = triple.subject.to_string();
            *stats.unique_subjects.entry(subject).or_insert(0) += 1;

            let predicate = triple.predicate.to_string();
            *stats.unique_predicates.entry(predicate.clone()).or_insert(0) += 1;
            *stats.predicate_usage.entry(predicate).or_insert(0) += 1;
        }
    }

    Ok(stats)
}

// Usage
let stats = analyze_rdf_dataset("large_dataset.ttl")?;
println!("Total triples: {}", stats.total_triples);
println!("Unique subjects: {}", stats.unique_subjects.len());
println!("Unique predicates: {}", stats.unique_predicates.len());

// Top 10 most used predicates
let mut usage: Vec<_> = stats.predicate_usage.iter().collect();
usage.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
for (pred, count) in usage.iter().take(10) {
    println!("  {}: {} uses", pred, count);
}
```

### Example 4: Memory-Constrained Environment

Process huge dataset on limited memory system (e.g., 512MB RAM):

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};

fn process_on_limited_memory(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = StreamingConfig::default()
        .with_batch_size(1_000)  // Small batches
        .with_buffer_size(32 * 1024)  // 32KB read buffer
        .with_max_memory_bytes(50 * 1024 * 1024);  // 50MB limit

    let file = File::open(path)?;
    let parser = StreamingParser::with_config(file, config);

    for batch in parser.batches() {
        let triples = batch?;

        // Process immediately - don't accumulate
        for triple in triples {
            process_single_triple(&triple)?;
        }

        // Explicit memory cleanup hint
        drop(batch);
    }

    Ok(())
}
```

## Best Practices

1. **Choose appropriate batch size** based on available memory and dataset size
2. **Enable progress tracking** for long-running operations
3. **Use error recovery** for production systems handling untrusted data
4. **Profile memory usage** with tools like `valgrind` or `heaptrack`
5. **Consider parallel processing** for CPU-bound post-processing
6. **Use zero-copy mode** when working with trusted, well-formed data
7. **Implement backpressure** when streaming to slower downstream systems
8. **Monitor throughput** and adjust configuration based on metrics

## Performance Benchmarks

Streaming vs. in-memory parsing on a 1GB Turtle file:

| Method | Memory Usage | Parse Time | Throughput |
|--------|--------------|------------|------------|
| In-memory | ~3.2 GB | 8.5s | 310K triples/s |
| Streaming (1K batch) | ~45 MB | 9.2s | 287K triples/s |
| Streaming (10K batch) | ~120 MB | 8.7s | 303K triples/s |
| Streaming (100K batch) | ~850 MB | 8.4s | 314K triples/s |

**Recommendation**: Use 10K-50K batch size for optimal memory/performance trade-off.

## Troubleshooting

### "Out of memory" errors

- Reduce batch size
- Enable max_memory_bytes limit
- Use streaming serialization for output
- Process and discard batches immediately

### Slow parsing

- Increase batch size (if memory allows)
- Increase buffer size for I/O-bound workloads
- Use parallel processing for CPU-bound workloads
- Profile with `TtlProfiler` to identify bottlenecks

### Parse errors in middle of file

- Enable error recovery mode
- Log errors for later review
- Consider validating entire file first in lenient mode

## See Also

- [Async Usage Guide](ASYNC_GUIDE.md) - Non-blocking async I/O
- [Performance Tuning Guide](PERFORMANCE_GUIDE.md) - Advanced optimization
- [API Documentation](https://docs.rs/oxirs-ttl) - Full API reference
