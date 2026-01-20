# Async Usage Guide - OxiRS TTL

This guide covers asynchronous, non-blocking RDF parsing and serialization using Tokio.

## Table of Contents

- [Why Async?](#why-async)
- [Setup](#setup)
- [Basic Async Parsing](#basic-async-parsing)
- [Async Streaming](#async-streaming)
- [Concurrent Parsing](#concurrent-parsing)
- [Network Integration](#network-integration)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Real-World Examples](#real-world-examples)

## Why Async?

Asynchronous I/O is essential for:

- **Web servers**: Parse RDF from HTTP requests without blocking
- **Network clients**: Fetch and parse RDF from remote sources
- **Concurrent operations**: Parse multiple files simultaneously
- **Microservices**: Non-blocking integration with async frameworks

**Performance benefits**:
- Handle thousands of concurrent connections
- Efficient resource utilization (CPU + I/O overlap)
- Low latency for high-throughput applications

## Setup

Add the `async-tokio` feature to your `Cargo.toml`:

```toml
[dependencies]
oxirs-ttl = { version = "0.1.0", features = ["async-tokio"] }
tokio = { version = "1", features = ["full"] }
```

## Basic Async Parsing

### Async Turtle Parsing

```rust
use oxirs_ttl::async_parser::AsyncTurtleParser;
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open file asynchronously
    let file = File::open("data.ttl").await?;

    // Parse with async parser
    let parser = AsyncTurtleParser::new();
    let triples = parser.parse_async(file).await?;

    println!("Parsed {} triples", triples.len());
    Ok(())
}
```

### Async N-Triples Parsing

```rust
use oxirs_ttl::async_parser::AsyncNTriplesParser;
use tokio::fs::File;
use tokio::io::AsyncBufReadExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("data.nt").await?;
    let reader = tokio::io::BufReader::new(file);
    let parser = AsyncNTriplesParser::new();

    let mut lines = reader.lines();
    let mut count = 0;

    // Parse line by line asynchronously
    while let Some(line) = lines.next_line().await? {
        if let Ok(triple) = parser.parse_line(&line) {
            count += 1;
            // Process triple immediately
        }
    }

    println!("Parsed {} triples", count);
    Ok(())
}
```

### Async TriG Parsing

```rust
use oxirs_ttl::async_parser::AsyncTriGParser;
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("data.trig").await?;
    let parser = AsyncTriGParser::new();

    let quads = parser.parse_async(file).await?;

    for quad in quads {
        println!("Graph: {:?}, Triple: {}", quad.graph_name, quad);
    }

    Ok(())
}
```

## Async Streaming

### Async Streaming with Tokio

Stream large files asynchronously:

```rust
use oxirs_ttl::async_parser::AsyncStreamingParser;
use tokio::fs::File;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("large.ttl").await?;
    let parser = AsyncStreamingParser::new(file);

    // Get async stream of batches
    let mut stream = parser.batches();

    let mut total = 0;
    while let Some(batch) = stream.next().await {
        let triples = batch?;
        total += triples.len();

        // Async processing
        async_process_batch(&triples).await?;
    }

    println!("Total: {} triples", total);
    Ok(())
}

async fn async_process_batch(
    triples: &[Triple]
) -> Result<(), Box<dyn std::error::Error>> {
    // Async database insert, network call, etc.
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    Ok(())
}
```

### Configuring Async Streaming

```rust
use oxirs_ttl::{AsyncStreamingParser, StreamingConfig};

let config = StreamingConfig::default()
    .with_batch_size(10_000)
    .with_buffer_size(128 * 1024);  // 128KB async read buffer

let file = File::open("data.ttl").await?;
let parser = AsyncStreamingParser::with_config(file, config);

let mut stream = parser.batches();
while let Some(batch) = stream.next().await {
    let triples = batch?;
    // Process asynchronously...
}
```

## Concurrent Parsing

### Parse Multiple Files Concurrently

```rust
use oxirs_ttl::async_parser::AsyncTurtleParser;
use tokio::fs::File;
use futures::future::join_all;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files = vec!["file1.ttl", "file2.ttl", "file3.ttl"];

    // Parse all files concurrently
    let parse_tasks: Vec<_> = files
        .into_iter()
        .map(|path| async move {
            let file = File::open(path).await?;
            let parser = AsyncTurtleParser::new();
            parser.parse_async(file).await
        })
        .collect();

    let results = join_all(parse_tasks).await;

    let mut total = 0;
    for (i, result) in results.into_iter().enumerate() {
        match result {
            Ok(triples) => {
                println!("File {}: {} triples", i, triples.len());
                total += triples.len();
            }
            Err(e) => eprintln!("File {}: Error: {}", i, e),
        }
    }

    println!("Total across all files: {} triples", total);
    Ok(())
}
```

### Concurrent Streaming with Tokio Tasks

```rust
use oxirs_ttl::async_parser::AsyncStreamingParser;
use tokio::task;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files = vec!["large1.ttl", "large2.ttl", "large3.ttl"];

    let mut handles = vec![];

    for path in files {
        let handle = task::spawn(async move {
            let file = File::open(path).await.unwrap();
            let parser = AsyncStreamingParser::new(file);
            let mut stream = parser.batches();

            let mut count = 0;
            while let Some(batch) = stream.next().await {
                let triples = batch.unwrap();
                count += triples.len();
            }

            (path, count)
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let (path, count) = handle.await?;
        println!("{}: {} triples", path, count);
    }

    Ok(())
}
```

### Rate-Limited Concurrent Parsing

Control concurrency with semaphores:

```rust
use tokio::sync::Semaphore;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files: Vec<String> = (0..100)
        .map(|i| format!("file{}.ttl", i))
        .collect();

    // Limit to 10 concurrent parses
    let semaphore = Arc::new(Semaphore::new(10));
    let mut handles = vec![];

    for path in files {
        let sem = semaphore.clone();

        let handle = task::spawn(async move {
            // Acquire permit (blocks if 10 tasks already running)
            let _permit = sem.acquire().await.unwrap();

            let file = File::open(&path).await.unwrap();
            let parser = AsyncTurtleParser::new();
            let triples = parser.parse_async(file).await.unwrap();

            (path, triples.len())
        });

        handles.push(handle);
    }

    for handle in handles {
        let (path, count) = handle.await?;
        println!("{}: {} triples", path, count);
    }

    Ok(())
}
```

## Network Integration

### Fetch and Parse RDF from HTTP

```rust
use oxirs_ttl::async_parser::AsyncTurtleParser;
use tokio::io::AsyncReadExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch RDF from remote server
    let response = reqwest::get("https://example.org/data.ttl").await?;
    let bytes = response.bytes().await?;

    // Parse from bytes
    let parser = AsyncTurtleParser::new();
    let triples = parser.parse_async(&bytes[..]).await?;

    println!("Fetched and parsed {} triples", triples.len());
    Ok(())
}
```

### Streaming HTTP Response

Stream large RDF files from network:

```rust
use oxirs_ttl::async_parser::AsyncStreamingParser;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::get("https://example.org/large.ttl").await?;

    // Wrap response body in async reader
    let reader = tokio_util::io::StreamReader::new(
        response.bytes_stream().map(|r| {
            r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        })
    );

    let parser = AsyncStreamingParser::new(reader);
    let mut stream = parser.batches();

    let mut total = 0;
    while let Some(batch) = stream.next().await {
        let triples = batch?;
        total += triples.len();

        // Process while downloading continues
        async_process_batch(&triples).await?;
    }

    println!("Downloaded and parsed {} triples", total);
    Ok(())
}
```

### WebSocket RDF Streaming

Real-time RDF updates over WebSocket:

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures::{StreamExt, SinkExt};
use oxirs_ttl::async_parser::AsyncTurtleParser;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (ws_stream, _) = connect_async("ws://example.org/rdf-stream").await?;
    let (mut write, mut read) = ws_stream.split();

    let parser = AsyncTurtleParser::new();

    while let Some(msg) = read.next().await {
        let msg = msg?;

        if let Message::Text(text) = msg {
            // Parse RDF chunk
            match parser.parse_document(&text) {
                Ok(triples) => {
                    println!("Received {} triples", triples.len());
                    // Process real-time...
                }
                Err(e) => eprintln!("Parse error: {}", e),
            }
        }
    }

    Ok(())
}
```

## Error Handling

### Async Result Handling

```rust
use oxirs_ttl::async_parser::AsyncTurtleParser;
use oxirs_ttl::TurtleParseError;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("data.ttl").await?;
    let parser = AsyncTurtleParser::new();

    match parser.parse_async(file).await {
        Ok(triples) => {
            println!("Success: {} triples", triples.len());
        }
        Err(TurtleParseError::Syntax(e)) => {
            eprintln!("Syntax error at {}:{}: {}",
                e.position.line, e.position.column, e.message);
        }
        Err(TurtleParseError::Io(e)) => {
            eprintln!("I/O error: {}", e);
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
        }
    }

    Ok(())
}
```

### Timeout Handling

Add timeouts to prevent hanging:

```rust
use tokio::time::{timeout, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("data.ttl").await?;
    let parser = AsyncTurtleParser::new();

    // 30 second timeout
    match timeout(Duration::from_secs(30), parser.parse_async(file)).await {
        Ok(Ok(triples)) => {
            println!("Parsed {} triples", triples.len());
        }
        Ok(Err(e)) => {
            eprintln!("Parse error: {}", e);
        }
        Err(_) => {
            eprintln!("Parse timed out after 30 seconds");
        }
    }

    Ok(())
}
```

### Retry Logic

Implement exponential backoff for network failures:

```rust
use tokio::time::{sleep, Duration};

async fn fetch_and_parse_with_retry(
    url: &str,
    max_retries: u32
) -> Result<Vec<Triple>, Box<dyn std::error::Error>> {
    let mut retries = 0;

    loop {
        match try_fetch_and_parse(url).await {
            Ok(triples) => return Ok(triples),
            Err(e) if retries < max_retries => {
                retries += 1;
                let delay = Duration::from_secs(2_u64.pow(retries));
                eprintln!("Retry {} after {:?}: {}", retries, delay, e);
                sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

async fn try_fetch_and_parse(
    url: &str
) -> Result<Vec<Triple>, Box<dyn std::error::Error>> {
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    let parser = AsyncTurtleParser::new();
    Ok(parser.parse_async(&bytes[..]).await?)
}
```

## Performance Optimization

### Buffered Async I/O

Use proper buffering for optimal performance:

```rust
use tokio::io::BufReader;

let file = File::open("data.ttl").await?;

// Wrap in BufReader for efficient async reads
let buffered = BufReader::with_capacity(128 * 1024, file);  // 128KB buffer

let parser = AsyncStreamingParser::new(buffered);
```

### Async Batch Processing Pipeline

Overlap I/O and processing:

```rust
use tokio::sync::mpsc;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (tx, mut rx) = mpsc::channel(10);  // Buffer 10 batches

    // Parser task
    let parser_task = tokio::spawn(async move {
        let file = File::open("large.ttl").await.unwrap();
        let parser = AsyncStreamingParser::new(file);
        let mut stream = parser.batches();

        while let Some(batch) = stream.next().await {
            if tx.send(batch).await.is_err() {
                break;  // Receiver dropped
            }
        }
    });

    // Processing task
    let processor_task = tokio::spawn(async move {
        while let Some(batch) = rx.recv().await {
            let triples = batch.unwrap();
            // CPU-intensive processing happens while next batch is parsing
            process_batch(&triples).await.unwrap();
        }
    });

    // Wait for both tasks
    tokio::try_join!(parser_task, processor_task)?;
    Ok(())
}
```

### Parallel Async Operations

Combine async I/O with parallel CPU processing:

```rust
use rayon::prelude::*;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("large.ttl").await?;
    let parser = AsyncStreamingParser::new(file);
    let mut stream = parser.batches();

    while let Some(batch) = stream.next().await {
        let triples = batch?;

        // Move CPU-intensive work to thread pool
        let processed = task::spawn_blocking(move || {
            triples.par_iter()
                .map(|t| expensive_computation(t))
                .collect::<Vec<_>>()
        }).await?;

        // Continue with results...
    }

    Ok(())
}
```

## Real-World Examples

### Example 1: Async Web Service

Axum web service for RDF parsing:

```rust
use axum::{
    Router,
    extract::Multipart,
    response::IntoResponse,
    routing::post,
};
use oxirs_ttl::async_parser::AsyncTurtleParser;

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/parse", post(parse_rdf));

    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn parse_rdf(mut multipart: Multipart) -> impl IntoResponse {
    while let Some(field) = multipart.next_field().await.unwrap() {
        let data = field.bytes().await.unwrap();

        let parser = AsyncTurtleParser::new();
        match parser.parse_async(&data[..]).await {
            Ok(triples) => {
                return format!("Parsed {} triples", triples.len());
            }
            Err(e) => {
                return format!("Error: {}", e);
            }
        }
    }

    "No data received".to_string()
}
```

### Example 2: Distributed RDF Processing

Process RDF files from S3 with AWS SDK:

```rust
use aws_sdk_s3::Client;
use oxirs_ttl::async_parser::AsyncStreamingParser;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = aws_config::load_from_env().await;
    let client = Client::new(&config);

    // List objects in bucket
    let objects = client
        .list_objects_v2()
        .bucket("my-rdf-bucket")
        .send()
        .await?;

    for obj in objects.contents.unwrap_or_default() {
        let key = obj.key.unwrap();

        // Get object stream
        let output = client
            .get_object()
            .bucket("my-rdf-bucket")
            .key(&key)
            .send()
            .await?;

        // Parse streaming
        let parser = AsyncStreamingParser::new(output.body.into_async_read());
        let mut stream = parser.batches();

        let mut count = 0;
        while let Some(batch) = stream.next().await {
            let triples = batch?;
            count += triples.len();
        }

        println!("{}: {} triples", key, count);
    }

    Ok(())
}
```

### Example 3: Real-time RDF Updates

Process RDF updates from Kafka:

```rust
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::config::ClientConfig;
use futures::StreamExt;
use oxirs_ttl::async_parser::AsyncTurtleParser;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", "localhost:9092")
        .set("group.id", "rdf-parser")
        .create()?;

    consumer.subscribe(&["rdf-updates"])?;

    let parser = AsyncTurtleParser::new();
    let mut message_stream = consumer.stream();

    while let Some(message) = message_stream.next().await {
        let message = message?;

        if let Some(payload) = message.payload() {
            match parser.parse_async(payload).await {
                Ok(triples) => {
                    println!("Received update with {} triples", triples.len());
                    // Process triples...
                }
                Err(e) => eprintln!("Parse error: {}", e),
            }
        }
    }

    Ok(())
}
```

## Best Practices

1. **Use proper buffering**: Wrap file readers with `BufReader` for efficient async I/O
2. **Set timeouts**: Always use timeouts for network operations
3. **Handle backpressure**: Use bounded channels to prevent memory issues
4. **Limit concurrency**: Use semaphores to control concurrent tasks
5. **Error recovery**: Implement retry logic for transient failures
6. **Monitor performance**: Track throughput and latency metrics
7. **CPU-bound work**: Use `spawn_blocking` for CPU-intensive operations
8. **Resource cleanup**: Ensure proper cleanup with `Drop` or `finally` patterns

## Troubleshooting

### "Too many open files" error

Reduce concurrent parsing or increase file descriptor limit:

```bash
ulimit -n 4096  # Increase to 4096
```

```rust
// Limit concurrent operations
let semaphore = Arc::new(Semaphore::new(100));  // Max 100 concurrent
```

### Slow async performance

- Use `BufReader` for file I/O
- Increase buffer sizes for network I/O
- Profile with `tokio-console` to identify bottlenecks
- Consider using `spawn_blocking` for CPU-intensive work

### Memory leaks in long-running services

- Ensure proper cleanup of resources
- Use weak references for caches
- Monitor memory with tools like `valgrind`
- Consider periodic restarts for critical services

## See Also

- [Streaming Tutorial](STREAMING_TUTORIAL.md) - Memory-efficient batch processing
- [Performance Tuning Guide](PERFORMANCE_GUIDE.md) - Optimization techniques
- [Tokio Documentation](https://tokio.rs) - Async runtime documentation
- [API Documentation](https://docs.rs/oxirs-ttl) - Full API reference
