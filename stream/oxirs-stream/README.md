# OxiRS Stream - Real-time RDF Streaming

[![Version](https://img.shields.io/badge/version-0.1.0--alpha.3-orange)](https://github.com/cool-japan/oxirs/releases)

**Status**: Alpha Release (v0.1.0-alpha.3) - Released October 12, 2025

⚠️ **Alpha Software**: This is an early alpha release. Experimental features. APIs may change without notice. Not recommended for production use.

Real-time RDF data streaming with support for Kafka, NATS, and other message brokers. Process RDF streams with windowing, aggregation, and pattern matching.

## Features

### Message Brokers
- **Apache Kafka** - Distributed streaming platform
- **NATS** - Lightweight, high-performance messaging
- **RabbitMQ** - Reliable message queuing
- **Custom Adapters** - Bring your own message broker

### Stream Processing
- **Windowing** - Tumbling, sliding, and session windows
- **Aggregation** - Count, sum, average over windows
- **Pattern Matching** - Detect patterns in RDF streams
- **Filtering** - Stream-based SPARQL filters

### Features
- **At-Least-Once Delivery** - Reliable message processing
- **Backpressure** - Handle fast producers
- **Checkpointing** - Resume from failures
- **Metrics** - Monitor stream performance

## Installation

Add to your `Cargo.toml`:

```toml
# Experimental feature
[dependencies]
oxirs-stream = "0.1.0-alpha.3"

# Enable specific brokers
oxirs-stream = { version = "0.1.0-alpha.3", features = ["kafka", "nats"] }
```

## Quick Start

### Basic Streaming

```rust
use oxirs_stream::{StreamSource, KafkaConfig};
use oxirs_core::Triple;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure Kafka source
    let config = KafkaConfig {
        bootstrap_servers: vec!["localhost:9092".to_string()],
        topic: "rdf-triples".to_string(),
        group_id: "oxirs-consumer".to_string(),
        ..Default::default()
    };

    // Create stream
    let mut stream = StreamSource::kafka(config).await?;

    // Process triples
    while let Some(triple) = stream.next().await {
        let triple = triple?;
        println!("{} {} {}", triple.subject, triple.predicate, triple.object);

        // Process triple...
    }

    Ok(())
}
```

### Stream Processing with Windows

```rust
use oxirs_stream::{StreamProcessor, WindowConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let processor = StreamProcessor::builder()
        .source(kafka_source)
        .window(WindowConfig::tumbling(Duration::from_secs(60)))
        .build()?;

    // Process windowed batches
    let mut windows = processor.process().await?;

    while let Some(window) = windows.next().await {
        let triples = window?;
        println!("Window received {} triples", triples.len());

        // Aggregate, validate, or process batch
        process_window(triples)?;
    }

    Ok(())
}
```

## Message Broker Configuration

### Kafka

```rust
use oxirs_stream::KafkaConfig;

let config = KafkaConfig {
    bootstrap_servers: vec!["kafka1:9092".to_string(), "kafka2:9092".to_string()],
    topic: "rdf-events".to_string(),
    group_id: "my-consumer-group".to_string(),

    // Performance tuning
    fetch_min_bytes: 1024,
    fetch_max_wait_ms: 500,
    max_partition_fetch_bytes: 1048576,

    // Reliability
    enable_auto_commit: false,
    auto_commit_interval_ms: 5000,

    // Security
    security_protocol: Some("SASL_SSL".to_string()),
    sasl_mechanism: Some("PLAIN".to_string()),
    sasl_username: Some(std::env::var("KAFKA_USERNAME")?),
    sasl_password: Some(std::env::var("KAFKA_PASSWORD")?),
};
```

### NATS

```rust
use oxirs_stream::NatsConfig;

let config = NatsConfig {
    servers: vec!["nats://localhost:4222".to_string()],
    subject: "rdf.>".to_string(),  // Wildcard subscription
    queue_group: Some("oxirs-processors".to_string()),

    // Credentials
    credentials_path: Some("./nats.creds".into()),

    // JetStream (persistent)
    use_jetstream: true,
    stream_name: Some("RDF_STREAM".to_string()),
    durable_name: Some("oxirs-consumer".to_string()),
};
```

## Windowing

### Tumbling Windows

Fixed-size, non-overlapping windows:

```rust
use oxirs_stream::{WindowConfig, WindowType};
use std::time::Duration;

let config = WindowConfig {
    window_type: WindowType::Tumbling,
    size: Duration::from_secs(60),
    ..Default::default()
};

// Process 60-second windows
```

### Sliding Windows

Overlapping windows:

```rust
let config = WindowConfig {
    window_type: WindowType::Sliding,
    size: Duration::from_secs(60),
    slide: Duration::from_secs(30),  // 30-second slide
    ..Default::default()
};

// Windows: [0-60s], [30-90s], [60-120s], ...
```

### Session Windows

Dynamic windows based on inactivity gaps:

```rust
let config = WindowConfig {
    window_type: WindowType::Session,
    gap: Duration::from_secs(300),  // 5-minute inactivity closes window
    ..Default::default()
};
```

## Stream Operations

### Filtering

```rust
use oxirs_stream::filters::SparqlFilter;

let filter = SparqlFilter::new(r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    FILTER EXISTS {
        ?s a foaf:Person .
        ?s foaf:age ?age .
        FILTER (?age >= 18)
    }
"#)?;

let filtered_stream = stream.filter(filter);
```

### Mapping

```rust
let transformed_stream = stream.map(|triple| {
    // Transform each triple
    transform_triple(triple)
});
```

### Aggregation

```rust
use oxirs_stream::aggregation::{Count, Sum, Average};

let processor = StreamProcessor::builder()
    .source(source)
    .window(WindowConfig::tumbling(Duration::from_secs(60)))
    .aggregate(Count::new("?person", "foaf:Person"))
    .aggregate(Average::new("?age", "foaf:age"))
    .build()?;

let results = processor.process().await?;
```

## Pattern Matching

### Temporal Patterns

```rust
use oxirs_stream::patterns::TemporalPattern;

let pattern = TemporalPattern::builder()
    .event("A", "?person foaf:login ?time")
    .followed_by("B", "?person foaf:logout ?time2", Duration::from_secs(3600))
    .within(Duration::from_hours(24))
    .build()?;

let matches = stream.detect_pattern(pattern).await?;
```

### Graph Patterns

```rust
use oxirs_stream::patterns::GraphPattern;

let pattern = GraphPattern::parse(r#"
    {
        ?person a foaf:Person .
        ?person foaf:knows ?friend .
        ?friend foaf:age ?age .
        FILTER (?age > 18)
    }
"#)?;

let matches = stream.match_pattern(pattern).await?;
```

## Reliability

### Checkpointing

```rust
use oxirs_stream::checkpoint::CheckpointConfig;

let checkpoint_config = CheckpointConfig {
    interval: Duration::from_secs(60),
    storage: CheckpointStorage::File("./checkpoints".into()),
    max_failures: 3,
};

let processor = StreamProcessor::builder()
    .source(source)
    .checkpoint(checkpoint_config)
    .build()?;

// Automatically recovers from last checkpoint on failure
```

### Error Handling

```rust
use oxirs_stream::error_handling::{ErrorPolicy, RetryPolicy};

let error_policy = ErrorPolicy {
    retry: RetryPolicy::exponential_backoff(3),
    dead_letter_topic: Some("rdf-errors".to_string()),
    log_errors: true,
};

let processor = StreamProcessor::builder()
    .source(source)
    .error_policy(error_policy)
    .build()?;
```

## Integration

### With oxirs-shacl (Streaming Validation)

```rust
use oxirs_stream::StreamProcessor;
use oxirs_shacl::ValidationEngine;

let validator = ValidationEngine::new(&shapes, config);

let processor = StreamProcessor::builder()
    .source(kafka_source)
    .window(WindowConfig::tumbling(Duration::from_secs(10)))
    .validate_with(validator)
    .build()?;

let mut results = processor.process().await?;

while let Some(window_result) = results.next().await {
    let (triples, validation_report) = window_result?;

    if !validation_report.conforms {
        eprintln!("Validation failed: {} violations",
            validation_report.violations.len());
    }
}
```

### With oxirs-arq (Stream Queries)

```rust
use oxirs_stream::StreamProcessor;
use oxirs_arq::StreamingQueryEngine;

let query_engine = StreamingQueryEngine::new();

let query = r#"
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT ?person (COUNT(?friend) as ?friendCount)
    WHERE {
        ?person a foaf:Person .
        ?person foaf:knows ?friend .
    }
    GROUP BY ?person
    HAVING (COUNT(?friend) > 10)
"#;

let processor = StreamProcessor::builder()
    .source(source)
    .window(WindowConfig::tumbling(Duration::from_secs(60)))
    .query(query_engine, query)
    .build()?;
```

## Performance

### Throughput Benchmarks

| Message Broker | Throughput | Latency (p99) |
|---------------|------------|---------------|
| Kafka | 100K triples/s | 15ms |
| NATS | 80K triples/s | 8ms |
| RabbitMQ | 50K triples/s | 20ms |

*Benchmarked on M1 Mac with local brokers*

### Optimization Tips

```rust
// Batch processing
let processor = StreamProcessor::builder()
    .source(source)
    .batch_size(1000)  // Process in batches of 1000
    .parallelism(4)    // 4 parallel workers
    .build()?;

// Backpressure control
let processor = StreamProcessor::builder()
    .source(source)
    .buffer_size(10000)
    .backpressure_strategy(BackpressureStrategy::Block)
    .build()?;
```

## Status

### Alpha Release (v0.1.0-alpha.3)
- ✅ Kafka/NATS integrations with persisted offset checkpoints
- ✅ Windowing, filtering, and mapping tied into CLI persistence workflows
- ✅ SPARQL stream federation with `SERVICE` bridging to remote endpoints
- ✅ Prometheus/SciRS2 metrics for throughput, lag, and error rates
- 🚧 Aggregation operators (tumbling/sliding) final polish (in progress)
- 🚧 Pattern matching DSL and CEP (in progress)
- ⏳ Exactly-once semantics (planned for beta)
- ⏳ Distributed stream processing (planned for v0.2.0)

## Contributing

This is an experimental module. Feedback welcome!

## License

MIT OR Apache-2.0

## See Also

- [oxirs-shacl](../../engine/oxirs-shacl/) - Stream validation
- [oxirs-arq](../../engine/oxirs-arq/) - Stream queries
- [oxirs-federate](../oxirs-federate/) - Federated streams