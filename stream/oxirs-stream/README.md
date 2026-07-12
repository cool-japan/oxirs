# OxiRS Stream - Real-time RDF Streaming

[![Version](https://img.shields.io/badge/version-0.3.2-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: v0.3.2 - Released 2026-07-12

✨ **Production Release**: Production-ready with API stability guarantees and comprehensive testing.

Real-time RDF data streaming with support for MQTT 5.0/3.1.1, NATS JetStream, RabbitMQ, Redis Streams, and AWS Kinesis in-tree, plus Kafka and Pulsar via separate adapter crates. Process RDF streams with windowing, aggregation, and pattern matching.

## Features

### Message Brokers
- **MQTT 5.0 / 3.1.1** - IoT and Industry 4.0 integration (Sparkplug B, OPC-UA), with a full MQTT 5.0 property codec (`backend::mqtt::properties`)
- **NATS** - Lightweight, high-performance messaging with JetStream persistence
- **RabbitMQ** - Reliable message queuing
- **AWS Kinesis / Redis Streams** - Cloud-native and in-memory backends
- **Apache Kafka / Apache Pulsar** - Quarantined out of the default build per COOLJAPAN Pure Rust Policy v2; available as the separate `oxirs-stream-adapter-rdkafka` / `oxirs-stream-adapter-pulsar` crates (`publish = false`, workspace-internal — see [Kafka & Pulsar](#kafka--pulsar-adapter-crates) below)
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
[dependencies]
oxirs-stream = "0.3.2"

# Default features enable only the in-memory backend. Turn on the backends you need:
oxirs-stream = { version = "0.3.2", features = ["mqtt", "nats"] }

# `industry40` bundles mqtt + opcua + sparkplug for Industry 4.0 deployments.
# `all-backends` bundles nats + kinesis + redis + rabbitmq + mqtt + opcua.
# Kafka and Pulsar are NOT Cargo features of this crate — see "Kafka & Pulsar" below.
```

## Quick Start

### MQTT 5.0: Connect, Publish, and Decode Properties

Requires the `mqtt` feature (`rumqttc`-backed, `use-rustls` TLS).

```rust
use oxirs_stream::backend::mqtt::properties::{decode_mqtt5_properties, encode_mqtt5_properties};
use oxirs_stream::backend::mqtt::types::MqttMessageProperties;
use oxirs_stream::backend::mqtt::{MqttClient, MqttConfig, QoS};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // MQTT 5.0 property codec: encode/decode PUBLISH-relevant properties
    // (Payload Format Indicator, Message Expiry Interval, Content Type,
    // Response Topic, Correlation Data, Subscription Identifier, Topic Alias,
    // repeatable User Properties) per the MQTT 5.0 spec (section 2.2.2).
    let props = MqttMessageProperties {
        payload_format_indicator: Some(1), // UTF-8
        message_expiry_interval: Some(3600),
        topic_alias: None,
        response_topic: Some("rdf/responses".to_string()),
        correlation_data: None,
        user_properties: HashMap::from([("source".to_string(), "sensor-42".to_string())]),
        subscription_identifier: None,
        content_type: Some("application/rdf+turtle".to_string()),
    };
    let encoded: Vec<u8> = encode_mqtt5_properties(&props);
    let (decoded, _consumed) = decode_mqtt5_properties(&encoded)?;
    assert_eq!(decoded.content_type, props.content_type);

    // Connect and publish
    let config = MqttConfig {
        broker_url: "tcp://localhost:1883".to_string(),
        client_id: "oxirs-consumer".to_string(),
        ..Default::default()
    };
    let mut client = MqttClient::new(config);
    let _event_loop = client.connect().await?;
    client
        .publish(
            "factory/line1/sensor/temp",
            b"{\"value\": 21.5}".to_vec(),
            QoS::AtLeastOnce,
            false,
        )
        .await?;

    // v4/v5 bridge scenarios: recover properties embedded by an upstream MQTT 5.0
    // broker from a raw byte slice (VarInt-length-prefixed, per section 2.2.2).
    let _props_from_bytes = MqttClient::parse_properties_from_bytes(&encoded)?;

    Ok(())
}
```

## Message Broker Configuration

### Kafka & Pulsar (adapter crates)

The in-tree `kafka`/`pulsar` Cargo features were removed in the COOLJAPAN Pure Rust
Policy v2 migration (`rdkafka`/`rdkafka-sys`/`libz-sys` and `pulsar`/`native-tls`/`lz4-sys`
are not Pure Rust). The former in-tree backends moved **verbatim** into two
`publish = false`, workspace-internal adapter crates that stay API-compatible with the
original in-tree types:

- `oxirs-stream-adapter-rdkafka` — `KafkaBackend`, `KafkaProducerConfig`, `SaslConfig`, `SslConfig`, schema registry client
- `oxirs-stream-adapter-pulsar` — the former in-tree `PulsarProducer`/`PulsarConsumer`

Construct `oxirs_stream_adapter_rdkafka::KafkaBackend` / `oxirs_stream_adapter_pulsar::PulsarProducer`
directly rather than going through `oxirs-stream`'s own backend enum, which now returns a
typed "moved to the adapter crate" error for the Kafka/Pulsar variants.

### NATS

```rust
use oxirs_stream::backend::nats::config::{NatsAuthConfig, NatsStorageType};
use oxirs_stream::backend::nats::{NatsConfig, NatsConsumerConfig};

let config = NatsConfig {
    url: "nats://localhost:4222".to_string(),
    subject_prefix: "rdf".to_string(), // publishes/subscribes under "rdf.>"
    stream_name: "RDF_STREAM".to_string(),
    storage_type: NatsStorageType::File, // JetStream persistence (vs. Memory)
    auth_config: Some(NatsAuthConfig {
        token: None,
        username: None,
        password: None,
        nkey: None,
        jwt: None,
        creds_file: Some("./nats.creds".to_string()),
    }),
    consumer_config: NatsConsumerConfig {
        queue_group: Some("oxirs-processors".to_string()),
        ..NatsConsumerConfig::default()
    },
    ..Default::default()
};
```

## Windowing

`WindowConfig`/`WindowType`/`WindowSize` (re-exported from the `rsp` — RDF Stream
Processing — module) drive the window semantics consumed by `RspProcessor`.

### Tumbling Windows

Fixed-size, non-overlapping windows:

```rust
use oxirs_stream::{WindowConfig, WindowSize, WindowType};
use chrono::Duration;

let config = WindowConfig {
    window_type: WindowType::Tumbling,
    size: WindowSize::Time(Duration::seconds(60)),
    slide: None,
    start_time: None,
    end_time: None,
};

// Process 60-second windows
```

### Sliding Windows

Overlapping windows:

```rust
let config = WindowConfig {
    window_type: WindowType::Sliding,
    size: WindowSize::Time(Duration::seconds(60)),
    slide: Some(WindowSize::Time(Duration::seconds(30))),  // 30-second slide
    start_time: None,
    end_time: None,
};

// Windows: [0-60s], [30-90s], [60-120s], ...
```

### Session Windows

Dynamic windows based on inactivity gaps:

```rust
let config = WindowConfig {
    window_type: WindowType::Session {
        gap: Duration::seconds(300),  // 5-minute inactivity closes the window
    },
    size: WindowSize::Time(Duration::seconds(0)), // unused by Session windows, but required by the struct
    slide: None,
    start_time: None,
    end_time: None,
};
```

Windows are also count-based via `WindowSize::Triples(n)` instead of `WindowSize::Time(_)`.

> **Note on the sections below:** the snippets from here through "Performance" sketch the
> streaming operations this crate implements (filtering, mapping, aggregation,
> temporal/graph pattern detection, checkpointing, retry/dead-letter handling, and
> store/query integration) as a single fluent `StreamProcessor` builder. That flat facade
> is aspirational and does not exist in the current public API — treat these as conceptual
> sketches, not compilable code. The real, tested entry points for the same capabilities
> are: `cep_engine` (pattern detection, rule engine, event correlation), `data_quality`
> (validators, cleansers, profilers), `stream_router` / `dead_letter_queue` /
> `event_filter` (routing, filtering, DLQ), `aggregation::ExactlyOnceAggregator`,
> `checkpoint` / `fault_tolerance` (checkpoint coordination and recovery), and
> `store_integration::RealtimeUpdateManager` (pushing stream changes into an
> `oxirs-core` store — the practical equivalent of the SHACL/ARQ integration examples
> below). See `src/lib.rs` for the full, current public API.

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
| Kafka (`oxirs-stream-adapter-rdkafka`) | 100K triples/s | 15ms |
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

### Production Release (v0.3.2)
- ✅ MQTT 5.0 property codec (`backend::mqtt::properties`) — encode/decode for the PUBLISH-relevant
  property set (Payload Format Indicator, Message Expiry Interval, Content Type, Response Topic,
  Correlation Data, Subscription Identifier, Topic Alias, repeatable User Properties), wired into
  `MqttClient::parse_properties_from_bytes()`
- ✅ NATS JetStream integration with persisted consumer/offset configuration
- ✅ Kafka/Pulsar available via the separate `oxirs-stream-adapter-{rdkafka,pulsar}` crates
  (COOLJAPAN Pure Rust Policy v2 quarantine — see "Kafka & Pulsar" above)
- ✅ Windowing (tumbling/sliding/session/landmark), filtering, and mapping
- ✅ Aggregation operators and pattern matching / CEP engine
- ✅ SPARQL stream federation with `SERVICE` bridging to remote endpoints
- ✅ Prometheus + OTLP metrics/tracing (`MonitoringConfig.otlp_endpoint`, env
  `OTEL_EXPORTER_OTLP_ENDPOINT` — replaces the deprecated `opentelemetry-jaeger` exporter) for
  throughput, lag, and error rates
- ✅ Exactly-once semantics (Chandy-Lamport checkpointing + idempotent producers + atomic ingress transactions)
- ✅ Distributed stream processing across cluster nodes (Raft-backed operator state via oxirs-cluster)
- ✅ 1735 tests passing (`--all-features`), zero warnings

## Contributing

Feedback and contributions welcome — see [CONTRIBUTING.md](../../CONTRIBUTING.md).

## License

Apache-2.0

## See Also

- [oxirs-shacl](../../engine/oxirs-shacl/) - Stream validation
- [oxirs-arq](../../engine/oxirs-arq/) - Stream queries
- [oxirs-federate](../oxirs-federate/) - Federated streams