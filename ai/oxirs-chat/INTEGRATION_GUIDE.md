# OxiRS Chat Integration Guide

This guide provides comprehensive instructions for integrating oxirs-chat into your Rust applications.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Advanced Features](#advanced-features)
4. [Performance Optimization](#performance-optimization)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Setup

```rust
use oxirs_chat::{ChatConfig, OxiRSChat};
use oxirs_core::ConcreteStore;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize RDF store
    let store = Arc::new(ConcreteStore::new()?);

    // 2. Configure chat system
    let config = ChatConfig {
        max_context_tokens: 8000,
        temperature: 0.7,
        max_tokens: 2000,
        ..Default::default()
    };

    // 3. Create chat instance
    let chat = OxiRSChat::new(config, store).await?;

    // 4. Create a session
    let session = chat.create_session("user_123".to_string()).await?;

    // 5. Process messages
    let response = chat.process_message(
        "user_123",
        "What is semantic web?".to_string()
    ).await?;

    println!("Response: {}", response.content);

    Ok(())
}
```

## Configuration

### Chat Configuration

```rust
use oxirs_chat::ChatConfig;

let config = ChatConfig {
    // Context window settings
    max_context_tokens: 16000,          // Maximum tokens in context
    sliding_window_size: 50,            // Number of messages to keep
    enable_context_compression: true,   // Compress older messages

    // LLM settings
    temperature: 0.8,                   // Creativity (0.0-2.0)
    max_tokens: 4000,                   // Max response length
    timeout_seconds: 60,                // Request timeout

    // Advanced features
    enable_topic_tracking: true,        // Track conversation topics
    enable_sentiment_analysis: true,    // Analyze message sentiment
    enable_intent_detection: true,      // Detect user intent
};
```

### LLM Integration

```rust
use oxirs_chat::llm::{LLMConfig, CircuitBreakerConfig};

let llm_config = LLMConfig {
    // API Keys (read from environment)
    openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
    anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),

    // Provider settings
    default_provider: "openai".to_string(),
    enable_fallback: true,

    // Circuit breaker for resilience
    circuit_breaker: CircuitBreakerConfig {
        failure_threshold: 5,
        success_threshold: 2,
        timeout_seconds: 30,
        half_open_max_calls: 3,
    },

    ..Default::default()
};

let chat = OxiRSChat::new_with_llm_config(config, store, Some(llm_config)).await?;
```

### RAG Configuration

```rust
use oxirs_chat::rag::RagConfig;

let rag_config = RagConfig {
    retrieval: oxirs_chat::rag::RetrievalConfig {
        top_k: 10,                      // Number of results to retrieve
        min_similarity: 0.7,            // Minimum similarity threshold
        enable_reranking: true,         // Re-rank results

        // Advanced features
        enable_quantum_enhancement: true,
        enable_consciousness_integration: true,
        ..Default::default()
    },

    embedding: oxirs_chat::rag::EmbeddingConfig {
        model_name: "all-MiniLM-L6-v2".to_string(),
        dimension: 384,
        normalize: true,
        ..Default::default()
    },

    ..Default::default()
};
```

## Advanced Features

### Streaming Responses

```rust
use tokio_stream::StreamExt;

// Get streaming response
let mut stream = chat.process_message_stream(
    "user_123",
    "Explain quantum computing".to_string()
).await?;

// Process chunks as they arrive
while let Some(chunk) = stream.recv().await {
    match chunk {
        StreamResponseChunk::Status { stage, progress, .. } => {
            println!("Progress: {:.1}%", progress * 100.0);
        }
        StreamResponseChunk::Content { text, .. } => {
            print!("{}", text);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
        StreamResponseChunk::Complete { total_time, .. } => {
            println!("\nCompleted in {:.2}s", total_time.as_secs_f64());
        }
        _ => {}
    }
}
```

### Schema-Aware Query Generation

```rust
// Enable schema discovery
chat.discover_schema().await?;

// Get discovered schema
if let Some(schema) = chat.get_discovered_schema().await {
    println!("Discovered {} classes", schema.classes.len());

    for class in &schema.classes {
        println!("Class: {} ({} instances)",
            class.uri, class.instance_count);
    }
}

// Now queries will be schema-aware
let response = chat.process_message(
    "user_123",
    "Show me all Person entities".to_string()
).await?;
```

### Real-Time Collaboration

```rust
use oxirs_chat::collaboration::{CollaborationManager, CollaborationConfig};

let collab_config = CollaborationConfig {
    enable_cursor_sharing: true,
    enable_presence: true,
    enable_version_control: true,
    max_participants: 10,
    ..Default::default()
};

let collab_manager = CollaborationManager::new(collab_config);

// Create shared session
let shared_session = collab_manager
    .create_shared_session("project_123".to_string())
    .await?;

// Add participants
collab_manager.add_participant(
    "project_123",
    "user_1",
    oxirs_chat::collaboration::ParticipantRole::Editor
).await?;

// Get real-time updates
let mut updates = collab_manager.subscribe_to_updates("project_123").await?;
while let Some(update) = updates.recv().await {
    println!("Update: {:?}", update);
}
```

### Analytics and Monitoring

```rust
use oxirs_chat::dashboard::{DashboardAnalytics, DashboardConfig, TimeRange};

let dashboard_config = DashboardConfig {
    enable_real_time_updates: true,
    update_interval_seconds: 5,
    retention_hours: 24,
    ..Default::default()
};

let dashboard = DashboardAnalytics::new(dashboard_config);

// Get dashboard overview
let overview = dashboard.get_dashboard_overview(TimeRange::Last24Hours).await?;
println!("Total queries: {}", overview.total_queries);
println!("Active users: {}", overview.active_users);
println!("Avg response time: {:.2}ms", overview.avg_response_time_ms);

// Get detailed query analytics
let query_analytics = dashboard.get_query_analytics(TimeRange::Last7Days).await?;
for record in query_analytics.recent_queries.iter().take(10) {
    println!("{}: {} ({:.2}s)",
        record.timestamp.format("%H:%M:%S"),
        record.query_type,
        record.response_time.as_secs_f64()
    );
}
```

### Voice Interface

```rust
use oxirs_chat::voice::{VoiceInterface, VoiceConfig, SttProviderType, TtsProviderType};

let voice_config = VoiceConfig {
    stt_provider: SttProviderType::Whisper,
    tts_provider: TtsProviderType::ElevenLabs,
    enable_stt: true,
    enable_tts: true,
    ..Default::default()
};

let voice = VoiceInterface::new(voice_config)?;

// Speech to text
let audio_data = std::fs::read("audio.wav")?;
let transcription = voice.transcribe(&audio_data).await?;
println!("Transcribed: {}", transcription.text);

// Process with chat
let response = chat.process_message(
    "user_123",
    transcription.text
).await?;

// Text to speech
let audio = voice.synthesize(&response.content.to_string()).await?;
std::fs::write("response.mp3", audio)?;
```

## Performance Optimization

### Connection Pooling

```rust
// Use Arc for shared access
let store = Arc::new(ConcreteStore::new()?);
let chat = Arc::new(OxiRSChat::new(config, store).await?);

// Share across tasks
let chat_clone = chat.clone();
tokio::spawn(async move {
    // Process messages in parallel
    chat_clone.process_message("user_1", "Query 1".to_string()).await
});
```

### Caching

```rust
use oxirs_chat::cache::semantic::SemanticCache;

// Enable semantic caching for repeated queries
let cache = SemanticCache::new(cache_config)?;

// Cache is automatically used for similar queries
let response1 = chat.process_message("user_1", "What is RDF?".to_string()).await?;
let response2 = chat.process_message("user_2", "Explain RDF".to_string()).await?;
// Second query may use cached results if similarity > threshold
```

### Batch Processing

```rust
// Process multiple queries concurrently
let queries = vec![
    ("user_1", "Query 1"),
    ("user_2", "Query 2"),
    ("user_3", "Query 3"),
];

let futures: Vec<_> = queries.iter().map(|(user, query)| {
    chat.process_message(user, query.to_string())
}).collect();

let results = futures::future::join_all(futures).await;
```

## Production Deployment

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Redis (for caching)
export REDIS_URL="redis://localhost:6379"

# PostgreSQL (for session persistence)
export DATABASE_URL="postgresql://user:pass@localhost/oxirs_chat"
```

### Docker Deployment

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release -p oxirs-chat

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates
COPY --from=builder /app/target/release/oxirs-chat /usr/local/bin/
EXPOSE 8080
CMD ["oxirs-chat"]
```

### Health Checks

```rust
use oxirs_chat::health_monitoring::{HealthMonitor, HealthMonitoringConfig};

let health_config = HealthMonitoringConfig {
    enable_health_checks: true,
    check_interval_seconds: 30,
    enable_auto_healing: true,
    ..Default::default()
};

let health_monitor = HealthMonitor::new(health_config);

// Generate health report
let report = health_monitor.generate_health_report().await?;
match report.overall_status {
    HealthStatus::Healthy => println!("✓ System healthy"),
    HealthStatus::Degraded => println!("⚠ System degraded"),
    HealthStatus::Critical => println!("✗ System critical"),
    _ => {}
}
```

### Logging and Tracing

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

tracing_subscriber::registry()
    .with(tracing_subscriber::EnvFilter::new(
        std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into())
    ))
    .with(tracing_subscriber::fmt::layer())
    .init();
```

## Troubleshooting

### Common Issues

#### 1. LLM Provider Timeout

```rust
// Increase timeout
let config = ChatConfig {
    timeout_seconds: 120,  // Increase from default 60s
    ..Default::default()
};

// Enable fallback
let llm_config = LLMConfig {
    enable_fallback: true,
    ..Default::default()
};
```

#### 2. Memory Usage

```rust
// Reduce context window
let config = ChatConfig {
    max_context_tokens: 4000,  // Reduce from default 8000
    sliding_window_size: 20,   // Keep fewer messages
    enable_context_compression: true,
    ..Default::default()
};
```

#### 3. Circuit Breaker Open

```rust
// Reset circuit breaker
chat.reset_circuit_breaker("openai").await?;

// Check circuit breaker stats
let stats = chat.get_circuit_breaker_stats().await?;
for (provider, stat) in stats {
    println!("{}: {:?}", provider, stat.state);
}
```

### Debug Mode

```rust
// Enable detailed logging
std::env::set_var("RUST_LOG", "oxirs_chat=debug");

// Get detailed session statistics
let stats = chat.get_session_statistics("user_123").await?;
println!("Session stats: {:#?}", stats);

// Export session data for debugging
let session_data = chat.export_session("user_123").await?;
println!("Session data: {:#?}", session_data);
```

## Examples

See the `examples/` directory for complete working examples:

- `examples/advanced_features_demo.rs` - Comprehensive feature demonstration
- `examples/streaming_demo.rs` - Streaming response example
- `examples/quick_start_guide.rs` - Basic usage (from oxirs-embed)

Run examples with:
```bash
cargo run --example advanced_features_demo
cargo run --example streaming_demo
```

## Benchmarks

Run benchmarks to measure performance:

```bash
cargo bench --bench comprehensive_benchmarks
```

## Support

- Documentation: https://docs.rs/oxirs-chat
- Repository: https://github.com/cool-japan/oxirs
- Issues: https://github.com/cool-japan/oxirs/issues

## License

Licensed under Apache 2.0 or MIT license.
