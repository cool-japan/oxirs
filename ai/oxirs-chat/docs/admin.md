# OxiRS Chat — Administrator Guide

This guide covers deployment, configuration, and observability for production
`oxirs-chat` deployments.

---

## 1. Deployment topology

### Single-node (development / small workloads)

The simplest topology runs `oxirs-chat` in-process alongside the Fuseki HTTP
server:

```
┌───────────────────────────────────────┐
│  oxirs-fuseki (HTTP :3030)            │
│    └─ oxirs-chat (in-process)         │
│         ├─ RagEngine                  │
│         ├─ LLMManager → OpenAI / etc. │
│         └─ NL2SPARQLSystem            │
│  oxirs-tdb (local file store)         │
└───────────────────────────────────────┘
```

Scale-out is achieved by running multiple instances behind a load balancer with
session affinity (sticky sessions). Session state is serialised to a shared
volume or object store via `OxiRSChat::save_sessions` / `load_sessions`.

### Multi-node (production)

```
            ┌─────────────────────────────┐
            │  Load Balancer (e.g. Nginx) │
            └──────────┬──────────────────┘
                       │ (session-affinity)
         ┌─────────────▼─────────────┐
         │  oxirs-fuseki node A      │   oxirs-fuseki node B
         │  oxirs-chat in-process    │   oxirs-chat in-process
         └───────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  oxirs-cluster (Raft)     │   Shared RDF store
         │  oxirs-tdb replicas       │
         └───────────────────────────┘
```

Session JSON files should be stored on shared persistent storage (NFS, S3-backed
FUSE, or a key-value store) so any node can reload sessions after a restart.

---

## 2. Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | No | — | OpenAI API key. Provider is disabled when absent. |
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key. Provider is disabled when absent. |
| `OXIRS_CHAT_SESSION_DIR` | No | `/tmp/oxirs-chat-sessions` | Directory for session persistence. |
| `OXIRS_CHAT_SESSION_TIMEOUT_SECS` | No | `3600` | Idle session eviction threshold in seconds. |
| `OXIRS_CHAT_MAX_CONTEXT_TOKENS` | No | `8000` | Max token budget per context window. |
| `OXIRS_CHAT_MAX_RESULTS` | No | `10` | RAG retrieval candidate limit. |
| `OXIRS_CHAT_GRAPH_DEPTH` | No | `2` | Graph traversal depth for entity expansion. |
| `RUST_LOG` | No | `info` | Log filter (see Logging section). |

Sensitive values (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) must **never** be
committed to version control. Use a secret manager (HashiCorp Vault, AWS
Secrets Manager, Kubernetes Secrets) and inject them at runtime.

---

## 3. Configuration file format

`oxirs-chat` reads optional configuration from a TOML file. The file path is
passed via the `--config` CLI flag when using the binary target, or loaded
programmatically via the `config` crate.

```toml
# oxirs-chat.toml

[session]
max_context_tokens   = 8000
sliding_window_size  = 20
enable_context_compression = true
temperature          = 0.7
max_tokens           = 2000
timeout_seconds      = 30
enable_topic_tracking = true
enable_sentiment_analysis = true
enable_intent_detection = true

[llm]
# Provider keys come from environment variables; only routing policy goes here.
[llm.routing]
strategy = "QualityFirst"   # QualityFirst | CostOptimized | LatencyOptimized | Balanced | RoundRobin
quality_threshold = 0.8

[llm.fallback]
enabled = true
max_attempts = 3

[llm.rate_limits]
requests_per_minute = 60
tokens_per_minute   = 10000
burst_allowed = true

[llm.circuit_breaker]
failure_threshold = 5
timeout_duration_secs = 60
recovery_threshold = 3
sliding_window_size = 20

[rag]
[rag.retrieval]
max_results = 10
similarity_threshold = 0.7
graph_traversal_depth = 2
enable_entity_expansion = true
enable_quantum_enhancement = false
enable_consciousness_integration = false
```

---

## 4. Logging configuration

`oxirs-chat` uses the [`tracing`](https://docs.rs/tracing) framework. Log output
is structured JSON or human-readable text depending on the subscriber registered
by the host application.

### Default initialisation (human-readable)

```rust
tracing_subscriber::fmt()
    .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
    .init();
```

Set `RUST_LOG` to control verbosity:

```bash
# Info-level globally, debug for oxirs_chat::rag
RUST_LOG="info,oxirs_chat::rag=debug" ./oxirs-fuseki
```

### Structured JSON (production)

```rust
use tracing_subscriber::prelude::*;

tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer().json())
    .with(tracing_subscriber::EnvFilter::from_default_env())
    .init();
```

### Key span names

| Span / event | Meaning |
|---|---|
| `oxirs_chat` | Top-level crate events |
| `oxirs_chat::rag` | RAG retrieval pipeline |
| `oxirs_chat::llm` | LLM provider calls |
| `oxirs_chat::nl2sparql` | NL → SPARQL translation |
| `oxirs_chat::session` | Session lifecycle events |

---

## 5. Observability hooks

### Metrics

`oxirs-chat` emits counters and histograms through the
[`metrics`](https://docs.rs/metrics) facade. Plug in any compatible exporter
(Prometheus, StatsD, etc.):

```rust
// Register a Prometheus exporter before creating OxiRSChat:
use metrics_exporter_prometheus::PrometheusBuilder;

PrometheusBuilder::new()
    .install()
    .expect("failed to install Prometheus recorder");

// Now create OxiRSChat normally; metrics flow automatically.
```

Key metrics emitted:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `oxirs_chat_messages_total` | counter | `session_id`, `role` | Messages processed |
| `oxirs_chat_rag_results` | histogram | — | Context results per query |
| `oxirs_chat_llm_latency_seconds` | histogram | `provider` | LLM round-trip latency |
| `oxirs_chat_circuit_breaker_state` | gauge | `provider` | 0=Closed, 1=HalfOpen, 2=Open |
| `oxirs_chat_sessions_active` | gauge | — | Currently active sessions |

### Tracing / distributed traces

`oxirs-chat` is instrumented with `#[tracing::instrument]` on hot paths. To
forward traces to an OpenTelemetry collector:

```rust
use opentelemetry::sdk::trace;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::prelude::*;

let tracer = opentelemetry_jaeger::new_agent_pipeline()
    .with_service_name("oxirs-chat")
    .install_batch(opentelemetry::runtime::Tokio)?;

tracing_subscriber::registry()
    .with(OpenTelemetryLayer::new(tracer))
    .with(tracing_subscriber::EnvFilter::from_default_env())
    .init();
```

### Health endpoint

When embedding `oxirs-chat` behind the Fuseki HTTP server, the health monitoring
module provides a machine-readable status report:

```rust
use oxirs_chat::health_monitoring::{HealthMonitor, HealthMonitoringConfig, HealthStatus};

let monitor = HealthMonitor::new(HealthMonitoringConfig::default());
let report = monitor.generate_health_report().await?;

match report.overall_status {
    HealthStatus::Healthy  => println!("OK"),
    HealthStatus::Degraded => println!("DEGRADED"),
    _                      => println!("CRITICAL"),
}
println!("Uptime: {:?}", report.uptime);
```

Expose this via an HTTP `/health` route in your Axum router for load-balancer
health checks.

---

## 6. Docker / container deployment

A reference `Dockerfile` and `docker-compose.yml` are included in the crate
root. Key points:

- The `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` environment variables must be
  injected at container runtime — never baked into the image.
- Mount a persistent volume for session storage at the path configured in
  `OXIRS_CHAT_SESSION_DIR`.
- The binary target is `oxirs-chat` (built via `cargo build --release --bin oxirs-chat`).

---

## See also

- [tutorial.md](tutorial.md) — User-facing guide: sessions, queries, persistence.
- [API reference](https://docs.rs/oxirs-chat) — Complete rustdoc.
