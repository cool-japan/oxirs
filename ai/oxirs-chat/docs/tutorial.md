# OxiRS Chat — User Tutorial

This guide walks through the main workflows for building a conversational interface
over a knowledge graph with `oxirs-chat`.

## Prerequisites

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
oxirs-chat = { version = "0.3.0" }
oxirs-core = { version = "0.3.0" }
anyhow = "1"
tokio = { version = "1", features = ["full"] }
```

---

## 1. Starting a chat session against a local knowledge graph

The entry point is `OxiRSChat`. It wraps an `oxirs_core::Store`, a RAG engine, an
LLM manager, and a natural-language-to-SPARQL (NL2SPARQL) engine.

```rust
use oxirs_chat::{ChatConfig, OxiRSChat};
use oxirs_core::ConcreteStore;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Open an in-memory RDF store (swap ConcreteStore for any oxirs_core::Store impl).
    let store = Arc::new(ConcreteStore::new()?);

    // 2. Build the chat system with default settings.
    let chat = OxiRSChat::new(ChatConfig::default(), store).await?;

    // 3. Create a named session (one per user is typical).
    let _session = chat.create_session("alice".to_string()).await?;

    // 4. Send a message. The system performs RAG retrieval, optional NL2SPARQL
    //    translation, and LLM response generation automatically.
    let response = chat
        .process_message("alice", "What triples are stored about BRCA1?".to_string())
        .await?;

    println!("{}", response.content.text());
    Ok(())
}
```

**Lifecycle helpers:**

```rust
// List active sessions.
let ids: Vec<String> = chat.list_sessions().await;

// Remove a session when the user logs out.
chat.remove_session("alice").await;

// Evict sessions that have been idle past the timeout.
let evicted = chat.cleanup_expired_sessions().await;
println!("Evicted {} sessions", evicted);
```

---

## 2. Configuring the LLM provider

`OxiRSChat::new_with_llm_config` accepts an `Option<LLMConfig>`. `LLMConfig`
holds one `ProviderConfig` per provider name (defaults include `"openai"` and
`"anthropic"`). Provider selection is configuration-time, not runtime; to
switch providers between requests you should rebuild the `LLMConfig` before
creating the chat instance.

```rust
use oxirs_chat::{ChatConfig, LLMConfig, OxiRSChat};
use oxirs_chat::llm::ProviderConfig;
use oxirs_core::ConcreteStore;
use std::sync::Arc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = Arc::new(ConcreteStore::new()?);

    // Build an LLMConfig that enables only the Anthropic provider.
    let mut providers = HashMap::new();
    providers.insert("anthropic".to_string(), ProviderConfig::anthropic_default());

    let llm_config = LLMConfig {
        providers,
        ..LLMConfig::default()
    };

    let chat = OxiRSChat::new_with_llm_config(
        ChatConfig::default(),
        store,
        Some(llm_config),
    )
    .await?;

    let _session = chat.create_session("bob".to_string()).await?;
    println!("Chat with Anthropic provider ready.");
    Ok(())
}
```

The system automatically falls back to the next available provider if the
primary one is unavailable (controlled by `FallbackConfig` inside `LLMConfig`).
Circuit-breaker state per provider is accessible via:

```rust
let stats = chat.get_circuit_breaker_stats().await?;
for (provider, s) in &stats {
    println!("{}: {:?}", provider, s.state);
}
```

---

## 3. Configuring retrieval depth

Retrieval behaviour is controlled through `rag::RagConfig` and its nested
`RetrievalConfig`. Build a custom `RagEngine` directly and embed it, or create
the `OxiRSChat` with the defaults and then note the key knobs below.

| Field | Default | Meaning |
|---|---|---|
| `max_results` | 10 | Maximum semantic-search hits per query |
| `similarity_threshold` | 0.7 | Minimum cosine similarity to include a result |
| `graph_traversal_depth` | 2 | Graph hops to follow from seed entities |
| `enable_entity_expansion` | true | Expand seed entities via graph traversal |
| `enable_quantum_enhancement` | false | Enable quantum-inspired retrieval |
| `enable_consciousness_integration` | false | Enable consciousness-aware context |

These fields live in `rag::RetrievalConfig`. To use non-default values, create
the chat system with `OxiRSChat::new_with_llm_config` and supply a custom
`rag::RagConfig` to `rag::RagEngine` directly, then integrate it:

```rust
use oxirs_chat::rag::{RagConfig, RetrievalConfig};

let rag_config = RagConfig {
    retrieval: RetrievalConfig {
        max_results: 20,          // return more candidates
        graph_traversal_depth: 3, // explore 3 hops
        similarity_threshold: 0.6,
        enable_entity_expansion: true,
        enable_quantum_enhancement: false,
        enable_consciousness_integration: false,
    },
    ..RagConfig::default()
};
```

---

## 4. Persisting session history

Sessions can be serialised to disk as JSON and reloaded across process restarts.

```rust
use oxirs_chat::{ChatConfig, OxiRSChat};
use oxirs_core::ConcreteStore;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = Arc::new(ConcreteStore::new()?);
    let chat = OxiRSChat::new(ChatConfig::default(), store.clone()).await?;

    // Restore previously saved sessions.
    let loaded = chat.load_sessions("/var/lib/oxirs-chat/sessions").await?;
    println!("Loaded {} sessions from disk", loaded);

    // ... run the application ...

    // Flush all active sessions before shutdown.
    let saved = chat.save_sessions("/var/lib/oxirs-chat/sessions").await?;
    println!("Saved {} sessions to disk", saved);
    Ok(())
}
```

Each session is stored in its own `<session_id>.json` file. `SessionData` is
the serde-serialisable snapshot, available via `chat.export_session(id).await`
for ad-hoc persistence or export:

```rust
let data = chat.export_session("alice").await?;
let json = serde_json::to_string_pretty(&data)?;
println!("{}", json);

// Restore into a different chat instance:
chat.import_session(data).await?;
```

---

## 5. Switching between LLM providers at runtime

Full runtime provider switching is not supported at the session level; providers
are selected at construction time via `LLMConfig`. However, you can reset a
provider's circuit breaker to re-enable it after failures:

```rust
// Force-reset the OpenAI circuit breaker after fixing connectivity issues.
chat.reset_circuit_breaker("openai").await?;
```

To genuinely switch the primary provider for a user mid-session, the recommended
pattern is:

1. Export the session data with `chat.export_session`.
2. Construct a new `OxiRSChat` with the desired `LLMConfig`.
3. Import the session data with `chat.import_session`.

---

## 6. Streaming responses

For real-time UIs, use `process_message_stream` to receive incremental chunks:

```rust
use oxirs_chat::StreamResponseChunk;
use tokio_stream::StreamExt; // or tokio::sync::mpsc::Receiver

let mut rx = chat
    .process_message_stream("alice", "Explain RDF graphs.".to_string())
    .await?;

while let Some(chunk) = rx.recv().await {
    match chunk {
        StreamResponseChunk::Status { stage, progress, .. } => {
            eprintln!("[{:.0}%] {:?}", progress * 100.0, stage);
        }
        StreamResponseChunk::Content { text, .. } => {
            print!("{}", text);
        }
        StreamResponseChunk::Complete { total_time, .. } => {
            println!("\nDone in {:.2}s", total_time.as_secs_f64());
            break;
        }
        StreamResponseChunk::Error { error, .. } => {
            eprintln!("Error: {}", error.message);
            break;
        }
        _ => {}
    }
}
```

---

## Next steps

- See [admin.md](admin.md) for deployment and observability configuration.
- See the [API reference](https://docs.rs/oxirs-chat) for complete type documentation.
