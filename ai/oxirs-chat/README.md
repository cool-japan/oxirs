# OxiRS Chat

[![Version](https://img.shields.io/badge/version-0.3.2-blue)](https://github.com/cool-japan/oxirs/releases)
[![Tests](https://img.shields.io/badge/tests-1247%20passing-brightgreen)](https://github.com/cool-japan/oxirs)

**AI-powered conversational interface for RDF knowledge graphs with RAG and natural language to SPARQL**

---

## Overview

`oxirs-chat` provides an intelligent conversational interface for querying and
exploring RDF datasets. It combines Large Language Models (LLMs) with
Retrieval-Augmented Generation (RAG) to enable natural language queries over
semantic data, automatic SPARQL generation, and contextual explanations.

Key capabilities:

- **Natural Language to SPARQL** — Convert plain-English questions to SPARQL.
- **Context-aware entity extraction** — Rule-based NL entity recognition returning typed, span-and-confidence-scored entities (`nl2sparql::context_aware::extract_entities_rich`).
- **RAG Integration** — Retrieve relevant triples from the knowledge graph and pass them as context to the LLM.
- **Multi-provider LLM** — OpenAI and Anthropic out of the box; local providers configurable.
- **Streaming responses** — Word-by-word output via a channel-based API.
- **Session persistence** — Save and restore conversation sessions as JSON.
- **Circuit breakers** — Automatic fallback across LLM providers on failure.
- **Schema introspection** — Discover the RDF schema automatically to improve NL2SPARQL accuracy.
- **Consciousness-inspired retrieval** — Optional quantum-enhanced and consciousness-aware context assembly.

---

## Installation

```toml
[dependencies]
oxirs-chat = "0.3.2"
oxirs-core  = "0.3.2"
tokio = { version = "1", features = ["full"] }
anyhow = "1"
```

Optional features:

| Feature | Description |
|---|---|
| `llm-integration` | Enable extra LLM integration hooks |
| `nl2sparql` | Enable the NL-to-SPARQL subsystem |
| `excel-export` | Enable Excel (`.xlsx`) export of query results |
| `openai` | Enable the OpenAI provider (chat + Whisper STT/TTS) via `async-openai` |

---

## Quick Start

```rust,no_run
use oxirs_chat::{ChatConfig, OxiRSChat};
use oxirs_core::ConcreteStore;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Open an in-memory RDF store.
    let store = Arc::new(ConcreteStore::new()?);

    // 2. Build the chat system.
    let chat = OxiRSChat::new(ChatConfig::default(), store).await?;

    // 3. Create a named session (one per user).
    let _session = chat.create_session("alice".to_string()).await?;

    // 4. Ask a natural-language question.
    //    Set OPENAI_API_KEY or ANTHROPIC_API_KEY for full LLM responses.
    let response = chat
        .process_message("alice", "What genes are linked to breast cancer?".to_string())
        .await?;

    println!("{}", response.content.to_text());
    Ok(())
}
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/tutorial.md](docs/tutorial.md) | Step-by-step user guide: sessions, queries, LLM providers, retrieval depth, persistence |
| [docs/admin.md](docs/admin.md) | Deployment guide: topology, environment variables, logging, observability |
| [API reference](https://docs.rs/oxirs-chat) | Full rustdoc |

---

## Examples

| Example | Description |
|---|---|
| `examples/kg_chat.rs` | Chat over a populated RDF dataset; demonstrates session + NL2SPARQL |
| `examples/rag_chat.rs` | Direct RAG engine retrieval + full pipeline via `OxiRSChat` |
| `examples/chat_advanced_features_demo.rs` | Collaboration, schema discovery, circuit breakers |
| `examples/streaming_demo.rs` | Word-by-word streaming response with progress stages |

Run any example with:

```bash
OPENAI_API_KEY=sk-... cargo run --example kg_chat -p oxirs-chat
```

---

## Configuration Reference

### ChatConfig (session)

```rust,no_run
use oxirs_chat::ChatConfig;

let config = ChatConfig {
    max_context_tokens:      8000,   // Token budget for context window
    sliding_window_size:     20,     // Messages kept in sliding window
    enable_context_compression: true,
    temperature:             0.7,
    max_tokens:              2000,   // Max tokens in LLM response
    timeout_seconds:         30,
    enable_topic_tracking:   true,
    enable_sentiment_analysis: true,
    enable_intent_detection: true,
};
```

### LLM provider selection

```rust,no_run
use oxirs_chat::{LLMConfig, OxiRSChat, ChatConfig};
use oxirs_core::ConcreteStore;
use std::sync::Arc;

# async fn example() -> anyhow::Result<()> {
let store = Arc::new(ConcreteStore::new()?);

// Use Anthropic provider only.
// ANTHROPIC_API_KEY is read from the environment automatically.
let mut llm_config = LLMConfig::default();
llm_config.providers.retain(|k, _| k == "anthropic");

let chat = OxiRSChat::new_with_llm_config(
    ChatConfig::default(),
    store,
    Some(llm_config),
).await?;
# Ok(())
# }
```

### RAG retrieval depth

```rust,no_run
use oxirs_chat::rag::{RagConfig, RetrievalConfig};

let rag_config = RagConfig {
    retrieval: RetrievalConfig {
        max_results: 20,           // More retrieval candidates
        graph_traversal_depth: 3,  // Follow 3 hops in the graph
        similarity_threshold: 0.6,
        enable_entity_expansion: true,
        ..RetrievalConfig::default()
    },
    ..RagConfig::default()
};
```

---

## Session Persistence

```rust,no_run
use oxirs_chat::{ChatConfig, OxiRSChat};
use oxirs_core::ConcreteStore;
use std::sync::Arc;

# async fn example() -> anyhow::Result<()> {
let store = Arc::new(ConcreteStore::new()?);
let chat = OxiRSChat::new(ChatConfig::default(), store).await?;

// Restore sessions saved in a previous run.
let loaded = chat.load_sessions("/var/lib/oxirs-chat/sessions").await?;
println!("Restored {} sessions", loaded);

// ... application runs ...

// Flush before shutdown.
chat.save_sessions("/var/lib/oxirs-chat/sessions").await?;
# Ok(())
# }
```

---

## Streaming Responses

```rust,no_run
use oxirs_chat::{ChatConfig, OxiRSChat, StreamResponseChunk};
use oxirs_core::ConcreteStore;
use std::sync::Arc;

# async fn example() -> anyhow::Result<()> {
let store = Arc::new(ConcreteStore::new()?);
let chat = OxiRSChat::new(ChatConfig::default(), store).await?;
chat.create_session("alice".to_string()).await?;

let mut rx = chat
    .process_message_stream("alice", "Explain RDF graphs.".to_string())
    .await?;

while let Some(chunk) = rx.recv().await {
    match chunk {
        StreamResponseChunk::Content { text, .. } => print!("{}", text),
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
# Ok(())
# }
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (provider disabled when absent) |
| `ANTHROPIC_API_KEY` | Anthropic API key (provider disabled when absent) |
| `RUST_LOG` | Log filter (e.g. `info,oxirs_chat::rag=debug`) |

---

## Related Crates

- [`oxirs-core`](../../core/oxirs-core/) — RDF data model and store
- [`oxirs-vec`](../../engine/oxirs-vec/) — Vector search and HNSW indexes
- [`oxirs-embed`](../oxirs-embed/) — Knowledge graph embedding models
- [`oxirs-arq`](../../engine/oxirs-arq/) — SPARQL query engine
- [`oxirs-fuseki`](../../server/oxirs-fuseki/) — SPARQL HTTP server

---

## License

Apache-2.0 — see [LICENSE-APACHE](../../LICENSE-APACHE).

*OxiRS Chat v0.3.2 — AI-powered conversational RDF interface by COOLJAPAN OU (Team Kitasan)*
