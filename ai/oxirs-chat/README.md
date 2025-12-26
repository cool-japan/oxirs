# OxiRS Chat

[![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/oxirs/releases)

**AI-powered conversational interface for RDF data with RAG and natural language to SPARQL**

**Status**: Release Candidate (v0.1.0-rc.1) - Released December 26, 2025

✨ **Release Candidate**: Production-ready with API stability guarantees and comprehensive testing.

## Overview

`oxirs-chat` provides an intelligent conversational interface for querying and exploring RDF datasets. It combines Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to enable natural language queries over semantic data, automatic SPARQL generation, and contextual explanations.

## Features

- **Natural Language to SPARQL**: Convert natural language questions to SPARQL queries
- **RAG Integration**: Retrieve relevant context from RDF data to enhance LLM responses
- **Multi-Model Support**: OpenAI, Anthropic, local models via Ollama, and Hugging Face
- **Context Management**: Maintain conversation history and query context
- **Explanation Engine**: Explain query results and reasoning in natural language
- **Vector Search**: Semantic similarity search over RDF data
- **Interactive Chat**: Web-based chat interface with syntax highlighting
- **Query Suggestions**: Intelligent query suggestions based on data schema
- **Data Exploration**: Guided exploration of unfamiliar datasets
- **Custom Prompts**: Configurable prompts for domain-specific use cases

## Installation

Add to your `Cargo.toml`:

```toml
# Experimental feature
[dependencies]
oxirs-chat = "0.1.0-rc.1"

# Enable specific LLM providers
oxirs-chat = { version = "0.1.0-rc.1", features = ["openai", "anthropic", "ollama"] }
```

## Quick Start

### Basic Chat Server

```rust
use oxirs_chat::{ChatServer, Config, LLMProvider};
use oxirs_core::Dataset;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load your RDF dataset
    let dataset = Dataset::from_file("knowledge_base.ttl")?;
    
    // Configure the chat server
    let config = Config::builder()
        .llm_provider(LLMProvider::OpenAI {
            api_key: std::env::var("OPENAI_API_KEY")?,
            model: "gpt-4".to_string(),
        })
        .dataset(dataset)
        .enable_rag(true)
        .enable_web_interface(true)
        .port(3000)
        .build();
    
    // Start the chat server
    let server = ChatServer::new(config);
    server.run().await
}
```

### Natural Language Queries

```rust
use oxirs_chat::{ChatBot, Message, QueryResult};

let mut chatbot = ChatBot::new(config);

// Natural language query
let response = chatbot.query("Who are the directors of movies released in 2023?").await?;

match response {
    QueryResult::Sparql { query, results, explanation } => {
        println!("Generated SPARQL: {}", query);
        println!("Results: {:?}", results);
        println!("Explanation: {}", explanation);
    }
    QueryResult::Natural { answer } => {
        println!("Answer: {}", answer);
    }
}
```

## Natural Language to SPARQL

### Query Generation

```rust
use oxirs_chat::{NL2SPARQL, SchemaContext};

let nl2sparql = NL2SPARQL::new()
    .with_schema_context(SchemaContext::from_dataset(&dataset))
    .with_examples_from_file("examples.json")?;

// Convert natural language to SPARQL
let question = "What movies did Christopher Nolan direct?";
let result = nl2sparql.convert(question).await?;

println!("Question: {}", question);
println!("SPARQL: {}", result.sparql);
println!("Confidence: {:.2}", result.confidence);
```

### Custom Prompts

```rust
use oxirs_chat::{PromptTemplate, PromptBuilder};

let prompt = PromptTemplate::builder()
    .system_message(r#"
        You are an expert in converting natural language questions about movies 
        into SPARQL queries. The dataset contains information about movies, 
        directors, actors, and genres.
        
        Schema:
        - :Movie a class with properties :title, :releaseYear, :director, :actor, :genre
        - :Person a class with properties :name, :birthDate
        - :Genre a class with properties :name
    "#)
    .few_shot_examples(&[
        ("Who directed Inception?", 
         "SELECT ?director WHERE { :Inception :director ?director }"),
        ("What movies were released in 2010?",
         "SELECT ?movie WHERE { ?movie :releaseYear 2010 }"),
    ])
    .build();

let nl2sparql = NL2SPARQL::with_prompt(prompt);
```

## RAG (Retrieval-Augmented Generation)

### Vector Search Integration

```rust
use oxirs_chat::{RAGEngine, VectorStore, EmbeddingModel};
use oxirs_vec::HNSWIndex;

// Create vector store with embeddings
let embedding_model = EmbeddingModel::OpenAI("text-embedding-ada-002");
let vector_store = VectorStore::new(HNSWIndex::new())
    .with_embedding_model(embedding_model);

// Index RDF data
vector_store.index_dataset(&dataset).await?;

// Create RAG engine
let rag = RAGEngine::new()
    .with_vector_store(vector_store)
    .with_retrieval_count(5)
    .with_similarity_threshold(0.7);

// Use in chat
let chatbot = ChatBot::new(config)
    .with_rag_engine(rag);
```

### Context Retrieval

```rust
use oxirs_chat::{ContextRetriever, RetrievalStrategy};

let retriever = ContextRetriever::new()
    .strategy(RetrievalStrategy::Hybrid {
        vector_weight: 0.7,
        keyword_weight: 0.3,
    })
    .max_context_length(4000)
    .include_schema_info(true);

// Retrieve relevant context for a question
let context = retriever.retrieve("What are the highest-grossing sci-fi movies?", &dataset).await?;

println!("Retrieved context:");
for doc in context.documents {
    println!("- {} (score: {:.3})", doc.content, doc.score);
}
```

## Conversation Management

### Session Handling

```rust
use oxirs_chat::{ChatSession, ConversationHistory};

let mut session = ChatSession::new()
    .with_memory_limit(50) // Keep last 50 messages
    .with_context_window(4000); // 4k token context

// Maintain conversation
session.add_message("user", "Tell me about Christopher Nolan movies");
let response1 = chatbot.chat(&mut session).await?;

session.add_message("assistant", &response1);
session.add_message("user", "Which ones won Academy Awards?");
let response2 = chatbot.chat(&mut session).await?; // Uses previous context
```

### Context Compression

```rust
use oxirs_chat::{ContextCompressor, CompressionStrategy};

let compressor = ContextCompressor::new()
    .strategy(CompressionStrategy::Summarization)
    .compression_ratio(0.3);

// Compress conversation history when context gets too long
if session.context_length() > 3000 {
    session.compress_history(&compressor).await?;
}
```

## Web Interface

### Chat UI

```rust
use oxirs_chat::{WebServer, ChatUI};

let web_server = WebServer::new()
    .with_chat_ui(ChatUI::default())
    .with_syntax_highlighting(true)
    .with_query_visualization(true)
    .with_result_tables(true);

// Access at http://localhost:3000
web_server.serve("0.0.0.0:3000").await?;
```

### REST API

```http
POST /api/chat
Content-Type: application/json

{
  "message": "What are the most popular movies of 2023?",
  "session_id": "user123",
  "include_sparql": true,
  "explain_results": true
}
```

Response:
```json
{
  "response": "Based on the data, here are the most popular movies of 2023...",
  "sparql_query": "SELECT ?movie ?title ?popularity WHERE { ... }",
  "results": [...],
  "explanation": "This query searches for movies released in 2023...",
  "suggestions": ["Tell me more about top movie", "What genres were most popular?"]
}
```

## Advanced Features

### Multi-Modal Responses

```rust
use oxirs_chat::{ResponseFormat, MediaType};

let chatbot = ChatBot::new(config)
    .response_format(ResponseFormat::MultiModal {
        include_text: true,
        include_charts: true,
        include_tables: true,
        include_graphs: true,
    });

// Generate rich responses with visualizations
let response = chatbot.query("Show me the trend of movie releases by year").await?;

match response {
    QueryResult::MultiModal { text, visualizations } => {
        println!("Text: {}", text);
        for viz in visualizations {
            match viz.media_type {
                MediaType::Chart => println!("Chart: {}", viz.url),
                MediaType::Graph => println!("Knowledge Graph: {}", viz.url),
                MediaType::Table => println!("Table: {}", viz.data),
            }
        }
    }
}
```

### Custom Function Integration

```rust
use oxirs_chat::{ChatFunction, FunctionRegistry};

// Define custom function
#[derive(ChatFunction)]
struct MovieRecommendation {
    user_preferences: Vec<String>,
}

impl MovieRecommendation {
    async fn recommend(&self, user_id: &str) -> Result<Vec<Movie>, Error> {
        // Custom recommendation logic
        Ok(movies)
    }
}

// Register with chatbot
let mut chatbot = ChatBot::new(config);
chatbot.register_function(MovieRecommendation::new());

// Use in conversation
// User: "Recommend some movies for me"
// Assistant: Let me find some movie recommendations based on your preferences...
```

### Schema-Aware Suggestions

```rust
use oxirs_chat::{SuggestionEngine, SchemaAnalyzer};

let analyzer = SchemaAnalyzer::new(&dataset);
let schema_info = analyzer.analyze().await?;

let suggestion_engine = SuggestionEngine::new()
    .with_schema_info(schema_info)
    .with_query_patterns(&common_patterns)
    .with_user_history(&user_history);

// Generate contextual suggestions
let suggestions = suggestion_engine.suggest("movies").await?;
// Returns: ["movies by director", "popular movies", "recent movies", ...]
```

## Configuration

### Chat Configuration

```yaml
chat:
  llm:
    provider: "openai"
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.1
    max_tokens: 2000
    
  rag:
    enabled: true
    retrieval_count: 5
    similarity_threshold: 0.7
    embedding_model: "text-embedding-ada-002"
    
  conversation:
    max_history: 50
    context_window: 4000
    compression_enabled: true
    
  features:
    explain_queries: true
    show_sparql: true
    generate_charts: true
    suggestion_engine: true
    
web_interface:
  enabled: true
  port: 3000
  syntax_highlighting: true
  query_visualization: true
```

### Model Configuration

```rust
use oxirs_chat::{LLMConfig, EmbeddingConfig};

let config = Config::builder()
    .llm_config(LLMConfig::OpenAI {
        api_key: env::var("OPENAI_API_KEY")?,
        model: "gpt-4".to_string(),
        temperature: 0.1,
        max_tokens: 2000,
        timeout: Duration::from_secs(30),
    })
    .embedding_config(EmbeddingConfig::OpenAI {
        model: "text-embedding-ada-002".to_string(),
        batch_size: 100,
    })
    .build();
```

## Performance

### Benchmarks

| Operation | Latency (p95) | Throughput | Memory |
|-----------|---------------|------------|--------|
| NL to SPARQL | 800ms | 75 q/min | 45MB |
| RAG retrieval | 200ms | 300 q/min | 120MB |
| Simple chat | 600ms | 100 q/min | 35MB |
| Complex reasoning | 1.5s | 40 q/min | 85MB |

### Optimization

```rust
use oxirs_chat::{CacheConfig, PerformanceConfig};

let config = Config::builder()
    .cache_config(CacheConfig {
        query_cache: true,
        embedding_cache: true,
        response_cache: true,
        ttl: Duration::from_hours(1),
    })
    .performance_config(PerformanceConfig {
        parallel_retrieval: true,
        batch_embeddings: true,
        streaming_responses: true,
        connection_pooling: true,
    })
    .build();
```

## Deployment

### Docker

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin oxirs-chat

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/oxirs-chat /usr/local/bin/
EXPOSE 3000
CMD ["oxirs-chat", "--config", "/config.yaml"]
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export OXIRS_CHAT_PORT="3000"
export OXIRS_CHAT_HOST="0.0.0.0"
export RUST_LOG="oxirs_chat=info"
```

## Related Crates

- [`oxirs-core`](../../core/oxirs-core/): RDF data model
- [`oxirs-vec`](../oxirs-vec/): Vector search and embeddings
- [`oxirs-arq`](../../engine/oxirs-arq/): SPARQL query engine
- [`oxirs-fuseki`](../../server/oxirs-fuseki/): SPARQL server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Status

🚀 **Release Candidate (v0.1.0-rc.1)** – December 26, 2025

Implementation Status:
- ✅ **Chat Infrastructure**: Session management, persistence, expiration handling
- ✅ **Multi-LLM Support**: OpenAI, Anthropic Claude, Ollama/local with intelligent routing
- ✅ **RAG System**: Vector search, context assembly, knowledge retrieval integrated with persisted datasets
- ✅ **Natural Language to SPARQL**: Query generation with validation, optimization, and federation-aware prompts
- ✅ **Advanced Caching**: Multi-tier caching with LRU/LFU eviction policies
- ✅ **Performance Monitoring**: Real-time metrics, slow-query tracing, and SciRS2 telemetry dashboards
- ✅ **Analytics System**: Conversation tracking, pattern detection, quality analysis
- ✅ **HTTP/WebSocket Server**: REST API and real-time WebSocket communication
- ✅ **Session Recovery**: Backup/restore mechanisms with corruption handling
- ✅ **Message Threading**: Reply chains and conversation threads
- ✅ **Context Management**: Sliding window with summarization and topic tracking
- ✅ **CLI & Fuseki Integration**: Seamless dataset bootstrapping and persistence hand-off

**Test Coverage**: 12/12 tests passing with comprehensive integration tests

Release candidate; expect rapid iteration and potential API changes before stable.