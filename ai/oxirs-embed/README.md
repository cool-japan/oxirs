# OxiRS Embed - Knowledge Graph Embeddings

[![Version](https://img.shields.io/badge/version-0.1.0--alpha.3-orange)](https://github.com/cool-japan/oxirs/releases)

**Status**: Alpha Release (v0.1.0-alpha.3) - Released October 12, 2025

⚠️ **Alpha Software**: This is an early alpha release. Experimental features. APIs may change without notice. Not recommended for production use.

Generate vector embeddings for RDF knowledge graphs enabling semantic similarity search, entity linking, and neural-symbolic AI integration.

## Features

### Embedding Models
- **Sentence Transformers** - Pre-trained models from HuggingFace
- **OpenAI Embeddings** - GPT-based embeddings via API
- **Custom Models** - Bring your own embedding models
- **Multi-lingual Support** - Models for various languages

### Knowledge Graph Embedding
- **Entity Embeddings** - Generate embeddings for RDF entities
- **Relation Embeddings** - Embed predicates and relationships
- **Graph Embeddings** - Whole-graph vector representations
- **Contextual Embeddings** - Use graph context for better embeddings

### Applications
- **Semantic Search** - Find similar entities by meaning
- **Entity Linking** - Link mentions to knowledge graph entities
- **Relation Prediction** - Predict missing relationships
- **Clustering** - Group similar entities

## Installation

Add to your `Cargo.toml`:

```toml
# Experimental feature
[dependencies]
oxirs-embed = "0.1.0-alpha.3"

# Enable specific providers
oxirs-embed = { version = "0.1.0-alpha.3", features = ["openai", "sentence-transformers"] }
```

## Quick Start

### Basic Entity Embedding

```rust
use oxirs_embed::{EmbeddingModel, ModelProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load embedding model
    let model = EmbeddingModel::builder()
        .provider(ModelProvider::SentenceTransformers)
        .model_name("all-mpnet-base-v2")
        .build()
        .await?;

    // Generate embedding
    let text = "Machine learning researcher specializing in NLP";
    let embedding = model.encode(text).await?;

    println!("Embedding dimension: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding[..5]);

    Ok(())
}
```

### Knowledge Graph Embedding

```rust
use oxirs_embed::KnowledgeGraphEmbedder;
use oxirs_core::Dataset;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load knowledge graph
    let dataset = Dataset::from_file("knowledge_graph.ttl")?;

    // Create embedder
    let embedder = KnowledgeGraphEmbedder::builder()
        .model("sentence-transformers/all-mpnet-base-v2")
        .use_labels(true)
        .use_descriptions(true)
        .use_context(true)  // Include neighboring entities
        .build()
        .await?;

    // Embed all entities
    let embeddings = embedder.embed_dataset(&dataset).await?;

    // Get embedding for specific entity
    let entity_uri = "http://example.org/Person/Alice";
    if let Some(embedding) = embeddings.get(entity_uri) {
        println!("Embedding for Alice: {} dimensions", embedding.len());
    }

    Ok(())
}
```

### Semantic Similarity

```rust
use oxirs_embed::similarity::{cosine_similarity, find_similar};

// Find similar entities
let query_embedding = model.encode("AI researcher").await?;

let similar_entities = find_similar(
    &query_embedding,
    &embeddings,
    10,  // top 10 results
    0.7  // minimum similarity threshold
)?;

for (entity, score) in similar_entities {
    println!("{}: {:.3}", entity, score);
}
```

## Supported Embedding Providers

### Sentence Transformers (Local)

```rust
use oxirs_embed::{EmbeddingModel, ModelProvider};

let model = EmbeddingModel::builder()
    .provider(ModelProvider::SentenceTransformers)
    .model_name("all-mpnet-base-v2")  // or other models
    .device("cuda")  // Optional GPU support
    .build()
    .await?;
```

Popular models:
- `all-mpnet-base-v2` - General purpose, 768 dimensions
- `all-MiniLM-L6-v2` - Faster, 384 dimensions
- `multi-qa-mpnet-base-dot-v1` - For Q&A tasks
- `paraphrase-multilingual-mpnet-base-v2` - Multi-lingual

### OpenAI Embeddings (API)

```rust
use oxirs_embed::{EmbeddingModel, ModelProvider};

let model = EmbeddingModel::builder()
    .provider(ModelProvider::OpenAI)
    .api_key(std::env::var("OPENAI_API_KEY")?)
    .model_name("text-embedding-3-small")
    .build()
    .await?;

let embedding = model.encode("Your text here").await?;
```

Available models:
- `text-embedding-3-small` - 1536 dimensions, cost-effective
- `text-embedding-3-large` - 3072 dimensions, highest quality
- `text-embedding-ada-002` - Legacy model

### Custom Models

```rust
use oxirs_embed::{EmbeddingModel, CustomModelConfig};

let config = CustomModelConfig {
    model_path: "./my-model".into(),
    tokenizer_path: "./my-tokenizer".into(),
    dimension: 768,
};

let model = EmbeddingModel::custom(config).await?;
```

## Advanced Features

### Batch Processing

```rust
let texts = vec![
    "Machine learning",
    "Natural language processing",
    "Computer vision",
];

// Process in batches for efficiency
let embeddings = model.encode_batch(&texts, 32).await?;
```

### Contextual Embeddings

Use graph context for better embeddings:

```rust
use oxirs_embed::ContextualEmbedder;

let embedder = ContextualEmbedder::builder()
    .model("all-mpnet-base-v2")
    .context_depth(2)  // Include 2-hop neighbors
    .weight_by_relation(true)  // Different weights for different relations
    .build()
    .await?;

let embedding = embedder.embed_entity_with_context(
    "http://example.org/Alice",
    &dataset
).await?;
```

### Entity Linking

```rust
use oxirs_embed::EntityLinker;

let linker = EntityLinker::new(model, &entity_embeddings);

// Link text mention to knowledge graph entity
let mention = "machine learning expert from Stanford";
let candidates = linker.link(mention, 5).await?;

for (entity_uri, score) in candidates {
    println!("{}: {:.3}", entity_uri, score);
}
```

### Relation Prediction

```rust
use oxirs_embed::RelationPredictor;

let predictor = RelationPredictor::new(
    entity_embeddings,
    relation_embeddings
);

// Predict relation between entities
let predictions = predictor.predict_relation(
    "http://example.org/Alice",
    "http://example.org/Bob"
)?;

for (relation, score) in predictions {
    println!("Predicted relation: {} ({:.3})", relation, score);
}
```

## Integration with OxiRS

### With oxirs-vec (Vector Search)

```rust
use oxirs_embed::EmbeddingModel;
use oxirs_vec::VectorStore;

// Generate embeddings
let model = EmbeddingModel::load("all-mpnet-base-v2").await?;
let embeddings = model.encode_dataset(&dataset).await?;

// Index in vector store
let mut store = VectorStore::new(IndexType::HNSW, 768)?;
for (entity, embedding) in embeddings {
    store.add_vector(&entity, &embedding)?;
}
store.build_index()?;
```

### With oxirs-chat (RAG)

```rust
use oxirs_embed::EmbeddingModel;
use oxirs_chat::RagSystem;

let model = EmbeddingModel::load("all-mpnet-base-v2").await?;
let rag = RagSystem::builder()
    .embedding_model(model)
    .knowledge_graph(&dataset)
    .build()?;
```

## Performance

### Benchmark Results

| Model | Embedding Time | Dimension | Quality (Avg) |
|-------|---------------|-----------|---------------|
| all-mpnet-base-v2 | 15ms | 768 | 0.85 |
| all-MiniLM-L6-v2 | 5ms | 384 | 0.78 |
| text-embedding-3-small | 50ms* | 1536 | 0.88 |

*API call latency

### Optimization Tips

```rust
// Use batch processing
let embeddings = model.encode_batch(&texts, batch_size: 32).await?;

// Cache embeddings
let cache = EmbeddingCache::new("./cache")?;
let embedding = cache.get_or_compute(text, || model.encode(text))?;

// GPU acceleration (if available)
let model = EmbeddingModel::builder()
    .device("cuda")
    .build()
    .await?;
```

## Status

### Alpha Release (v0.1.0-alpha.3)
- ✅ Sentence Transformers integration with batch streaming + persistence
- ✅ OpenAI embeddings support with provider failover and caching
- ✅ Entity/graph embeddings wired into CLI ingest/export pipelines
- ✅ Semantic similarity search via `oxirs-vec` + SPARQL federation hooks
- 🚧 Contextual embeddings (expanded graph context) – in progress
- 🚧 Relation prediction (knowledge completion) – in progress
- ⏳ Fine-tuning support (planned for v0.2.0)

## Contributing

This is an experimental module. Feedback welcome!

## License

MIT OR Apache-2.0

## See Also

- [oxirs-vec](../../engine/oxirs-vec/) - Vector search engine
- [oxirs-chat](../oxirs-chat/) - AI-powered chat with RAG
- [oxirs-core](../../core/oxirs-core/) - RDF data model