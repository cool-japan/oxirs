# OxiRS Embed - Knowledge Graph Embeddings

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: Production Release (v0.1.0) - Released January 2026

✨ **Production Release**: Production-ready with API stability guarantees. Semantic versioning enforced.

Generate vector embeddings for RDF knowledge graphs enabling semantic similarity search, entity linking, and neural-symbolic AI integration.

## Features

### Embedding Models
- **TransE** - Translational distance models for knowledge graphs
- **DistMult** - Bilinear diagonal models for symmetric relations
- **ComplEx** - Complex-valued embeddings for asymmetric relations
- **RotatE** - Rotation-based models in complex space
- **HolE** - Holographic embeddings using circular correlation (NEW in v0.1.0)
- **ConvE** - Convolutional 2D neural network embeddings (NEW in v0.1.0)
- **TuckER** - Tucker decomposition for multi-relational learning
- **QuatE** - Quaternion embeddings for complex patterns

### Advanced Features (NEW in v0.1.0)
- **Link Prediction** - Predict missing triples (head/tail/relation)
  - Filtered ranking to remove known triples
  - Batch prediction for efficiency
  - Evaluation metrics (MRR, Hits@K, Mean Rank)
- **Entity Clustering** - Group similar entities
  - K-Means with K-Means++ initialization
  - Hierarchical (agglomerative) clustering
  - DBSCAN (density-based) clustering
  - Spectral clustering
  - Quality metrics (silhouette score, inertia)
- **Community Detection** - Find communities in knowledge graphs
  - Louvain modularity optimization
  - Label propagation
  - Girvan-Newman edge betweenness
  - Embedding-based detection
- **Vector Search** - High-performance semantic search (NEW in 0.1.0)
  - Exact search with multiple distance metrics
  - Cosine similarity, Euclidean, dot product, Manhattan
  - Batch search for multiple queries
  - Radius-based filtering
  - Parallel processing support
- **Visualization** - t-SNE, PCA, UMAP, Random Projection
  - 2D and 3D dimensionality reduction
  - Export to CSV/JSON formats
  - Cluster-aware visualizations
- **Interpretability** - Model understanding tools
  - Similarity analysis and nearest neighbors
  - Feature importance analysis
  - Counterfactual explanations
  - Embedding space diagnostics
- **Mixed Precision Training** - FP16/FP32 for faster training
- **Model Quantization** - Int8/Int4/Binary compression (3-4x size reduction)

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
- **Knowledge Graph Completion** - Fill missing facts in KGs
- **Anomaly Detection** - Detect unusual patterns in graphs

## Installation

Add to your `Cargo.toml`:

```toml
# Experimental feature
[dependencies]
oxirs-embed = "0.1.0"

# Enable specific providers
oxirs-embed = { version = "0.1.0", features = ["openai", "sentence-transformers"] }
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

## New Models (v0.1.0)

### HolE (Holographic Embeddings)

HolE uses circular correlation to model entity and relation interactions. Effective for capturing symmetric and asymmetric patterns.

```rust
use oxirs_embed::{
    models::hole::{HoLE, HoLEConfig},
    EmbeddingModel, ModelConfig, NamedNode, Triple,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Configure HolE model
    let config = HoLEConfig {
        base: ModelConfig {
            dimensions: 100,
            learning_rate: 0.01,
            max_epochs: 100,
            ..Default::default()
        },
        regularization: 0.0001,
        margin: 1.0,
        num_negatives: 5,
        use_sigmoid: true,
    };

    let mut model = HoLE::new(config);

    // Add triples
    model.add_triple(Triple::new(
        NamedNode::new("paris")?,
        NamedNode::new("capital_of")?,
        NamedNode::new("france")?,
    ))?;

    // Train
    let stats = model.train(Some(100)).await?;

    // Score triple
    let score = model.score_triple("paris", "capital_of", "france")?;
    println!("Score: {:.4}", score);

    Ok(())
}
```

### ConvE (Convolutional Embeddings)

ConvE uses 2D CNNs for expressive knowledge graph embeddings. Parameter-efficient with shared convolutional filters.

```rust
use oxirs_embed::models::conve::{ConvE, ConvEConfig};

let config = ConvEConfig {
    base: ModelConfig {
        dimensions: 200,
        learning_rate: 0.001,
        max_epochs: 100,
        ..Default::default()
    },
    reshape_width: 20,  // 200 / 20 = 10 height
    num_filters: 32,
    kernel_size: 3,
    dropout_rate: 0.3,
    ..Default::default()
};

let mut model = ConvE::new(config);

// Add triples and train as before
model.add_triple(triple)?;
model.train(Some(100)).await?;
```

### Link Prediction

Predict missing entities or relations in knowledge graphs.

```rust
use oxirs_embed::link_prediction::{LinkPredictionConfig, LinkPredictor};

// Create predictor
let pred_config = LinkPredictionConfig {
    top_k: 5,
    filter_known_triples: true,
    min_confidence: 0.0,
    parallel: true,
    batch_size: 100,
};

let predictor = LinkPredictor::new(pred_config, model);

// Predict tail entity (object prediction)
let candidates = vec!["bob".to_string(), "charlie".to_string()];
let predictions = predictor.predict_tail("alice", "knows", &candidates)?;

for pred in predictions {
    println!("{} (score: {:.4}, rank: {})", pred.entity, pred.score, pred.rank);
}

// Predict head entity (subject prediction)
let predictions = predictor.predict_head("knows", "bob", &candidates)?;

// Predict relation
let relations = vec!["knows".to_string(), "friend_of".to_string()];
let predictions = predictor.predict_relation("alice", "bob", &relations)?;
```

### Entity Clustering

Group similar entities based on learned embeddings.

```rust
use oxirs_embed::clustering::{ClusteringAlgorithm, ClusteringConfig, EntityClustering};
use std::collections::HashMap;

// Extract embeddings
let mut embeddings = HashMap::new();
for entity in model.get_entities() {
    if let Ok(emb) = model.get_entity_embedding(&entity) {
        let array = scirs2_core::ndarray_ext::Array1::from_vec(emb.values);
        embeddings.insert(entity, array);
    }
}

// K-Means clustering
let config = ClusteringConfig {
    algorithm: ClusteringAlgorithm::KMeans,
    num_clusters: 5,
    max_iterations: 100,
    ..Default::default()
};

let mut clustering = EntityClustering::new(config);
let result = clustering.cluster(&embeddings)?;

println!("Silhouette score: {:.3}", result.silhouette_score);
println!("Cluster assignments:");
for (entity, cluster_id) in result.assignments {
    println!("  {} -> Cluster {}", entity, cluster_id);
}
```

### Community Detection

Find communities in knowledge graphs using graph structure and embeddings.

```rust
use oxirs_embed::community_detection::{CommunityAlgorithm, CommunityConfig, CommunityDetector};

let config = CommunityConfig {
    algorithm: CommunityAlgorithm::Louvain,
    min_community_size: 2,
    resolution: 1.0,
    ..Default::default()
};

let mut detector = CommunityDetector::new(config);
let result = detector.detect(&triples)?;

println!("Modularity: {:.3}", result.modularity);
println!("Found {} communities", result.communities.len());
```

### Vector Search

High-performance semantic search for knowledge graph embeddings.

```rust
use oxirs_embed::vector_search::{VectorSearchIndex, SearchConfig, DistanceMetric};

// Build search index
let config = SearchConfig {
    metric: DistanceMetric::Cosine,
    parallel: true,
    normalize: true,
    ..Default::default()
};

let mut index = VectorSearchIndex::new(config);
index.build(&embeddings)?;

// Search for similar entities
let query_embedding = embeddings["iphone"].to_vec();
let results = index.search(&query_embedding, 10)?;

for result in results {
    println!("{}: similarity = {:.3}", result.entity_id, result.score);
}

// Batch search
let queries = vec![query1, query2, query3];
let batch_results = index.batch_search(&queries, 10)?;

// Radius search (find all within distance)
let radius_results = index.radius_search(&query_embedding, 0.5)?;
```

### Visualization

Visualize embeddings in 2D/3D using dimensionality reduction.

```rust
use oxirs_embed::visualization::{EmbeddingVisualizer, ReductionMethod, VisualizationConfig};

// PCA visualization
let config = VisualizationConfig {
    method: ReductionMethod::PCA,
    target_dims: 2,
    ..Default::default()
};

let mut visualizer = EmbeddingVisualizer::new(config);
let result = visualizer.visualize(&embeddings)?;

// t-SNE visualization (better for discovering clusters)
let tsne_config = VisualizationConfig {
    method: ReductionMethod::TSNE,
    target_dims: 2,
    tsne_perplexity: 30.0,
    max_iterations: 1000,
    ..Default::default()
};

let mut tsne_viz = EmbeddingVisualizer::new(tsne_config);
let tsne_result = tsne_viz.visualize(&embeddings)?;

// Export to CSV for plotting
for (entity, coords) in &tsne_result.coordinates {
    println!("{},{},{}", entity, coords[0], coords[1]);
}
```

### Interpretability

Understand why models make certain predictions.

```rust
use oxirs_embed::interpretability::{InterpretabilityAnalyzer, InterpretabilityConfig, InterpretationMethod};

// Similarity analysis
let config = InterpretabilityConfig {
    method: InterpretationMethod::SimilarityAnalysis,
    top_k: 10,
    ..Default::default()
};

let analyzer = InterpretabilityAnalyzer::new(config);
let analysis = analyzer.similarity_analysis("alice", &embeddings)?;

println!("Most similar to 'alice':");
for (entity, score) in &analysis.similar_entities {
    println!("  {}: {:.3}", entity, score);
}

// Feature importance
let importance_config = InterpretabilityConfig {
    method: InterpretationMethod::FeatureImportance,
    top_k: 10,
    ..Default::default()
};

let imp_analyzer = InterpretabilityAnalyzer::new(importance_config);
let importance = imp_analyzer.feature_importance("alice", &embeddings)?;

// Counterfactual explanations
let counterfactual = analyzer.counterfactual_explanation("alice", "bob", &embeddings)?;
println!("To be like Bob, Alice would need to change {} dimensions",
    counterfactual.required_changes.len());
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

### Production Release (v0.1.0)
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