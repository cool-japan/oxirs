# OxiRS Embed - Knowledge Graph Embeddings

[![Version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: v0.4.0 - Released 2026-07-19

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
[dependencies]
oxirs-embed = "0.3.2"

# Enable optional feature groups (advanced models, GPU acceleration, API server, ...)
oxirs-embed = { version = "0.3.2", features = ["advanced-models", "gpu"] }
```

## Quick Start

### Basic Entity Embedding

Free-text embedding uses the domain-specific transformer models in
[`biomedical_embeddings`](src/biomedical_embeddings/) (SciBERT, CodeBERT, BioBERT,
LegalBERT, FinBERT, ClinicalBERT, ChemBERT):

```rust
use oxirs_embed::biomedical_embeddings::{SpecializedTextEmbedding, SpecializedTextModel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a domain-specific text embedding model (SciBERT here)
    let mut model = SpecializedTextEmbedding::new(SpecializedTextEmbedding::scibert_config());

    // Generate an embedding for a piece of text
    let text = "Machine learning researcher specializing in NLP";
    let embedding = model.encode_text(text).await?;

    println!("Embedding dimension: {}", embedding.len());
    println!("Model: {:?}", SpecializedTextModel::SciBERT.model_name());

    Ok(())
}
```

### Knowledge Graph Embedding

Structural knowledge graph embedding trains one of the `EmbeddingModel` implementations
(`TransE`, `DistMult`, `ComplEx`, `RotatE`, `HolE`, `ConvE`, `TuckER`, `QuatE`, ...)
directly on RDF triples:

```rust
use oxirs_embed::{EmbeddingModel, ModelConfig, NamedNode, TransE, Triple};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a knowledge graph embedding model
    let config = ModelConfig::default().with_dimensions(128);
    let mut model = TransE::new(config);

    // Add knowledge triples
    model.add_triple(Triple::new(
        NamedNode::new("http://example.org/Person/Alice")?,
        NamedNode::new("http://example.org/knows")?,
        NamedNode::new("http://example.org/Person/Bob")?,
    ))?;

    // Train the model
    let stats = model.train(Some(100)).await?;
    println!("Training completed: {stats:?}");

    // Get the embedding for a specific entity
    let embedding = model.get_entity_embedding("http://example.org/Person/Alice")?;
    println!("Embedding for Alice: {} dimensions", embedding.dimensions);

    Ok(())
}
```

`TransE` also implements `Default` (delegating to `ModelConfig::default()`, i.e. 100
dimensions, L2 distance, margin 1.0), so `TransE::default()` works anywhere a plain
`M: EmbeddingModel + Default` bound is needed — for example in generic benchmark/test
harnesses. Prefer `TransE::new(config)` when you need a specific dimension or learning
rate.

### Semantic Similarity

```rust
use oxirs_embed::embedding_aggregator::cosine_similarity;

// Compare two entity embeddings (both `&[f32]`, e.g. from `Vector::values`)
let score = cosine_similarity(&alice_embedding.values, &bob_embedding.values);
println!("Alice <-> Bob similarity: {score:.3}");
```

For ranked nearest-neighbor search over many entities at once, use the
[Vector Search](#vector-search) index below rather than a single pairwise call.

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
    println!("{} (score: {:.4}, rank: {})", pred.predicted_id, pred.score, pred.rank);
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
let result = detector.detect_from_triples(&triples)?;

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

`oxirs-embed` ships two families of models rather than a single pluggable "provider"
facade: structural knowledge-graph embedders (trained locally on your triples) and
specialized domain text-embedding models (also local — no network calls).

### Structural Knowledge Graph Models

Selected via Cargo features (`basic-models` is the default):

| Feature | Models |
|---------|--------|
| `basic-models` (default) | `TransE`, `ComplEx`, `DistMult`, `HoLE` |
| `advanced-models` | `RotatE`, `ConvE`, `TuckER`, `QuatE` |

Each is a plain struct implementing the `EmbeddingModel` trait — see
[Knowledge Graph Embedding](#knowledge-graph-embedding) above.

### Specialized Domain Text Models

`oxirs_embed::biomedical_embeddings::SpecializedTextModel` covers seven pretrained
architectures, each with a matching `SpecializedTextEmbedding::{name}_config()` helper
(`scibert_config()`, `codebert_config()`, `biobert_config()`, ...):

- `SciBERT` - Scientific literature (768 dimensions)
- `CodeBERT` - Code and programming languages
- `BioBERT` - Biomedical literature
- `LegalBERT` - Legal documents
- `FinBERT` - Financial texts
- `ClinicalBERT` - Clinical notes
- `ChemBERT` - Chemical compounds

```rust
use oxirs_embed::biomedical_embeddings::{SpecializedTextConfig, SpecializedTextEmbedding, SpecializedTextModel};

let config = SpecializedTextConfig {
    model_type: SpecializedTextModel::BioBERT,
    ..Default::default()
};
let mut model = SpecializedTextEmbedding::new(config);
let embedding = model.encode_text("BRCA1 is associated with breast cancer risk").await?;
```

## Advanced Features

### Batch Processing

The `EmbeddingModel::encode` trait method already accepts multiple texts per call —
there is no separate batch-sized entry point:

```rust
use oxirs_embed::EmbeddingModel;

let texts: Vec<String> = vec![
    "Machine learning".to_string(),
    "Natural language processing".to_string(),
    "Computer vision".to_string(),
];

// One call encodes the whole batch: Vec<Vec<f32>>, one embedding per input text
let embeddings = model.encode(&texts).await?;
```

### Contextual Embeddings

`ContextualEmbeddingModel` adapts embeddings to query/user/task/temporal context.
It is still evolving (its crate-root re-export is currently disabled), so import it
via its module path:

```rust,no_run
use oxirs_embed::contextual::{ContextualConfig, ContextualEmbeddingModel, EmbeddingContext};
use oxirs_embed::Triple;

# async fn example(triples: &[Triple]) -> anyhow::Result<()> {
let mut model = ContextualEmbeddingModel::new(ContextualConfig::default())?;
let context = EmbeddingContext::default();

let embeddings = model.embed_with_context(triples, &context).await?;
println!("Generated {} contextual embeddings", embeddings.len());
# Ok(())
# }
```

### Entity Linking

```rust
use oxirs_embed::entity_linking::{EntityLinker, EntityLinkerConfig};
use scirs2_core::ndarray_ext::Array1;
use std::collections::HashMap;

let linker = EntityLinker::new(EntityLinkerConfig::default(), entity_embeddings)?;

// Link a (pre-embedded) mention to knowledge graph entities — the mention text
// "machine learning expert from Stanford" must already be embedded into the
// same vector space as `entity_embeddings` (e.g. via a text embedding model).
let mention_embedding: Array1<f32> = embed_mention("machine learning expert from Stanford");
let candidates = linker.link_entity(&mention_embedding, None)?;

for result in candidates {
    println!("{}: confidence={:.3}", result.entity_id, result.confidence);
}
```

### Relation Prediction

```rust
use oxirs_embed::entity_linking::{RelationPredictor, RelationPredictorConfig};

let predictor = RelationPredictor::new(
    RelationPredictorConfig::default(),
    relation_embeddings,
    entity_embeddings,
);

// Predict relations between two entities
let predictions = predictor.predict_relations(
    "http://example.org/Alice",
    "http://example.org/Bob",
)?;

for pred in predictions {
    println!("Predicted relation: {} (score={:.3})", pred.relation, pred.score);
}
```

## Integration with OxiRS

### With oxirs-vec (Vector Search)

```rust
use oxirs_embed::{EmbeddingModel, TransE};
use oxirs_vec::{Vector as VecVector, VectorStore, VectorStoreTrait};

// Index every trained entity embedding in an oxirs-vec store
let mut store = VectorStore::new();
for entity in model.get_entities() {
    let embedding = model.get_entity_embedding(&entity)?;
    store.insert_vector(entity, VecVector::new(embedding.values))?;
}

// Find the entities closest to Alice
let query = model.get_entity_embedding("http://example.org/Person/Alice")?;
let results = store.search_similar(&VecVector::new(query.values), 10)?;
```

### With oxirs-chat (RAG)

`oxirs-chat`'s `rag` module consumes entity/text embeddings produced by
`oxirs-embed` as part of its retrieval-augmented generation pipeline. See
[oxirs-chat](../oxirs-chat/) for the current pipeline API.

## Performance

### Benchmarks

Run the criterion suite for up-to-date numbers on your own hardware:

```bash
cargo bench -p oxirs-embed --bench performance_validation
```

### Optimization Tips

```rust
use oxirs_embed::EmbeddingModel;

// encode() already batches: pass every text in one call instead of looping
let embeddings = model.encode(&texts).await?;

// Cache embeddings (fixed-capacity LRU, keyed by content hash + model id)
use oxirs_embed::embedding_cache::{CacheKey, EmbeddingCache};

let mut cache = EmbeddingCache::new(10_000);
let key = CacheKey::new(content_hash, "scibert");
if let Some(cached) = cache.get(&key) {
    // reuse cached embedding
} else {
    let embedding = model.encode(&[text.to_string()]).await?;
    cache.insert(key, embedding[0].clone());
}
```

GPU acceleration (feature `gpu`, Pure-Rust via `scirs2-core`/`scirs2-linalg`) is
provided by `oxirs_embed::gpu_acceleration::{GpuAccelerationConfig, GpuAccelerationManager}`
— see the crate-level docs for a full example.

## Status

### Production Release (v0.4.0)
- ✅ Structural KG embedding models: TransE, DistMult, ComplEx, RotatE, HoLE, ConvE, TuckER, QuatE
- ✅ Specialized domain text embeddings: SciBERT, CodeBERT, BioBERT, LegalBERT, FinBERT, ClinicalBERT, ChemBERT
- ✅ Link prediction, entity clustering, community detection, vector search, visualization, interpretability
- ✅ Entity linking and relation prediction (`entity_linking` module)
- ✅ Fine-tuning (`fine_tuning`/`fine_tuner`) and model ensembling (`ensemble`)
- ✅ Model zoo with SHA256-verified manifests (`model_zoo`)
- ✅ GPU acceleration behind the optional `gpu` feature (Pure-Rust, no CUDA/FFI required)
- 🚧 Contextual embeddings (`contextual` module) – implemented but not yet re-exported at the crate root while the API stabilizes

## Contributing

This is an experimental module. Feedback welcome!

## License

Apache-2.0

## See Also

- [oxirs-vec](../../engine/oxirs-vec/) - Vector search engine
- [oxirs-chat](../oxirs-chat/) - AI-powered chat with RAG
- [oxirs-core](../../core/oxirs-core/) - RDF data model