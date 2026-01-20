# OxiRS Embed - Feature Documentation

*Last Updated: 2026-01-06*

## Overview

OxiRS Embed provides comprehensive tools for knowledge graph embeddings with state-of-the-art models, advanced analysis capabilities, and production-ready optimizations.

## Table of Contents

1. [Embedding Models](#embedding-models)
2. [Link Prediction](#link-prediction)
3. [Clustering](#clustering)
4. [Community Detection](#community-detection)
5. [Visualization](#visualization)
6. [Interpretability](#interpretability)
7. [Performance Optimization](#performance-optimization)
8. [Integration Examples](#integration-examples)

---

## Embedding Models

### TransE (Translational Embeddings)
Translation-based model where `h + r ≈ t` for triple `(h, r, t)`.

```rust
use oxirs_embed::{TransE, TransEConfig, ModelConfig, Triple};

let config = TransEConfig {
    base: ModelConfig {
        dimensions: 128,
        learning_rate: 0.01,
        epochs: 100,
        ..Default::default()
    },
    margin: 1.0,
    norm: 2,
    ..Default::default()
};

let mut model = TransE::new(config);
model.add_triple(Triple {
    head: "alice".to_string(),
    relation: "knows".to_string(),
    tail: "bob".to_string(),
}).unwrap();

model.train(Some(100)).await.unwrap();
```

### HolE (Holographic Embeddings)
Uses circular correlation for entity-relation combinations.

```rust
use oxirs_embed::{HoLE, HoLEConfig};

let config = HoLEConfig {
    base: ModelConfig {
        dimensions: 100,
        ..Default::default()
    },
    use_sigmoid: true,
    ..Default::default()
};

let mut model = HoLE::new(config);
// Train model...
```

### ConvE (Convolutional Embeddings)
2D convolutional neural networks for knowledge graph completion.

```rust
use oxirs_embed::{ConvE, ConvEConfig};

let config = ConvEConfig {
    base: ModelConfig {
        dimensions: 200,
        ..Default::default()
    },
    reshape_width: 20,
    num_filters: 32,
    kernel_size: 3,
    dropout_rate: 0.3,
    ..Default::default()
};

let mut model = ConvE::new(config);
// Train model...
```

---

## Link Prediction

Predict missing links in knowledge graphs with comprehensive evaluation metrics.

```rust
use oxirs_embed::{LinkPredictor, LinkPredictionConfig};

let pred_config = LinkPredictionConfig {
    top_k: 10,
    min_confidence: 0.5,
    filter_known_triples: true,
    parallel: true,
    ..Default::default()
};

let predictor = LinkPredictor::new(pred_config, trained_model);

// Predict tail entities
let candidates = vec!["bob".to_string(), "charlie".to_string()];
let predictions = predictor.predict_tail("alice", "knows", &candidates)?;

for pred in predictions {
    println!("{}: score={:.4}, confidence={:.4}, rank={}",
        pred.predicted_id, pred.score, pred.confidence, pred.rank);
}

// Evaluate on test set
let test_triples = vec![/* ... */];
let metrics = predictor.evaluate(&test_triples, &candidates)?;

println!("MRR: {:.4}", metrics.mrr);
println!("Hits@1: {:.4}", metrics.hits_at_1);
println!("Hits@10: {:.4}", metrics.hits_at_10);
```

### Metrics

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for correct predictions
- **Hits@K**: Percentage of correct entities in top-K predictions
- **Mean Rank**: Average rank of correct entities

---

## Clustering

Group similar entities based on their embeddings.

```rust
use oxirs_embed::{EntityClustering, ClusteringConfig, ClusteringAlgorithm};

let config = ClusteringConfig {
    algorithm: ClusteringAlgorithm::KMeans,
    num_clusters: 10,
    max_iterations: 100,
    tolerance: 1e-4,
    ..Default::default()
};

let mut clustering = EntityClustering::new(config);
let result = clustering.cluster(&entity_embeddings)?;

println!("Found {} clusters", result.num_clusters);
println!("Silhouette score: {:.4}", result.silhouette_score);
println!("Inertia: {:.4}", result.inertia);

// Get cluster assignments
for (entity, cluster_id) in &result.assignments {
    println!("{} -> cluster {}", entity, cluster_id);
}
```

### Supported Algorithms

1. **K-Means**: Partitions entities into K clusters
2. **Hierarchical**: Agglomerative clustering with average linkage
3. **DBSCAN**: Density-based spatial clustering
4. **Spectral**: Graph-based clustering using normalized embeddings

---

## Community Detection

Discover communities in knowledge graph structure.

```rust
use oxirs_embed::{CommunityDetector, CommunityConfig, CommunityAlgorithm};

let config = CommunityConfig {
    algorithm: CommunityAlgorithm::Louvain,
    max_iterations: 100,
    resolution: 1.0,
    ..Default::default()
};

let mut detector = CommunityDetector::new(config);

// From graph triples
let result = detector.detect_from_triples(&triples)?;

// Or from embeddings
let result = detector.detect_from_embeddings(&embeddings)?;

println!("Communities: {}", result.num_communities);
println!("Modularity: {:.4}", result.modularity);
println!("Coverage: {:.4}", result.coverage);

// Access community members
for (community_id, members) in &result.communities {
    println!("Community {}: {:?}", community_id, members);
}
```

### Algorithms

1. **Louvain**: Modularity optimization
2. **Label Propagation**: Fast, scalable community detection
3. **Girvan-Newman**: Edge betweenness-based
4. **Embedding-Based**: Similarity-based community formation

---

## Visualization

Reduce high-dimensional embeddings to 2D/3D for visualization.

```rust
use oxirs_embed::{EmbeddingVisualizer, VisualizationConfig, ReductionMethod};

let config = VisualizationConfig {
    method: ReductionMethod::PCA,
    target_dims: 2,
    random_seed: Some(42),
    ..Default::default()
};

let mut visualizer = EmbeddingVisualizer::new(config);
let result = visualizer.visualize(&entity_embeddings)?;

// Export to JSON
let json = visualizer.export_json(&result)?;
std::fs::write("embeddings_2d.json", json)?;

// Export to CSV
let csv = visualizer.export_csv(&result)?;
std::fs::write("embeddings_2d.csv", csv)?;

// Access coordinates
for (entity, coords) in &result.coordinates {
    println!("{}: ({:.4}, {:.4})", entity, coords[0], coords[1]);
}
```

### Reduction Methods

1. **PCA**: Fast, preserves global structure
2. **t-SNE**: Preserves local structure, good for clusters
3. **UMAP**: Balanced global/local structure (approximate)
4. **Random Projection**: Fastest, good for quick visualization

---

## Interpretability

Understand and explain embedding models.

```rust
use oxirs_embed::{InterpretabilityAnalyzer, InterpretabilityConfig, InterpretationMethod};

let config = InterpretabilityConfig {
    method: InterpretationMethod::SimilarityAnalysis,
    top_k: 10,
    detailed: true,
    ..Default::default()
};

let analyzer = InterpretabilityAnalyzer::new(config);

// Similarity analysis
let analysis = analyzer.similarity_analysis("alice", &embeddings)?;
println!("Most similar: {:?}", analysis.similar_entities);
println!("Least similar: {:?}", analysis.dissimilar_entities);

// Feature importance
let importance = analyzer.feature_importance("alice", &embeddings)?;
for (dim, score) in &importance.important_features {
    println!("Dimension {}: importance = {:.4}", dim, score);
}

// Counterfactual explanation
let cf = analyzer.counterfactual_explanation("alice", "bob", &embeddings)?;
println!("Required changes: {} dimensions", cf.required_changes.len());
println!("Difficulty: {:.4}", cf.difficulty);

// Generate comprehensive report
let report = analyzer.generate_report("alice", &embeddings)?;
println!("{}", report);
```

### Analysis Types

1. **Similarity Analysis**: Find most/least similar entities
2. **Feature Importance**: Identify important embedding dimensions
3. **Counterfactual**: Explain differences between entities
4. **Nearest Neighbors**: Analyze local structure

---

## Performance Optimization

### Mixed Precision Training

Accelerate training with FP16/FP32 mixed precision.

```rust
use oxirs_embed::{MixedPrecisionTrainer, MixedPrecisionConfig};

let config = MixedPrecisionConfig {
    enabled: true,
    init_scale: 65536.0,
    dynamic_loss_scale: true,
    grad_clip_threshold: 1.0,
    gradient_accumulation: true,
    accumulation_steps: 4,
    ..Default::default()
};

let mut trainer = MixedPrecisionTrainer::new(config);

// During training
let scaled_loss = trainer.scale_loss(loss);
// ... backward pass ...
let unscaled_grads = trainer.unscale_gradients(&gradients)?;
trainer.update_parameters(&mut params, &unscaled_grads, learning_rate)?;

// Check training stability
if !trainer.is_stable() {
    println!("Warning: Training instability detected");
}

let stats = trainer.get_stats();
println!("Overflows: {}", stats.num_overflows);
println!("Current scale: {}", stats.current_scale);
```

### Quantization

Compress models with 8-bit, 4-bit, or binary quantization.

```rust
use oxirs_embed::{ModelQuantizer, QuantizationConfig, BitWidth, QuantizationScheme};

let config = QuantizationConfig {
    scheme: QuantizationScheme::Symmetric,
    bit_width: BitWidth::Int8,
    calibration: true,
    calibration_samples: 1000,
    ..Default::default()
};

let mut quantizer = ModelQuantizer::new(config);

// Calibrate (optional but recommended)
quantizer.calibrate(&embeddings)?;

// Quantize embeddings
let quantized = quantizer.quantize_embeddings(&embeddings)?;

// Check compression
let stats = quantizer.get_stats();
println!("Compression ratio: {:.2}x", stats.compression_ratio);
println!("Avg quantization error: {:.6}", stats.avg_quantization_error);
println!("Model size: {} bytes -> {} bytes",
    stats.original_size_bytes, stats.quantized_size_bytes);

// Dequantize for inference
let dequantized = quantizer.dequantize_embeddings(&quantized);

// Estimate speedup
let speedup = quantizer.estimate_speedup();
println!("Estimated inference speedup: {:.1}x", speedup);
```

---

## Integration Examples

### Complete Pipeline

```rust
use oxirs_embed::*;

async fn complete_pipeline() -> Result<()> {
    // 1. Train model
    let mut model = TransE::new(TransEConfig::default());
    // ... add triples and train ...

    // 2. Extract embeddings
    let entity_embeddings = get_entity_embeddings(&model)?;

    // 3. Cluster entities
    let clustering = EntityClustering::new(ClusteringConfig::default());
    let clusters = clustering.cluster(&entity_embeddings)?;

    // 4. Detect communities
    let detector = CommunityDetector::new(CommunityConfig::default());
    let communities = detector.detect_from_embeddings(&entity_embeddings)?;

    // 5. Visualize
    let mut visualizer = EmbeddingVisualizer::new(VisualizationConfig::default());
    let vis_result = visualizer.visualize(&entity_embeddings)?;

    // 6. Interpret
    let analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());
    let report = analyzer.generate_report("entity1", &entity_embeddings)?;

    // 7. Quantize for deployment
    let mut quantizer = ModelQuantizer::new(QuantizationConfig::default());
    let quantized = quantizer.quantize_embeddings(&entity_embeddings)?;

    println!("Pipeline complete!");
    println!("Compression: {:.2}x", quantizer.get_stats().compression_ratio);

    Ok(())
}
```

### Knowledge Graph Analytics

```rust
// Analyze entity relationships
let predictor = LinkPredictor::new(config, model);
let predictions = predictor.predict_tail("alice", "knows", &candidates)?;

// Find entity clusters
let clustering = EntityClustering::new(config);
let result = clustering.cluster(&embeddings)?;

// Interpret predictions
let analyzer = InterpretabilityAnalyzer::new(config);
let analysis = analyzer.similarity_analysis("alice", &embeddings)?;

// Visualize in 2D
let mut visualizer = EmbeddingVisualizer::new(config);
let vis = visualizer.visualize(&embeddings)?;
```

---

## Performance Considerations

### Memory Optimization

- Use quantization for large models (3-4x reduction)
- Enable gradient accumulation for batch training
- Use memory-efficient data structures from SciRS2

### Speed Optimization

- Enable parallel processing (default in most modules)
- Use mixed precision training (2-3x speedup)
- Choose appropriate algorithms:
  - Fast: Random Projection, K-Means, Label Propagation
  - Balanced: PCA, Hierarchical, Louvain
  - Accurate: t-SNE, DBSCAN, Girvan-Newman

### Scalability

- Batch operations for large-scale predictions
- Distributed training (planned feature)
- Incremental clustering and community detection

---

## Error Handling

All modules return `Result<T, Error>` for robust error handling:

```rust
match predictor.predict_tail("alice", "knows", &candidates) {
    Ok(predictions) => {
        // Handle predictions
    }
    Err(e) => {
        eprintln!("Prediction failed: {}", e);
        // Handle error
    }
}
```

---

## References

- **TransE**: Bordes et al. "Translating Embeddings for Modeling Multi-relational Data" (2013)
- **HolE**: Nickel et al. "Holographic Embeddings of Knowledge Graphs" (2016)
- **ConvE**: Dettmers et al. "Convolutional 2D Knowledge Graph Embeddings" (2018)
- **Link Prediction**: Standard benchmarks (FB15k, WN18)
- **Clustering**: Scikit-learn algorithms
- **Community Detection**: Newman, Blondel et al. (Louvain)
- **Visualization**: van der Maaten & Hinton (t-SNE), McInnes et al. (UMAP)

---

## Support

For issues, questions, or contributions:
- GitHub: https://github.com/cool-japan/oxirs
- Documentation: https://docs.rs/oxirs-embed
