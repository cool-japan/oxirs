# OxiRS-Embed Examples

This directory contains comprehensive examples demonstrating the capabilities of the **oxirs-embed** knowledge graph embedding library.

## 📚 Available Examples

### Core Embedding Examples

#### 1. **Basic Usage** (`basic_usage.rs`)
**Complexity:** ⭐ Beginner
**Run:**
```bash
cargo run --example basic_usage --features basic-models
```

**What it demonstrates:**
- Creating and training a simple TransE model
- Adding triples to build a knowledge graph
- Getting entity and relation embeddings
- Computing similarity scores
- Basic knowledge graph completion

**Use this when:** You're new to knowledge graph embeddings and want to understand the basics.

---

#### 2. **Embed Basic Usage** (`embed_basic_usage.rs`)
**Complexity:** ⭐ Beginner
**Run:**
```bash
cargo run --example embed_basic_usage --features basic-models
```

**What it demonstrates:**
- Alternative introduction to the embedding API
- Training workflow
- Entity embedding extraction
- Simple triple scoring

**Use this when:** You want a quick start guide with minimal complexity.

---

### Advanced Model Examples

#### 3. **HolE Model Demo** (`hole_model_demo.rs`)
**Complexity:** ⭐⭐ Intermediate
**Run:**
```bash
cargo run --example hole_model_demo --features basic-models
```

**What it demonstrates:**
- HolE (Holographic Embeddings) model
- Geographic knowledge graph construction
- Link prediction (tail, head, relation)
- Entity clustering with K-Means
- Embedding similarity analysis
- Domain discovery

**Use this when:** You want to explore advanced embedding models that use circular correlation for more expressive representations.

**Key Features:**
- Handles symmetric and asymmetric relations
- Efficient computation via Fourier transform
- Excellent for complex relational patterns
- Combines link prediction with clustering

---

### Application-Specific Examples

#### 4. **Link Prediction Demo** (`link_prediction_demo.rs`) ✨ NEW
**Complexity:** ⭐⭐⭐ Advanced
**Run:**
```bash
cargo run --example link_prediction_demo --features basic-models
```

**What it demonstrates:**
- **Comprehensive link prediction** for knowledge graph completion
- **Tail entity prediction** (object prediction): `(subject, relation, ?)`
- **Head entity prediction** (subject prediction): `(?, relation, object)`
- **Relation prediction**: `(subject, ?, object)`
- **Batch prediction** for efficient processing
- **Evaluation metrics**: MRR, Mean Rank, Hits@K
- **Model comparison**: TransE vs HolE

**Real-World Use Cases:**
- Academic collaboration prediction
- Publication venue recommendation
- Research area discovery
- Missing fact completion in knowledge bases

**Example Queries:**
```rust
// Who does Alice collaborate with?
predictor.predict_tail("alice", "collaborates_with", &candidates)?;

// Who is affiliated with Stanford?
predictor.predict_head("affiliated_with", "stanford", &candidates)?;

// What is the relationship between Alice and Stanford?
predictor.predict_relation("alice", "stanford", &relation_candidates)?;
```

**Key Metrics Explained:**
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank (higher is better)
- **Hits@K**: Percentage of correct predictions in top-K
- **Mean Rank**: Average rank of correct answer (lower is better)

---

#### 5. **Clustering Demo** (`clustering_demo.rs`) ✨ NEW
**Complexity:** ⭐⭐⭐ Advanced
**Run:**
```bash
cargo run --example clustering_demo --features basic-models
```

**What it demonstrates:**
- **4 clustering algorithms**:
  - **K-Means**: Fast, spherical clusters
  - **Hierarchical**: Nested structure, no K needed
  - **DBSCAN**: Arbitrary shapes, handles noise
  - **Spectral**: Graph-based, non-convex clusters
- **Cluster quality metrics**:
  - Silhouette score (quality measure)
  - Inertia (K-Means)
  - Noise detection (DBSCAN)
- **Domain discovery** and purity analysis
- **Centroid analysis** for cluster characterization
- **Algorithm comparison** with performance metrics

**Real-World Use Cases:**
- Entity type discovery (automatically find categories)
- Knowledge graph organization
- Semantic grouping for navigation
- Anomaly detection (outliers)
- Recommendation systems (user/item grouping)

**Example Output:**
```text
Cluster 0: [python, java, rust, javascript]  -> Programming Languages
Cluster 1: [paris, london, berlin, rome]     -> Cities
Cluster 2: [apple, banana, orange]           -> Fruits
Cluster 3: [dog, cat, elephant]              -> Animals
```

**Algorithm Selection Guide:**
| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| K-Means | Known cluster count | Fast, scalable | Requires K, spherical |
| Hierarchical | Small datasets, nested structure | No K needed | Slow, O(n²) |
| DBSCAN | Arbitrary shapes, noise | Finds outliers | Parameter sensitive |
| Spectral | Graph structure, non-convex | Complex shapes | Slow, requires K |

---

### Domain-Specific Examples

#### 6. **Biomedical Embedding Demo** (`biomedical_embedding_demo.rs`)
**Complexity:** ⭐⭐⭐ Advanced
**Run:**
```bash
cargo run --example biomedical_embedding_demo --features basic-models
```

**What it demonstrates:**
- Medical knowledge graph construction
- Disease-symptom relationships
- Drug-target interactions
- Clinical decision support
- Biomedical entity linking

**Use this when:** Working with healthcare or biomedical data.

---

#### 7. **Multimodal Embedding Demo** (`multimodal_embedding_demo.rs`)
**Complexity:** ⭐⭐⭐ Advanced
**Run:**
```bash
cargo run --example multimodal_embedding_demo --features basic-models
```

**What it demonstrates:**
- Combining multiple data modalities
- Text + Knowledge graph embeddings
- Cross-modal similarity
- Multimodal retrieval

**Use this when:** You need to integrate different types of data.

---

### System and Performance Examples

#### 8. **Advanced Embedding Server** (`advanced_embedding_server.rs`)
**Complexity:** ⭐⭐⭐⭐ Expert
**Run:**
```bash
cargo run --example advanced_embedding_server --features basic-models
```

**What it demonstrates:**
- Production-ready HTTP server
- REST API for embeddings
- Batch processing endpoints
- Model management
- Caching and optimization
- Concurrent request handling

**Use this when:** Building a production embedding service.

---

#### 9. **GPU Acceleration Demo** (`gpu_acceleration_demo.rs`)
**Complexity:** ⭐⭐⭐⭐ Expert
**Run:**
```bash
cargo run --example gpu_acceleration_demo --features gpu
```

**What it demonstrates:**
- GPU-accelerated training
- CUDA/Metal backend usage
- Performance benchmarking
- Memory management for large graphs

**Use this when:** Processing large-scale knowledge graphs (>100K entities).

**Requirements:**
- CUDA toolkit (NVIDIA) or Metal (Apple Silicon)
- GPU with sufficient VRAM

---

#### 10. **Comprehensive Benchmark** (`comprehensive_benchmark.rs`)
**Complexity:** ⭐⭐⭐ Advanced
**Run:**
```bash
cargo run --release --example comprehensive_benchmark --features basic-models
```

**What it demonstrates:**
- Performance comparison of embedding models
- Scalability testing
- Memory usage profiling
- Training time analysis
- Link prediction accuracy

**Use this when:** Evaluating which model to use for your application.

---

#### 11. **Integrated AI Platform Demo** (`integrated_ai_platform_demo.rs`)
**Complexity:** ⭐⭐⭐⭐ Expert
**Run:**
```bash
cargo run --example integrated_ai_platform_demo --features basic-models
```

**What it demonstrates:**
- Full AI stack integration
- RAG (Retrieval-Augmented Generation)
- Vector search
- Semantic caching
- End-to-end knowledge graph AI system

**Use this when:** Building a complete AI-powered knowledge system.

---

## 🎯 Quick Start Guide

### For Beginners

1. Start with **basic_usage.rs** to understand the fundamentals
2. Move to **hole_model_demo.rs** to explore advanced models
3. Try **link_prediction_demo.rs** for practical applications

### For Link Prediction Tasks

Go directly to **link_prediction_demo.rs** which covers:
- All three prediction types (tail, head, relation)
- Batch processing for efficiency
- Evaluation metrics
- Best practices

### For Entity Organization

Use **clustering_demo.rs** which provides:
- 4 different clustering algorithms
- Quality metrics
- Domain discovery
- Algorithm selection guidance

### For Production Deployments

Check out:
1. **advanced_embedding_server.rs** for HTTP API
2. **gpu_acceleration_demo.rs** for performance
3. **comprehensive_benchmark.rs** for evaluation

---

## 🚀 Running Examples

### Basic Run
```bash
cargo run --example <example_name> --features basic-models
```

### With GPU Support
```bash
cargo run --example gpu_acceleration_demo --features gpu
```

### Release Mode (Faster)
```bash
cargo run --release --example <example_name> --features basic-models
```

### With Logging
```bash
RUST_LOG=info cargo run --example <example_name> --features basic-models
```

---

## 📊 Feature Comparison

| Example | Link Prediction | Clustering | Visualization | GPU | Production-Ready |
|---------|----------------|------------|---------------|-----|------------------|
| basic_usage | ✅ | ❌ | ❌ | ❌ | ❌ |
| hole_model_demo | ✅ | ✅ | ❌ | ❌ | ❌ |
| link_prediction_demo | ✅✅✅ | ❌ | ❌ | ❌ | ❌ |
| clustering_demo | ❌ | ✅✅✅ | ❌ | ❌ | ❌ |
| biomedical_embedding_demo | ✅ | ✅ | ✅ | ❌ | ❌ |
| advanced_embedding_server | ✅ | ✅ | ❌ | ❌ | ✅ |
| gpu_acceleration_demo | ✅ | ✅ | ❌ | ✅ | ✅ |
| integrated_ai_platform_demo | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 💡 Tips and Best Practices

### Model Selection

**TransE:**
- ✅ Simple, fast, interpretable
- ✅ Good for hierarchical relations
- ❌ Struggles with symmetric relations
- **Use for:** Initial prototyping, simple KGs

**HolE:**
- ✅ Handles symmetric and asymmetric relations
- ✅ Efficient via FFT
- ✅ Expressive representations
- **Use for:** Complex relational patterns

**ConvE:**
- ✅ Very expressive with CNNs
- ✅ State-of-the-art accuracy
- ❌ Slower training
- **Use for:** Maximum accuracy, research

### Hyperparameter Tuning

**Embedding Dimensions:**
- Small KG (<1K entities): 50-100 dims
- Medium KG (1K-10K): 100-200 dims
- Large KG (>10K): 200-500 dims

**Learning Rate:**
- Start with 0.01
- Reduce if loss oscillates
- Increase if converging too slowly

**Training Epochs:**
- Monitor loss convergence
- Use early stopping
- Typical range: 50-300 epochs

### Performance Optimization

1. **Use batch processing** for multiple queries
2. **Enable parallel processing** in config
3. **GPU acceleration** for large graphs
4. **Model quantization** to reduce memory (see quantization tests)
5. **Mixed precision training** for faster convergence

---

## 📖 Additional Resources

### Documentation
- [API Documentation](https://docs.rs/oxirs-embed)
- [User Guide](../README.md)
- [Integration Tests](../tests/integration_new_features.rs)

### Research Papers
- **TransE**: "Translating Embeddings for Modeling Multi-relational Data" (Bordes et al., 2013)
- **HolE**: "Holographic Embeddings of Knowledge Graphs" (Nickel et al., 2016)
- **ConvE**: "Convolutional 2D Knowledge Graph Embeddings" (Dettmers et al., 2018)

### Related Projects
- [SciRS2](https://github.com/cool-japan/scirs) - Scientific computing foundation
- [OxiRS-Core](../../core/oxirs-core) - RDF/SPARQL engine
- [OxiRS-Vec](../../engine/oxirs-vec) - Vector search engine

---

## 🐛 Troubleshooting

### Common Issues

**"Out of memory" errors:**
- Reduce embedding dimensions
- Use smaller batch sizes
- Enable model quantization
- Use GPU if available

**Slow training:**
- Enable GPU acceleration
- Reduce embedding dimensions
- Use smaller learning rate
- Enable mixed precision training

**Poor link prediction accuracy:**
- Increase embedding dimensions
- Train for more epochs
- Try different model (HolE or ConvE)
- Add more training data

### Getting Help

- Check the [examples](.) for reference implementations
- Review [integration tests](../tests/) for correct usage
- Open an issue on [GitHub](https://github.com/cool-japan/oxirs)

---

## 🎓 Learning Path

### Week 1: Foundations
- Day 1-2: `basic_usage.rs` - Understand core concepts
- Day 3-4: `embed_basic_usage.rs` - API familiarity
- Day 5-7: `hole_model_demo.rs` - Advanced models

### Week 2: Applications
- Day 1-3: `link_prediction_demo.rs` - Knowledge completion
- Day 4-5: `clustering_demo.rs` - Entity organization
- Day 6-7: `biomedical_embedding_demo.rs` - Domain application

### Week 3: Production
- Day 1-3: `advanced_embedding_server.rs` - Build API
- Day 4-5: `gpu_acceleration_demo.rs` - Optimize performance
- Day 6-7: `integrated_ai_platform_demo.rs` - Full stack

---

## 🌟 Contributing

We welcome contributions! If you have an interesting example:

1. Fork the repository
2. Create your example in `examples/`
3. Add documentation following this README format
4. Submit a pull request

---

**Happy embedding!** 🚀

For more information, visit the [OxiRS project](https://github.com/cool-japan/oxirs).
