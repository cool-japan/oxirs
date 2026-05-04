# OxiRS Embed - TODO

*Version: 0.3.0 | Last Updated: May 3, 2026*

## Status: Production Ready

**oxirs-embed** provides knowledge graph embeddings with multiple models, model validation, and comprehensive analytics capabilities.

### Features
- **Multiple Embedding Models** - TransE, DistMult, ComplEx, RotatE, TuckER, HolE, ConvE
- **Fine-Tuning Capabilities** - Transfer learning with knowledge distillation
- **Model Selection** - Intelligent recommendations based on dataset characteristics
- **Link Prediction** - Head/tail/relation prediction with evaluation metrics
- **Clustering Support** - K-Means, Hierarchical, DBSCAN, Spectral clustering
- **Community Detection** - Louvain, Label propagation, Girvan-Newman
- **Visualization** - PCA, t-SNE, UMAP, Random Projection
- **Interpretability** - Similarity analysis, feature importance, counterfactuals
- **Temporal Embeddings** - Time-aware knowledge graphs
- **Performance Optimization** - Mixed precision, quantization, GPU acceleration
- **1345 tests passing** with zero warnings

## Recent Accomplishments (v0.2.3)

### Production Hardening
- ✅ **Model Validation Framework** - Comprehensive validation for embedding model quality and correctness
- ✅ **Performance Monitoring** - Real-time tracking of embedding quality metrics
- ✅ **Model Serving Infrastructure** - Production-grade serving with health checks and versioning

### Observability
- ✅ **Enhanced Monitoring** - Prometheus metrics for embedding performance
- ✅ **Quality Metrics** - Automatic tracking of embedding quality over time
- ✅ **Error Handling** - Robust error handling and recovery mechanisms

### Performance
- ✅ **GPU Optimization** - Enhanced GPU acceleration for training and inference
- ✅ **Batch Processing** - Optimized batch embedding generation
- ✅ **Memory Efficiency** - Reduced memory footprint for large-scale embeddings

## Future Roadmap

### v0.3.0 - Enhanced Models & Distribution (Q2 2026)
- [x] Additional embedding models (GraphSAGE, GAT) (planned 2026-04-17)
  - **Goal:** Implement GraphSAGE (mean aggregator) and Graph Attention Network (GAT) embedding models for graph-structured RDF data
  - **Design:** GraphSageEmbedder: k-hop neighborhood sampling with mean aggregation pooling, multi-layer with ReLU activations, supports inductive embedding; GatEmbedder: multi-head attention over neighbors, attention coefficients via softmax over LeakyReLU scores, layer normalization; both implement the Embedder trait; use ndarray for matrix operations
  - **Files:** src/models/graph_sage.rs (new), src/models/gat.rs (new), src/lib.rs
  - **Prerequisites:** ndarray (workspace), scirs2-core (workspace)
  - **Tests:** GraphSAGE convergence on Cora-like toy graph; GAT attention weight sum-to-1 property test; embedding dimension correctness; inductive generalization to unseen nodes
  - **Risk:** Full GNN training is matrix-intensive; use ndarray + scirs2-core for linear algebra; no GPU in default feature set
- [x] Distributed training support (completed 2026-04-30)
  - **Goal:** Parameter-server-style distributed training prototype for embedding training across 4-8 workers.
  - **Design:** `ParameterServer` — sharded model parameters, async/sync update modes. `Worker` — pulls latest params, computes local gradients, pushes back. `ModelShardManager` — partitions embedding tables by entity-ID hash. Wire training driver to use distributed mode when `--distributed` flag set. Scale: bounded to 4-8 workers initially (toy parameter server, not full DistBelief).
  - **Files:** `src/distributed_training/{parameter_server,worker,shard_manager}.rs` (new), `tests/distributed_training.rs` (new)
  - **Tests:** unit on shard partition stability + parameter server merge (async/sync); integration 4-worker distributed training on toy graph — model quality matches single-worker baseline ± epsilon
  - **Risk:** convergence with stale gradients in async mode. Mitigation: bounded staleness + eventual consistency.
- [x] Advanced model ensemble methods (planned 2026-04-17)
  - **Goal:** Implement ensemble aggregation strategies (voting, stacking, weighted averaging) over multiple embedding models
  - **Design:** EnsembleEmbedder<E: Embedder> supporting three strategies: voting (average of all model embeddings), stacking (meta-learner trained on model outputs), and performance-weighted averaging (weights derived from validation cosine similarity); EnsembleConfig for strategy selection and hyperparameters
  - **Files:** src/ensemble.rs (new), src/lib.rs
  - **Tests:** Ensemble embedding improves over best single model on toy benchmark; stacking meta-learner convergence; weighted average with zero-weight model excluded correctly
  - **Risk:** Stacking requires hold-out validation set; document this requirement clearly
- [x] A/B testing framework (planned 2026-04-17)
  - **Goal:** Build A/B testing framework for comparing embedding model quality with configurable traffic split and statistical significance testing
  - **Design:** AbTestFramework with configurable split ratio (default 50/50); tracks per-model metrics: hit rate, latency (ns), embedding quality (cosine similarity drift, downstream accuracy); Welch's t-test for statistical significance at configurable alpha; AbTestReport with confidence intervals and recommendation
  - **Files:** src/ab_testing.rs (new), src/lib.rs
  - **Tests:** Statistical significance detection at p<0.05 with synthetic data; equal-performance models -> no significant difference; traffic split ratio correctness; report serialization
  - **Risk:** Requires sufficient sample size for t-test validity; document minimum sample requirements
- [x] Production deployment templates (completed 2026-04-30)
  - **Goal:** Helm/K8s/Docker/docker-compose templates for production embedding service deployment.
  - **Design:** `deploy/` directory with: Dockerfile (multi-stage, scratch base); docker-compose.yml (single-node + Prometheus + Grafana sidecars); helm/oxirs-embed/{Chart.yaml,values.yaml,templates/*} (K8s helm chart); k8s/{deployment,service,configmap,hpa,pdb}.yaml (raw manifests); monitoring/{prometheus.yml,grafana-dashboard.json} (scrape config + dashboard). Documentation in deploy/README.md (terse, operational).
  - **Files:** `deploy/{Dockerfile,docker-compose.yml,helm/...,k8s/...,monitoring/...,README.md}` (new operational artifacts)
  - **Tests:** smoke test — helm template render must succeed; docker build must succeed (CI-gated)
  - **Risk:** docker build runtime in CI. Mitigation: cache base layers; use scratch+musl for slim runtime image.

### v1.0.0 - LTS Release (Q2 2026)
- [~] Long-term support guarantees (policy: docs/policies/lts.md)
- [~] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein)
- [x] Comprehensive benchmarks (completed 2026-04-29)
- [x] Model zoo and pretrained models (planned 2026-05-01)
  - **Goal:** First-class `ModelZoo` API. `ModelZoo::registry().load("transe-fb15k237")`
    returns a ready-to-use `TransEModel` with verified provenance. Catalog covers
    TransE/RotatE/ComplEx/DistMult on FB15k-237 and WN18RR. SHA256 verification
    on every load; license metadata enforced.
  - **Design:**
    - New module `ai/oxirs-embed/src/model_zoo/` with `mod.rs`, `registry.rs`,
      `manifest.rs`, `loader.rs`.
    - `ModelManifest` (TOML): `name`, `model_type`, `dataset`, `dimensions`,
      `entities`, `relations`, `sha256`, `source`, `license`, `citation`,
      `version`, `created`. `serde` derive + `toml` parser.
    - `ModelZoo`: `HashMap<String, ModelManifest>` populated from
      `include_str!("manifests/*.toml")` at compile time. `Self::registry()`
      returns the global default; `Self::with_manifest_dir(path)` for custom.
    - `Loader`: opens local file (`source: "file:///..."`), or HTTP URL when
      the optional `download` feature is enabled (default OFF). Always verifies
      SHA256 before deserialization.
    - Built-in catalog: TransE-FB15k237 (200d), TransE-WN18RR (100d),
      RotatE-FB15k237 (200d), ComplEx-WN18RR (200d), DistMult-FB15k237 (200d).
      Each ships as a *small synthetic seed checkpoint* — NOT actual research
      weights — clearly documented in the manifest's `notes` field. Real weights
      go behind `cargo run --example fetch_real_weights` (network-gated).
    - **Integration with existing persistence layer (verified 2026-05-01):**
      `load_model` is a method on `persistence::ModelRepository`, NOT a free
      function. Call shape: `ModelRepository::new(base_dir, RepoConfig::default())?.load_model(&manifest.name)`.
      Returns `anyhow::Result<Box<dyn EmbeddingModel>>` and currently dispatches
      on a string `model_type` to 6 hardcoded types: TransE, DistMult, ComplEx,
      RotatE, HoLE, GNNEmbedding. Manifests must use one of these strings; the
      loader returns a typed error if `model_type` is unrecognized.
  - **Files:**
    - `ai/oxirs-embed/src/model_zoo/{mod.rs,registry.rs,manifest.rs,loader.rs}`
    - `ai/oxirs-embed/src/model_zoo/manifests/*.toml` (5 manifests + 5 .ckpt synthetic seeds)
    - `ai/oxirs-embed/src/lib.rs` (re-export)
    - `ai/oxirs-embed/examples/load_pretrained.rs`
    - `ai/oxirs-embed/tests/model_zoo_test.rs`
    - `ai/oxirs-embed/Cargo.toml` (add optional `download` feature)
  - **Prerequisites:** existing `persistence::ModelRepository` — already
    implemented (613 lines in `persistence.rs`), verified 2026-05-01.
  - **Tests:** registry parse, manifest serde round-trip, SHA256 verify
    success/fail, end-to-end load synthetic checkpoint via `ModelRepository`,
    license refusal, missing entry error, unknown `model_type` error, listing
    & search APIs. Use `std::env::temp_dir()`.
  - **Risk:** real research weights are large (>100 MB) and not redistributable
    cleanly. Mitigation: ship documented synthetic seeds for the unit suite;
    point users at `examples/fetch_real_weights.rs` for the real artifacts.
    SHA mismatch returns a typed `ModelZooError::ChecksumMismatch`, never panics.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Embed v0.2.3 - Knowledge graph embeddings with production validation*

## Proposed follow-ups

- [~] Long-term support guarantees — RFC published at `docs/policies/lts.md`.
- [~] Enterprise features — decomposed in `docs/policies/enterprise.md`.
