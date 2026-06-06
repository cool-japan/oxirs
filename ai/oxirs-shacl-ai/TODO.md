# OxiRS SHACL-AI - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Status: Production Ready

**oxirs-shacl-ai** provides AI-enhanced SHACL validation with production-ready MLOps features.

### Quality Metrics
- **Test Status**: 1589/1589 tests passing (100% success rate)
- **Code Quality**: Zero warnings, zero errors
- **SciRS2 Compliance**: Full compliance (no direct ndarray/rand imports)
- **Code Size**: 153,329 lines total (401 Rust source files)

### Features

#### AI Models
- Graph Neural Networks for shape learning
- Transformer-based constraint generation with multi-head attention
- Reinforcement learning for optimization (Q-learning, DQN)
- VAE for synthetic test data generation
- Ensemble methods for robustness
- Meta-learning for few-shot adaptation
- Continual learning for evolving schemas
- Federated learning for privacy (Byzantine Fault Tolerance, Differential Privacy, SMPC)

#### Transfer Learning
- Pre-trained models for common domains
- Domain adaptation techniques
- Cross-lingual shape transfer (25+ languages)
- Zero-shot constraint prediction
- Multi-task learning framework (hard/soft parameter sharing, GradNorm)
- Knowledge distillation (response-based, feature-based, attention transfer)
- Transfer from OWL to SHACL

#### Active Learning
- Uncertainty sampling for validation
- Query-by-committee strategies
- Expected model change selection
- Diversity-based sampling
- Interactive labeling interface
- Budget-constrained learning
- Human-in-the-loop validation

#### Anomaly Detection
- Outlier detection in RDF data
- Novelty detection for new patterns
- Drift detection in data distributions
- Collective and contextual anomaly identification
- Explainable anomaly reports (SHAP, NLG, Decision Trees)
- Real-time anomaly streams
- Adaptive threshold tuning

#### Production Hardening
- Model versioning and registry
- A/B testing framework
- Performance benchmarking
- Scalability testing
- Security audit for AI models
- Bias detection and mitigation
- Explainability frameworks

#### Model Operations
- Automated retraining pipelines
- Model compression and quantization (INT8/INT4/FP16)
- Model drift monitoring (KL divergence, PSI, KS tests)
- Feature store integration
- Experiment tracking
- Hyperparameter optimization (Grid/Random/Bayesian/Hyperband/TPE/Genetic)
- Edge deployment support
- Model governance and compliance (GDPR, CCPA, EU AI Act)
- Production monitoring

#### Advanced SciRS2 Integration
- GPU acceleration for embeddings (CUDA, Metal)
- SIMD operations for vector/matrix computations
- Parallel processing for SPARQL queries
- Memory-efficient operations
- Performance profiling
- Metrics collection
- Cloud storage integration
- ML pipeline integration

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ GNNs, Transformer-based constraint generation, RL optimization, federated learning, 555 tests

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Enhanced GPU acceleration across all models
- ✅ Distributed training support
- ✅ Advanced model ensembles
- ✅ Real-time inference optimization
- ✅ Advanced reasoning chains
- ✅ Multi-modal constraint learning
- ✅ Pattern scorer, constraint ranker, rule generator
- ✅ 1589 tests passing

### v0.3.0 - Planned (Q2 2026)
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive ML certification suite (completed 2026-05-02)
  - **Design:** New module `ai/oxirs-shacl-ai/src/certification/` with three files:
    - `mod.rs` — public surface re-exports
    - `metrics.rs` — `ClassificationMetrics` (TP/FP/TN/FN with precision, recall, F1, accuracy, FPR, FNR, MCC), `ConstraintTypeMetrics` (per-constraint-type breakdown), `ConfusionMatrix` (multi-class record + one-vs-rest per_class_metrics)
    - `runner.rs` — `CertificationCase`, `CertificationSuite`, `CertificationRunner` (configurable F1/precision/recall thresholds; requires ≥ 10 cases; groups by constraint type)
    - `report.rs` — `CertificationReport` (overall + per-constraint metrics, ISO 8601 timestamp, Markdown rendering), `CertificationStatus` (Passed/Failed/Insufficient)
  - `lib.rs` re-exports all 8 public types at crate root
  - `tests/certification_test.rs` — 19 tests covering all scenarios
- [x] Model zoo and pretrained models (planned 2026-05-01)
  - **Goal:** Mirror oxirs-embed's `ModelZoo` for shape-learning models.
    Catalog: pretrained GAT, GraphSAGE, Graphormer-base, GT-base trained on
    a synthetic SHACL benchmark (LUBM-style ontology + violations). One-call
    load, SHA256 verified, license-tagged.
  - **Design:**
    - New module `ai/oxirs-shacl-ai/src/model_zoo/` mirroring oxirs-embed's
      structure (`mod.rs`, `registry.rs`, `manifest.rs`, `loader.rs`).
    - Shared `ModelManifest` shape **duplicated locally** (NOT re-exported from
      oxirs-embed — adding `oxirs-embed` as a dep just for a struct creates a
      heavy reverse coupling, oxirs-shacl-ai already has its own ML stack).
      Both manifests parse the same TOML schema so a future `oxirs-models`
      crate can unify them.
    - 4 manifests: `gat-shacl-base`, `graphsage-shacl-base`,
      `graphormer-shacl-base`, `gt-shacl-base`. All synthetic seeds.
    - Loader uses the shape-learning model serializer that already exists in
      this crate (`models/serialization.rs` per existing project structure)
      rather than calling oxirs-embed's `ModelRepository`. SHA256 verification
      is identical to oxirs-embed's pattern.
    - Wires the new `graph_transformer` module from the graph-transformer plan block.
  - **Files:** `ai/oxirs-shacl-ai/src/model_zoo/{mod.rs,registry.rs,manifest.rs,loader.rs}`,
    `ai/oxirs-shacl-ai/src/model_zoo/manifests/*.toml`,
    `ai/oxirs-shacl-ai/src/lib.rs`, `examples/load_shape_model.rs`,
    `tests/model_zoo_test.rs`.
  - **Prerequisites:** graph-transformer plan block must complete first (same run,
    dispatched with explicit ordering: graph-transformer first, then model-zoo).
  - **Tests:** registry parse, manifest round-trip, SHA verify, end-to-end load,
    license refusal. Use `std::env::temp_dir()`.
  - **Risk:** ordering with graph-transformer plan block. Mitigation: orchestrator
    will sequence subagents graph-transformer→model-zoo.
- [x] Large language model integration (planned 2026-05-01)
  - **Goal:** Pluggable `LlmProvider` trait powering shape-from-NL generation
    ("Person must have foaf:name; age must be ≤ 150") and constraint violation
    explanation in natural language. Three backends: `OpenAiProvider`,
    `AnthropicProvider`, `LocalProvider` (deterministic mock for tests).
  - **Design:**
    - New module `ai/oxirs-shacl-ai/src/llm/` with `mod.rs`, `provider.rs`,
      `openai.rs`, `anthropic.rs`, `local.rs`, `prompt.rs`.
    - `LlmProvider` trait:
      ```rust
      #[async_trait]
      pub trait LlmProvider: Send + Sync {
          async fn complete(&self, request: &CompletionRequest)
              -> Result<CompletionResponse, LlmError>;
          async fn embed(&self, texts: &[String])
              -> Result<Vec<Vec<f32>>, LlmError>;
          fn capabilities(&self) -> &Capabilities;
      }
      ```
    - `LocalProvider`: deterministic stub returning canned responses keyed by
      a hash of the prompt — perfect for tests, zero network. Default backend.
    - `OpenAiProvider`, `AnthropicProvider` are feature-gated (`llm-network`)
      and use existing `reqwest` already in workspace.
    - `ShapeNlGenerator::with_llm(provider).propose(constraint_text)` returns
      candidate `NodeShape` / `PropertyShape` from natural language.
    - `ConstraintExplainer::explain(violation_report, provider) -> String`
      emits a human-readable summary.
  - **Files:** `ai/oxirs-shacl-ai/src/llm/`,
    `ai/oxirs-shacl-ai/src/shape_nl_generator.rs`,
    `ai/oxirs-shacl-ai/src/explainer.rs`,
    `ai/oxirs-shacl-ai/Cargo.toml` (add `llm-network` feature),
    `examples/explain_violation.rs`,
    `tests/llm_integration_test.rs`.
  - **Prerequisites:** none — uses `async-trait`, `reqwest`, `tokio` already in
    workspace; `LocalProvider` keeps the default-feature build offline.
  - **Tests:** `LocalProvider` returns deterministic completions; shape generator
    parses provider output and produces valid `PropertyShape`; explainer emits
    non-empty NL for canned violation; capabilities surface (does provider
    support tools, embeddings, streaming).
  - **Risk:** provider API drift. Mitigation: trait stays minimal (complete /
    embed / capabilities); each provider keeps its own request/response model
    internally so the public surface doesn't churn when OpenAI/Anthropic move.
- [x] Graph transformer architectures (planned 2026-05-01)
  - **Goal:** Add Graphormer (Ying et al. 2021) and Graph Transformer (GT,
    Dwivedi & Bresson 2020) implementations alongside the existing GAT/GraphSAGE.
    Both feed into the shape-learning pipeline. Real algorithms — degree
    centrality encoding + spatial encoding (shortest path) for Graphormer;
    Laplacian PE + edge-aware attention for GT. No shortcut implementations.
  - **Design:**
    - New module `ai/oxirs-shacl-ai/src/models/graph_transformer/`:
      `mod.rs`, `graphormer.rs`, `gt.rs`, `attention.rs`, `positional_encoding.rs`.
    - **Graphormer:**
      - Centrality encoding: in/out degree → learnable embeddings added to node
        features.
      - Spatial encoding: pairwise shortest-path distance (capped at K=20),
        each distance gets a learnable scalar that's added to attention scores.
      - Edge encoding: average of edge-feature embeddings along the shortest
        path between two nodes.
      - Standard transformer block (self-attn + FFN + LayerNorm).
    - **GT:**
      - Laplacian eigenvector PE: `A` → `L = I - D^{-1/2} A D^{-1/2}` → top-k
        eigenvectors. Random sign flip per train step.
      - Multi-head attention with edge features fused additively into K/Q.
      - Sparse attention mask via graph adjacency (no full O(N²)).
    - **Backbone:** `scirs2_core::ndarray_ext` for tensors,
      `scirs2_core::random` for init (per CLAUDE.md). No `ndarray` / `rand` direct.
    - Wires into the existing `ShapeLearner` trait so it can be dropped into
      the pipeline alongside GAT.
  - **Files:** `ai/oxirs-shacl-ai/src/models/graph_transformer/{mod.rs,graphormer.rs,gt.rs,attention.rs,positional_encoding.rs}`,
    `ai/oxirs-shacl-ai/src/models/mod.rs` (register), `lib.rs` (re-export),
    `examples/graph_transformer_demo.rs`, `tests/graph_transformer_test.rs`.
  - **Prerequisites:** scirs2-core (already in workspace at 0.3.1).
  - **Backward pass strategy (verified 2026-05-01):** `scirs2_core` does NOT
    provide autograd; the workspace has no `candle` / `tch` / `burn`. Follow
    the existing hand-rolled backprop patterns already in this crate:
    `ai/oxirs-shacl-ai/src/ml/gnn.rs:1090` (`fn backward_pass(...)`) and
    `ai/oxirs-shacl-ai/src/advanced_features/graph_neural_networks.rs:311`
    (`fn backward_and_update(...)`). Each transformer block exposes a
    matching `backward(...)` returning gradients w.r.t. inputs and updating
    its own parameters via SGD.
  - **Tests:** forward-pass shape check (8-node graph, 16-d hidden, 4 heads
    → output `[8, 16]`); deterministic init via fixed seed; gradient sanity
    via hand-rolled `backward()` (numerical-gradient check on a tiny 4-node
    fixture: analytic vs. finite-difference within 1e-3); positional
    encoding orthogonality; centrality encoding vs. uniform-degree baseline
    differs; save/load round-trip via the model-zoo manifest. No autograd
    framework dependency added.
  - **Risk:** numerical instability with deep attention. Mitigation: LayerNorm
    on every block, small init scale (Xavier), gradient clipping in test;
    fp64 in unit tests for stability checks. Hand-rolled backward finite-difference
    check is the safety net.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS SHACL-AI v0.2.3 - AI-enhanced SHACL validation*

## Proposed follow-ups

- [x] Long-term support guarantees — RFC published at `docs/policies/lts.md`. (completed 2026-05-17 via RFC-001)
- [x] Enterprise features — decomposed in `docs/policies/enterprise.md`. (completed 2026-05-17 via RFC-002)
- [x] Comprehensive ML certification suite — Resolved: ML prediction accuracy vs. deterministic SHACL engine (completed 2026-05-02). See v0.3.0 item above for full design.
