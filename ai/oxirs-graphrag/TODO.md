# OxiRS GraphRAG - TODO

*Version: 0.3.2 | Last Updated: July 12, 2026*

## Status: Production Ready

**oxirs-graphrag** provides Graph-based Retrieval Augmented Generation for knowledge graphs.

### Features
- Graph-based retrieval with community detection
- Entity retrieval and ranking
- Louvain community detection
- Context building from subgraphs
- Reciprocal Rank Fusion (RRF)
- Integration with oxirs-embed for embeddings

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ Graph-based retrieval, Louvain community detection, context building, RRF

### v0.2.3 - Released (March 16, 2026)
- ✅ Leiden community detection (graph/community.rs with full refinement phase)
- ✅ Query result caching (LRU with TTL — cache/query_cache.rs)
- ✅ Graph embedding integration (Node2Vec — embeddings/node2vec.rs with alias sampling)
- ✅ ColBERT-style reranker (fusion/colbert_reranker.rs)
- ✅ Hybrid retrieval (fusion/hybrid_retrieval.rs BM25 + dense)
- ✅ Multi-hop reasoning (reasoning/multihop.rs)
- ✅ Temporal knowledge graph retrieval (temporal/temporal_retrieval.rs)
- ✅ Streaming SPARQL for large subgraphs (retrieval/streaming_sparql.rs)
- ✅ Community detector, triple extractor, knowledge fusion
- ✅ 935 tests passing

### v0.4.0 - Planned (Q3 2026)
- [x] Graph summarization (planned 2026-04-17)
  - **Goal:** Build `GraphSummarizer` that compresses a large knowledge graph subgraph into a compact summary for LLM context via community detection, centrality scoring, and predicate frequency ranking
  - **Design:** `GraphSummarizer { max_nodes: usize, max_triples: usize }`; pipeline: (1) community detection (reuse Leiden from graph/community.rs), (2) per-community representative node selection by PageRank-like in-degree centrality, (3) predicate frequency ranking to select most informative relations, (4) output `GraphSummary { entities: Vec<String>, relations: Vec<(String, String, String)>, community_labels: Vec<String> }`; `to_text()` serializes as natural-language paragraph for LLM
  - **Files:** src/summarizer.rs (new), src/lib.rs
  - **Tests:** Summary respects `max_nodes` limit; community labels captured; predicate frequency ordering correct; `to_text()` non-empty on non-empty graph; empty graph returns empty summary gracefully
  - **Risk:** Large graphs may be slow — use rayon parallel iteration for community centroid computation; cap at max_nodes early
- [x] Interactive refinement with user feedback (planned 2026-04-17)
  - **Goal:** Implement feedback loop allowing users to mark retrieved triples as relevant/irrelevant, with the retriever adapting weights accordingly for subsequent queries in the same session
  - **Design:** `FeedbackSession { positive: HashSet<Triple>, negative: HashSet<Triple>, weights: HashMap<TripleId, f64> }`; `record_feedback(triple, Relevance::Positive | Negative)`; `apply_feedback()` boosts positive triples' scores and penalizes negative in the next retrieval pass via multiplicative weight adjustment; session-scoped (not persisted across sessions)
  - **Files:** src/feedback.rs (new), src/lib.rs
  - **Tests:** Positive feedback boosts retrieval score; negative feedback reduces score below threshold; neutral (no feedback) leaves score unchanged; weight bounds (0.0–2.0 clamp)
  - **Risk:** Feedback bias accumulation — clamp weights to [0.1, 2.0] to prevent runaway scores
- [x] Explainability (attention visualization, path explanation) (planned 2026-04-17)
  - **Goal:** Add explanation generation for GraphRAG answers: attention weight contributions, shortest explanation paths through the knowledge graph, and provenance chains linking answers back to source triples
  - **Design:** `ExplainabilityEngine`; `explain(context_triples: &[Triple], answer_entity: &str, max_hops: usize) -> Explanation`; `Explanation { attention_weights: Vec<(Triple, f64)>, paths: Vec<Vec<Triple>>, provenance: Vec<String> }`; attention_weights computed as cosine similarity of triple embedding to answer embedding, normalized to sum≈1.0; path finding via BFS on triple adjacency graph limited to max_hops and max_paths; provenance captures source graph URIs
  - **Files:** src/explainability.rs (new), src/lib.rs
  - **Tests:** Attention weights sum to ≈1.0 (tolerance 1e-6); explanation path connects answer entity to context; path length ≤ max_hops; provenance non-empty when context provided; empty context returns empty explanation gracefully
  - **Risk:** BFS on large graphs — enforce max_hops (default 3) and max_paths (default 5) caps
- [x] Benchmark suite against LangChain GraphRAG (planned 2026-04-30)
  - **Goal:** Stand-up benchmark suite comparing oxirs-graphrag against LangChain GraphRAG on standard KGQA datasets.
  - **Design:** Phase 1 (always): standalone KGQA benchmark on standard datasets so we have repeatable absolute numbers. Datasets: small WebQSP-derived subset + synthetic ontology-QA generator. Vendor under `benches/fixtures/kgqa-bench/`. Metrics: Hits@1, Hits@5, MRR, p50/p95/p99 latency, end-to-end throughput. Phase 2 (optional, gated on `LANGCHAIN_REF_FIXTURES` env var): when env var points at a directory of LangChain reference JSON outputs (captured offline by operator running pinned LangChain version), assert oxirs-graphrag Hits@5 within 5pp of reference. Skipped by default — no Python dep in CI. Capture-script lives at `benches/scripts/capture_langchain_reference.py` (vendored, never run in CI; only operators run locally).
  - **Files:** `benches/langchain_kgqa.rs` (new), `benches/fixtures/kgqa-bench/` (vendored datasets), `benches/scripts/capture_langchain_reference.py` (new — operator-only)
  - **Tests:** benchmark suite runs (criterion); when LANGCHAIN_REF_FIXTURES is set, assert oxirs Hits@5 within 5pp of LangChain on WebQSP subset
  - **Risk:** LangChain reference generation is operator-side. Mitigation: ship deterministic standalone baseline; comparative phase is optional.

### v1.0.0 - Planned (Q2 2026)
- [x] Hybrid GNN+LLM architecture (planned 2026-05-01, 3 phases) — all phases DONE 2026-05-01 (1017 tests pass)
  - **Phase a — GNN encoder** (DONE 2026-05-01):
    - **Goal:** GraphSAGE encoder over the KG producing fixed-dim entity embeddings. Hand-rolled backprop (no autograd dep). Link-prediction objective. Finite-difference gradient check.
    - **Design:** New module `src/gnn_encoder/{mod.rs,graphsage.rs,aggregator.rs,sampler.rs}`. Mean aggregator: `h_v^l = σ(W^l · CONCAT(h_v^{l-1}, MEAN({h_u : u ∈ N(v)})))`. Neighbour sampler K=10. Init via `scirs2_core::random` Xavier. Loss: margin-ranking `max(0, 1 - sim(h_s, h_o+) + sim(h_s, h_o-))`. SGD with gradient clipping max-norm 1.0. Hand-rolled backward following `ai/oxirs-shacl-ai/src/ml/gnn.rs:1090`.
    - **Files:** `src/gnn_encoder/{mod.rs,graphsage.rs,aggregator.rs,sampler.rs}`, `src/lib.rs` (+re-export), `examples/gnn_encoder_demo.rs`, `tests/gnn_encoder_test.rs`.
    - **Tests:** forward pass 8-node → `[8, hidden_dim]`; deterministic init; FD gradient check 1e-3; sampler ≤K; loss ↓≥30% over 50 epochs.
  - **Phase b — LLM head + frozen GNN** (planned 2026-05-01, after phase a):
    - **Goal:** Soft-prompt projector (`ℝ^{gnn_dim} → ℝ^{prompt_dim}`) that composes GNN embeddings into LLM context. GNN frozen. Local `LlmProvider` trait declared in graphrag (no cross-crate ML dep). `HybridLlmHead<P: LlmProvider>`. Offline `LocalProvider` for tests.
    - **Design:** New module `src/hybrid/{mod.rs,soft_prompt.rs,llm_head.rs,provider.rs}`. Top-K entity retrieval → GNN embeddings → projection → textual surrogate soft-prompt prepended to LLM input. Cross-entropy loss on KGQA answer tokens for projector training.
    - **Files:** `src/hybrid/{mod.rs,soft_prompt.rs,llm_head.rs,provider.rs}`, `src/lib.rs` (+re-export), `examples/hybrid_kgqa.rs`, `tests/hybrid_test.rs`.
    - **Tests:** projector forward `[k, prompt_dim]`; frozen-encoder check; LocalProvider KGQA E2E; cross-entropy loss ↓ over 20 epochs; capabilities surface.
  - **Phase c — joint training scaffold** (planned 2026-05-01, after phase b):
    - **Goal:** Alternating GNN-encoder + LLM-head projector training with param-group freeze controls. `JointTrainer` with `freeze_gnn()` / `freeze_projector()` toggles, `Schedule::AlternateEpoch`, `Schedule::Curriculum`. Hand-rolled backward composition through both stacks.
    - **Design:** New file `src/hybrid/joint_trainer.rs`. `TrainingHistory { epoch_loss, gnn_grad_norm, projector_grad_norm }`. 50-epoch demo on toy 4-entity KG.
    - **Files:** `src/hybrid/joint_trainer.rs`, `src/hybrid/mod.rs` (re-export), `examples/joint_training_demo.rs`, `tests/joint_training_test.rs`.
    - **Tests:** freeze toggle; alternating schedule param hashes; joint loss ↓ over 50 epochs; history records non-zero norms only for unfrozen group; curriculum warmup (5 epochs projector-only); FD gradient check on combined stack 5e-3.
- [x] Neuro-symbolic fusion with oxirs-physics (DONE 2026-05-01)
  - **Implementation:** New module `src/neuro_symbolic/{mod.rs,physics_context.rs,pinn_scorer.rs,retriever.rs}`
  - **Design:** PINN-driven scoring blending GNN cosine similarity (neural) with physics plausibility (symbolic) via `combined = (1-λ)·neural + λ·physics`. Four physics domains: ThermalDiffusion (Fourier number), FluidFlow (Reynolds number), StructuralMechanics (Hooke's law), Electromagnetic (Ohm's law). Pure arithmetic — no oxirs-physics dependency.
  - **Tests:** 14 integration tests in `tests/neuro_symbolic_test.rs` + 12 unit tests embedded in source files; all passing.
  - **Example:** `examples/pinn_retrieval.rs` — 4-node thermal KG with ranked entity output.
- [x] Comprehensive documentation and tutorials (planned 2026-05-01)
  - **Goal:** Top-level rustdoc that walks users through the full extract → embed
    → retrieve → generate lifecycle. Three end-to-end runnable examples.
    Architecture overview document. Doctests so the docs cannot rot silently.
  - **Design:**
    - Rewrite `lib.rs` `//!` doc with module-level overview, quickstart code, and
      a doctest covering a full mini-pipeline on a synthetic 8-node graph.
    - Three new examples:
      `kgqa_basic.rs` — Wikipedia-subset KGQA over a vendored mini-RDF dataset.
      `custom_domain_ingestion.rs` — extract triples from a domain corpus and
      run a query end-to-end.
      `hybrid_retrieval.rs` — combine SPARQL + vector search; demonstrate the
      community detection module.
    - `ai/oxirs-graphrag/docs/{tutorial.md,architecture.md}` (new permanent
      project docs — per skill rules, scratch goes to /tmp/, but permanent doc
      assets in the crate's docs/ are appropriate).
    - README quickstart + cross-links.
  - **Files:** `src/lib.rs`, `README.md`, `docs/tutorial.md`, `docs/architecture.md`,
    `examples/{kgqa_basic.rs,custom_domain_ingestion.rs,hybrid_retrieval.rs}`,
    `tests/examples_compile_test.rs`.
  - **Prerequisites:** none — community detection, embedding integration, and
    knowledge retrieval modules are already implemented per project memory.
  - **Tests:** lib.rs doctest must pass; all 3 examples compile + run via
    `cargo test --examples -p oxirs-graphrag`; doc-link check via `cargo doc -p oxirs-graphrag`.
  - **Risk:** docs drift from API. Mitigation: keep all examples as runnable
    integration tests; doctests in lib.rs cover the public API.

### v0.3.2 - Current Release (July 12, 2026)
- [x] Deterministic community detection — `graph::community::CommunityDetector` (Louvain + Leiden) no longer depends on HashMap/HashSet iteration order; node processing order and tie-breaking are now a pure function of `CommunityConfig::random_seed` via `scirs2_core::random::seeded_rng`
- [x] Modularity floor guarantee — Louvain/Leiden now compare their greedy result against the trivial single-community partition's modularity and fall back to it when the greedy result would score lower, so community detection never returns a partition worse than "no structure"
- [x] Modularity calculation bug fix — `calculate_modularity` rewritten to the standard per-community Newman-Girvan form (O(edges + nodes)); the duplicate, diverging O(n²) computation in hierarchical detection now delegates to the same function
- ✅ 1125 tests passing

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS GraphRAG v0.3.2 - Graph-based RAG for knowledge graphs*

## Proposed follow-ups

- [x] Hybrid GNN+LLM architecture — OVERSIZED: split into dedicated planning rounds: (a) GNN encoder over the knowledge graph, (b) LLM head with frozen GNN embeddings as soft prompt, (c) joint training loop. (resolved — see completed entry above)
- [x] Neuro-symbolic fusion with physics — VAGUE: clarification needed. What does "fusion" mean here? Rule grounding? PINN-driven retrieval? Please specify. (resolved — see completed entry above)
- [x] Hybrid GNN+LLM phase d — GGUF model loader + LoRA adapter fine-tuning scaffold (completed 2026-05-02)
  - **Implementation:** `src/model_loader/{mod.rs,gguf_parser.rs,registry.rs}` (feature-gated `gguf-loader`) + `src/hybrid/lora.rs` (always-on).
  - **GGUF parser:** Pure-Rust little-endian v2/v3 metadata reader. Parses magic, version, n_kv key-value entries (types 0–12 + array), n_tensor info records (name, dims, data_type, offset). No weights loaded — lazy metadata only. `GgufMetadata::total_params()` and `estimated_size_bytes()` helpers. `GgufModelArch` extracts architecture, context_length, embedding_length, head_count, layer_count, vocab_size from KV map.
  - **ModelRegistry:** `RwLock<HashMap<String, ModelInfo>>` keyed by name-based `ModelHandle`. `register()` parses the GGUF file; `register_with_metadata()` accepts pre-parsed metadata (used in tests). `get()` / `get_by_name()` / `list()` / `remove()` / `len()` / `is_empty()`.
  - **LoRA adapter:** A `[d_in, rank]`, B `[rank, d_out]` matrices. Xavier-uniform A, zero B. `forward_delta()`: `scale * (input @ A) @ B`. `backward()`: hand-rolled chain rule accumulates `grad_A`, `grad_B`. `sgd_step()`, `zero_grad()`, `grad_norm()`. `LoraTrainer` wraps adapter with a learning rate and drives `train_epoch()`.
  - **Tests:** 16 model_loader integration tests (feature-gated) + 13 LoRA tests (always-on). Includes FD gradient checks on both grad_B and grad_A.
