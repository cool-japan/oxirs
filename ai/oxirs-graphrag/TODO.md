# oxirs-graphrag TODO

## High Priority

- [ ] **Leiden Community Detection**: Implement Leiden algorithm (better than Louvain)
  - Better modularity optimization
  - Handles disconnected graphs better
  - Reference: scirs2-graph potential future support

- [ ] **Query Result Caching**: LRU cache for repeated queries
  - Cache key: query embedding + config hash
  - TTL-based expiration
  - Memory-bounded cache (e.g., 1GB max)

- [ ] **Graph Embedding Integration**: Use node embeddings for similarity
  - GraphSAGE for inductive learning
  - Node2Vec for structural similarity
  - Integration with oxirs-embed

- [ ] **Streaming SPARQL**: Handle large subgraphs efficiently
  - Chunked result processing
  - Incremental community detection
  - Memory-bounded graph expansion

## Medium Priority

- [ ] **Advanced Fusion Strategies**:
  - BM25 + Dense retrieval (ColBERT-style)
  - Cross-encoder reranking
  - Query expansion with synonyms

- [ ] **Multi-hop Reasoning**:
  - Integrate with oxirs-rule for inferencing
  - RDFS/OWL reasoning during expansion
  - Rule-based subgraph pruning

- [ ] **Temporal GraphRAG**:
  - Time-aware entity retrieval
  - Temporal graph patterns
  - Evolution tracking over time

- [ ] **Distributed GraphRAG**:
  - Federated graph expansion (SERVICE)
  - Distributed community detection
  - Integration with oxirs-cluster

## Low Priority

- [ ] **Graph Summarization**:
  - Entity type hierarchies
  - Predicate abstraction
  - Schema-aware summarization

- [ ] **Interactive Refinement**:
  - User feedback loop
  - Active learning for entity selection
  - Query reformulation suggestions

- [ ] **Explainability**:
  - Attention visualization for entity selection
  - Path explanation (why this entity was included)
  - Community influence scores

## Research Ideas

- [ ] **Hybrid GNN + LLM**:
  - Train GNN on graph structure
  - Use GNN embeddings instead of static embeddings
  - End-to-end differentiable retrieval

- [ ] **Quantum Graph Algorithms**:
  - Quantum walk for graph expansion
  - Integration with scirs2-quantum (future)

- [ ] **Neuro-Symbolic Fusion**:
  - Integrate with oxirs-physics for constraint checking
  - Physics-informed retrieval

## Testing & Quality

- [ ] Property-based testing with proptest
- [ ] Benchmark suite against LangChain GraphRAG
- [ ] Memory profiling for large graphs
- [ ] Concurrency testing

## Documentation

- [ ] Tutorial: Building a GraphRAG application
- [ ] Comparison: GraphRAG vs traditional RAG
- [ ] Algorithm deep-dive: RRF, Louvain, context building
- [ ] Integration guide with existing OxiRS modules

## Dependencies to Consider

- `approx` - Floating point comparisons in tests
- `criterion` - Benchmarking
- `rayon` - Parallel community detection
