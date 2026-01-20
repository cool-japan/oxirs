# OxiRS GraphRAG - TODO

*Version: 0.1.0 | Last Updated: 2026-01-06*

## Status: Production Ready

**oxirs-graphrag** provides Graph-based Retrieval Augmented Generation for knowledge graphs.

### Features
- Graph-based retrieval with community detection
- Entity retrieval and ranking
- Louvain community detection
- Context building from subgraphs
- Reciprocal Rank Fusion (RRF)
- Integration with oxirs-embed for embeddings

## Future Roadmap

### v0.2.0 - Enhanced Retrieval (Q1 2026 - Expanded)
- [ ] Leiden community detection (better than Louvain)
- [ ] Query result caching (LRU with TTL)
- [ ] Graph embedding integration (GraphSAGE, Node2Vec)
- [ ] Streaming SPARQL for large subgraphs
- [ ] Advanced fusion strategies (BM25 + Dense, ColBERT-style)
- [ ] Multi-hop reasoning with oxirs-rule
- [ ] Temporal GraphRAG (time-aware retrieval)
- [ ] Distributed GraphRAG (federated expansion)

### v0.4.0 - Production Features (Q3 2026)
- [ ] Graph summarization
- [ ] Interactive refinement with user feedback
- [ ] Explainability (attention visualization, path explanation)
- [ ] Benchmark suite against LangChain GraphRAG

### v1.0.0 - Research & Innovation (Q2 2026)
- [ ] Hybrid GNN + LLM architecture
- [ ] Neuro-symbolic fusion with oxirs-physics
- [ ] Comprehensive documentation and tutorials

## Testing & Quality
- [ ] Property-based testing with proptest
- [ ] Memory profiling for large graphs
- [ ] Concurrency testing

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS GraphRAG v0.1.0 - Graph-based RAG for knowledge graphs*
