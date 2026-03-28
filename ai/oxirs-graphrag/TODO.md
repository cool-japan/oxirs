# OxiRS GraphRAG - TODO

*Version: 0.2.3 | Last Updated: 2026-03-16*

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

### v0.2.3 - Current Release (March 16, 2026)
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
- [ ] Graph summarization
- [ ] Interactive refinement with user feedback
- [ ] Explainability (attention visualization, path explanation)
- [ ] Benchmark suite against LangChain GraphRAG

### v1.0.0 - Planned (Q2 2026)
- [ ] Hybrid GNN + LLM architecture
- [ ] Neuro-symbolic fusion with oxirs-physics
- [ ] Comprehensive documentation and tutorials

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS GraphRAG v0.2.3 - Graph-based RAG for knowledge graphs*
