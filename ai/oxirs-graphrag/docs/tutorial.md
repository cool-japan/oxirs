# oxirs-graphrag Tutorial

This tutorial walks you through the full **extract → embed → retrieve → generate**
lifecycle of `oxirs-graphrag` — from raw text to a KGQA answer.

---

## 1. Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-graphrag = { workspace = true }
```

Or with explicit version:

```toml
[dependencies]
oxirs-graphrag = "0.3"
```

The crate's default features enable community detection and hierarchical summarization.
Streaming SPARQL support requires the `streaming` feature:

```toml
oxirs-graphrag = { version = "0.3", features = ["streaming"] }
```

---

## 2. Extracting triples from text

`triple_extractor` converts natural-language sentences into RDF-like triples using
declarative pattern matching. It requires no external models.

```rust
use oxirs_graphrag::triple_extractor::{ExtractionConfig, TripleExtractor};

let config = ExtractionConfig {
    min_confidence: 0.3,
    max_triples_per_sentence: 10,
    normalize_predicates: true,
};
let extractor = TripleExtractor::with_defaults(config);

let triples = extractor.extract(
    "Alice is a data scientist. \
     Alice works at ACME. \
     ACME has a research division.",
);

for t in &triples {
    println!("({}, {}, {})", t.subject, t.predicate, t.object);
}
```

To add custom patterns (e.g. domain-specific relations):

```rust
use oxirs_graphrag::triple_extractor::{ExtractionConfig, ExtractionPattern, TripleExtractor};

let mut extractor = TripleExtractor::with_defaults(ExtractionConfig::default());
extractor.add_pattern(ExtractionPattern::new(
    "inhibits",
    "drug",
    vec!["inhibits".to_string()],
    "target",
));
```

---

## 3. Building the knowledge graph

Once triples are extracted, load them into a `KgSubgraph` for summarization or
into a `CommunityGraph` / `PathFinder` for structural analysis.

```rust
use oxirs_graphrag::summarizer::{KgEdge, KgNode, KgSubgraph};

let mut graph = KgSubgraph::new();

// Add typed nodes
graph.add_node(KgNode::simple("alice",    "Alice",    "Person"));
graph.add_node(KgNode::simple("acme",     "ACME",     "Organization"));
graph.add_node(KgNode::simple("research", "Research", "Department"));

// Add edges
graph.add_edge(KgEdge::unweighted("alice",    "acme",     "works_at"));
graph.add_edge(KgEdge::unweighted("acme",     "research", "has_division"));

println!("Nodes: {}  Edges: {}", graph.node_count(), graph.edge_count());
```

---

## 4. Embedding the graph for similarity search

`graph_embedder` provides structural node embeddings using Node2Vec-style
biased random walks and neighborhood aggregation — no deep learning required.

```rust
use oxirs_graphrag::graph_embedder::{Graph, GraphEmbedder, WalkConfig};

let mut g = Graph::new(5); // 5 nodes
g.add_edge(0, 1, 1.0);
g.add_edge(1, 2, 1.0);
g.add_edge(2, 3, 0.5);
g.add_edge(3, 4, 1.0);
g.add_edge(0, 3, 0.5);

// Structural embeddings (topology-only, deterministic)
let embeddings = GraphEmbedder::structural_embedding(&g, 16);
println!("Node 0 embedding dim: {}", embeddings[0].vector.len());

// Random-walk based embeddings (stochastic, call `embed` for EmbeddingResult)
let walk_config = WalkConfig {
    walk_length: 10,
    walks_per_node: 5,
    return_param_p: 1.0,
    in_out_param_q: 1.0,
};
let result = GraphEmbedder::embed(&g, &walk_config, 16);
println!("Embeddings generated: {}", result.embeddings.len());
```

---

## 5. Community detection

Use `community_detector` to group related entities (Louvain-inspired greedy
label propagation):

```rust
use oxirs_graphrag::community_detector::{CommunityDetector, CommunityGraph};

let mut cg = CommunityGraph::new();
for (id, label) in [(1u64, "Alice"), (2, "Bob"), (3, "ACME"), (4, "Research")] {
    cg.add_node(id, label);
}
cg.add_edge(1, 3, 1.0); // Alice – ACME
cg.add_edge(2, 3, 1.0); // Bob – ACME
cg.add_edge(3, 4, 1.0); // ACME – Research

let detector = CommunityDetector::new(/*min_size=*/ 2, /*max_iter=*/ 50);
let result = detector.detect(&mut cg);

println!("Communities: {}", result.communities.len());
println!("Modularity:  {:.4}", result.modularity);

for community in &result.communities {
    println!("  Community {:>2}: {} members", community.id, community.size());
}
```

For the full community pipeline on top of `Triple` data (SPARQL-backed),
see the `graph::community` module which implements Leiden algorithm with
the `CommunityConfig` struct.

---

## 6. Running a KGQA query

Use `PathFinder` to answer "how is X connected to Y?" queries over the
in-memory graph:

```rust
use oxirs_graphrag::path_finder::{KnowledgeEdge, PathFinder, PathFinderConfig};

let edges = vec![
    KnowledgeEdge::new("Alice",  "works_at",   "ACME"),
    KnowledgeEdge::new("ACME",   "located_in", "Berlin"),
    KnowledgeEdge::new("Bob",    "knows",       "Alice"),
];

let config = PathFinderConfig {
    max_depth: 3,
    max_paths: 10,
    ..Default::default()
};
let finder = PathFinder::new(edges, config);

let paths = finder.bfs_paths("Bob", "Berlin", 3);
for path in &paths {
    println!("{}", path.narrative());
    // e.g. "Bob —[knows]→ Alice —[works_at]→ ACME —[located_in]→ Berlin"
}
```

For path scoring with predicate relevance weights, populate
`PathFinderConfig::predicate_weights`.

---

## 7. Subgraph summarization (LLM context)

Before sending to an LLM, compress the retrieved subgraph into a readable
summary using `SubgraphSummarizer`:

```rust
use oxirs_graphrag::summarizer::{KgEdge, KgNode, KgSubgraph, SubgraphSummarizer};

let mut graph = KgSubgraph::new();
graph.add_node(KgNode::simple("alice", "Alice", "Person"));
graph.add_node(KgNode::simple("acme",  "ACME",  "Organization"));
graph.add_edge(KgEdge::unweighted("alice", "acme", "works_at"));

let summarizer = SubgraphSummarizer::new();

// Cluster nodes by type
let clusters = summarizer.summarize(&graph, 10);

// Generate natural-language paragraph
let context = summarizer.generate_text_summary(&clusters);
println!("{}", context);

// Top relation types for prompt engineering
let top_rels = summarizer.extract_key_relations(&graph, 5);
for (rel, count) in &top_rels {
    println!("  {rel}: {count}×");
}
```

---

## 8. Integrating with an LLM for RAG

For full RAG integration implement the four engine traits and construct
`GraphRAGEngine`:

```rust,ignore
use oxirs_graphrag::{
    GraphRAGConfig, GraphRAGEngine,
    VectorIndexTrait, EmbeddingModelTrait, SparqlEngineTrait, LlmClientTrait,
};
use std::sync::Arc;

// Implement the four traits on your types:
//   VectorIndexTrait  — wraps oxirs-vec HNSW index
//   EmbeddingModelTrait — wraps oxirs-embed model
//   SparqlEngineTrait  — wraps oxirs-arq engine
//   LlmClientTrait    — wraps your LLM HTTP client

let config = GraphRAGConfig {
    top_k: 20,
    expansion_hops: 2,
    max_subgraph_size: 500,
    enable_communities: true,
    vector_weight: 0.7,
    keyword_weight: 0.3,
    ..Default::default()
};

let engine = GraphRAGEngine::new(
    Arc::new(my_vec_index),
    Arc::new(my_embedder),
    Arc::new(my_sparql_engine),
    Arc::new(my_llm),
    config,
);

// Issue a KGQA query
let result = engine.query("What safety issues affect battery cells?").await?;
println!("Answer:     {}", result.answer);
println!("Confidence: {:.2}", result.confidence);
println!("Seed nodes: {}", result.seeds.len());
println!("Subgraph:   {} triples", result.subgraph.len());
```

The engine pipeline is:
1. Embed query → vector KNN search
2. BM25 keyword search via SPARQL REGEX
3. RRF fusion → seed entities
4. N-hop SPARQL graph expansion → subgraph
5. Community detection → hierarchical clusters
6. Context building → LLM prompt
7. LLM generation → answer + citations

---

## Next steps

- See the [architecture overview](architecture.md) for module internals.
- Run the provided examples: `cargo run --example kgqa_basic -p oxirs-graphrag`
- Browse the full API at `cargo doc -p oxirs-graphrag --open`.
