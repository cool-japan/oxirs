//! # Custom Domain Ingestion Example
//!
//! Demonstrates how to ingest a small domain-specific corpus, extract entities
//! and relations as RDF-like triples, build a knowledge graph, and run a simple
//! graph query — entirely with synthetic in-memory data.
//!
//! Use case: pharmaceutical knowledge graph where drugs, targets, and diseases
//! are interlinked.
//!
//! Pipeline:
//! 1. Define domain triples inline.
//! 2. Extract additional triples from free-text sentences.
//! 3. Merge all triples into a [`KgSubgraph`].
//! 4. Summarize the subgraph to produce LLM-ready context.
//! 5. Find paths between domain entities.

use oxirs_graphrag::path_finder::{KnowledgeEdge, PathFinder, PathFinderConfig};
use oxirs_graphrag::summarizer::{KgEdge, KgNode, KgSubgraph, SubgraphSummarizer};
use oxirs_graphrag::triple_extractor::{ExtractionConfig, TripleExtractor};

fn main() {
    // ── 1. Hand-coded domain triples (simulating a structured data import) ────
    let domain_triples: Vec<(&str, &str, &str, &str, &str)> = vec![
        // (subject, predicate, object, subject_type, object_type)
        ("aspirin", "inhibits", "COX1", "Drug", "Protein"),
        ("aspirin", "inhibits", "COX2", "Drug", "Protein"),
        ("ibuprofen", "inhibits", "COX1", "Drug", "Protein"),
        ("ibuprofen", "inhibits", "COX2", "Drug", "Protein"),
        (
            "COX1",
            "involved_in",
            "pain_signaling",
            "Protein",
            "Process",
        ),
        ("COX2", "involved_in", "inflammation", "Protein", "Process"),
        ("inflammation", "causes", "arthritis", "Process", "Disease"),
        ("aspirin", "treats", "arthritis", "Drug", "Disease"),
        (
            "aspirin",
            "has_side_effect",
            "bleeding_risk",
            "Drug",
            "SideEffect",
        ),
        (
            "ibuprofen",
            "has_side_effect",
            "gastric_irritation",
            "Drug",
            "SideEffect",
        ),
    ];

    // ── 2. Free-text sentences about additional domain knowledge ─────────────
    let free_text = [
        "Methotrexate is part of the DMARD class.",
        "DMARDs are known as disease modifying drugs.",
        "Celecoxib inhibits COX2.",
        "Celecoxib has fewer side effects than ibuprofen.",
    ];

    let extractor = TripleExtractor::with_defaults(ExtractionConfig {
        min_confidence: 0.2,
        max_triples_per_sentence: 5,
        normalize_predicates: true,
    });

    let text_triples: Vec<_> = free_text
        .iter()
        .flat_map(|s| extractor.extract(s))
        .collect();

    println!("=== Step 1: Domain ingestion ===");
    println!("  Hand-coded domain triples:    {}", domain_triples.len());
    println!("  Extracted from free text:     {}", text_triples.len());

    // ── 3. Build KgSubgraph ──────────────────────────────────────────────────
    let mut subgraph = KgSubgraph::new();
    let mut seen_nodes: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Add nodes and edges from domain triples
    for &(subj, pred, obj, subj_type, obj_type) in &domain_triples {
        if seen_nodes.insert(subj.to_string()) {
            subgraph.add_node(KgNode::simple(subj, subj, subj_type));
        }
        if seen_nodes.insert(obj.to_string()) {
            subgraph.add_node(KgNode::simple(obj, obj, obj_type));
        }
        subgraph.add_edge(KgEdge::unweighted(subj, obj, pred));
    }

    // Add nodes/edges from text-extracted triples
    for t in &text_triples {
        if seen_nodes.insert(t.subject.clone()) {
            subgraph.add_node(KgNode::simple(&t.subject, &t.subject, "Unknown"));
        }
        if seen_nodes.insert(t.object.clone()) {
            subgraph.add_node(KgNode::simple(&t.object, &t.object, "Unknown"));
        }
        subgraph.add_edge(KgEdge {
            source: t.subject.clone(),
            target: t.object.clone(),
            relation: t.predicate.clone(),
            weight: t.confidence,
        });
    }

    println!("\n=== Step 2: Knowledge graph statistics ===");
    println!("  Nodes: {}", subgraph.node_count());
    println!("  Edges: {}", subgraph.edge_count());

    // ── 4. Subgraph summarization ─────────────────────────────────────────────
    let summarizer = SubgraphSummarizer::new();
    let clusters = summarizer.summarize(&subgraph, 8);
    let top_relations = summarizer.extract_key_relations(&subgraph, 5);

    println!("\n=== Step 3: Subgraph summary (for LLM context) ===");
    let summary_text = summarizer.generate_text_summary(&clusters);
    println!("{}", summary_text);

    println!("\n  Top relations by frequency:");
    for (rel, count) in &top_relations {
        println!("    {:20} × {}", rel, count);
    }

    // ── 5. Domain query: paths from aspirin to arthritis ─────────────────────
    let edges: Vec<KnowledgeEdge> = domain_triples
        .iter()
        .map(|&(s, p, o, _, _)| KnowledgeEdge::new(s, p, o))
        .collect();

    let config = PathFinderConfig {
        max_depth: 4,
        max_paths: 5,
        ..Default::default()
    };
    let finder = PathFinder::new(edges, config);

    let query_source = "aspirin";
    let query_target = "arthritis";
    let paths = finder.bfs_paths(query_source, query_target, 4);

    println!("\n=== Step 4: Paths [{query_source}] → [{query_target}] ===");
    if paths.is_empty() {
        println!("  (no direct path — checking transitive)");
    } else {
        for (rank, p) in paths.iter().enumerate() {
            println!("  #{}: {}", rank + 1, p.narrative());
        }
    }

    println!("\nCustom domain ingestion example completed successfully.");
}
