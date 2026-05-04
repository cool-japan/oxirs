//! # Knowledge Graph QA — Basic Example
//!
//! Demonstrates a standalone KGQA pipeline using in-memory data:
//!
//! 1. Extract triples from a small synthetic text corpus.
//! 2. Build a community graph from those triples.
//! 3. Detect communities (cluster entities together).
//! 4. Use [`PathFinder`] to answer a "how is X related to Y?" query.
//! 5. Rank the paths and print the top-k results.
//!
//! No network access, no external files, no LLM required.

use oxirs_graphrag::community_detector::{CommunityDetector, CommunityGraph};
use oxirs_graphrag::path_finder::{KnowledgeEdge, PathFinder, PathFinderConfig};
use oxirs_graphrag::triple_extractor::{ExtractionConfig, TripleExtractor};

fn main() {
    // ── 1. Text corpus ────────────────────────────────────────────────────────
    let sentences = [
        "Alice is a data scientist.",
        "Alice works at ACME.",
        "Bob works at ACME.",
        "Carol is part of the AI team.",
        "Dave is a software engineer.",
        "Dave works at ACME.",
        "ACME has a research division.",
        "The research division is located in Berlin.",
    ];

    let extractor = TripleExtractor::with_defaults(ExtractionConfig::default());
    let mut extracted: Vec<(String, String, String)> = Vec::new();

    println!("=== Step 1: Extracted triples from corpus ===");
    for sentence in &sentences {
        for triple in extractor.extract(sentence) {
            println!(
                "  ({}, {}, {}) [conf={:.2}]",
                triple.subject, triple.predicate, triple.object, triple.confidence
            );
            extracted.push((triple.subject, triple.predicate, triple.object));
        }
    }
    println!("Total triples extracted: {}\n", extracted.len());

    // ── 2. Build community graph ──────────────────────────────────────────────
    // Use a fixed 8-node synthetic graph for reproducibility:
    let mut cg = CommunityGraph::new();
    for (id, label) in [
        (1u64, "Alice"),
        (2, "Bob"),
        (3, "Carol"),
        (4, "Dave"),
        (5, "ACME"),
        (6, "AI-Team"),
        (7, "Research"),
        (8, "Berlin"),
    ] {
        cg.add_node(id, label);
    }
    // Edges represent "same organisation" or direct relationships
    for (a, b, w) in [
        (1u64, 5u64, 1.5), // Alice – ACME
        (2, 5, 1.5),       // Bob – ACME
        (4, 5, 1.0),       // Dave – ACME
        (3, 6, 1.0),       // Carol – AI-Team
        (4, 6, 0.5),       // Dave – AI-Team
        (5, 7, 1.0),       // ACME – Research
        (7, 8, 1.0),       // Research – Berlin
        (1, 2, 0.5),       // Alice – Bob (colleagues)
    ] {
        cg.add_edge(a, b, w);
    }

    // ── 3. Community detection ────────────────────────────────────────────────
    let detector = CommunityDetector::new(2, 50);
    let result = detector.detect(&mut cg);

    println!("=== Step 2: Community detection ===");
    println!(
        "Modularity: {:.4}  |  Iterations: {}",
        result.modularity, result.iterations
    );
    for community in &result.communities {
        let members: Vec<String> = community
            .members
            .iter()
            .filter_map(|id| cg.nodes.get(id).map(|n| n.label.clone()))
            .collect();
        println!(
            "  Community {:>2}: {} member(s) — [{}]",
            community.id,
            community.size(),
            members.join(", ")
        );
    }
    println!();

    // ── 4. Path-based QA ──────────────────────────────────────────────────────
    // "How is Alice connected to Berlin?"
    let edges = vec![
        KnowledgeEdge::new("Alice", "works_at", "ACME"),
        KnowledgeEdge::new("Bob", "works_at", "ACME"),
        KnowledgeEdge::new("Carol", "member_of", "AI-Team"),
        KnowledgeEdge::new("Dave", "works_at", "ACME"),
        KnowledgeEdge::new("ACME", "has", "Research"),
        KnowledgeEdge::new("Research", "located_in", "Berlin"),
        KnowledgeEdge::new("AI-Team", "part_of", "ACME"),
    ];

    let config = PathFinderConfig {
        max_depth: 4,
        max_paths: 10,
        ..Default::default()
    };
    let finder = PathFinder::new(edges, config);

    let source = "Alice";
    let target = "Berlin";
    let paths = finder.bfs_paths(source, target, 4);

    println!(
        "=== Step 3: Top-{} paths: {} → {} ===",
        paths.len(),
        source,
        target
    );
    if paths.is_empty() {
        println!("  (no path found within 4 hops)");
    } else {
        for (rank, path) in paths.iter().enumerate() {
            println!(
                "  #{}: {} [score={:.2}]",
                rank + 1,
                path.narrative(),
                path.score
            );
        }
    }

    // ── 5. Stats ──────────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    println!("  Triples extracted: {}", extracted.len());
    println!("  Communities found: {}", result.communities.len());
    println!("  Paths {source}→{target}: {}", paths.len());
    println!("\nKGQA basic example completed successfully.");
}
