//! # Hybrid Retrieval Example
//!
//! Demonstrates combining **graph traversal** with **structural node embeddings**
//! and **community detection** for hybrid retrieval — the core innovation of GraphRAG.
//!
//! Pipeline:
//! 1. Build a synthetic 10-node knowledge graph.
//! 2. Compute structural node embeddings via [`GraphEmbedder`] (Node2Vec-style).
//! 3. Detect communities via [`CommunityDetector`].
//! 4. Use [`PathFinder`] to find traversal paths (graph-side retrieval).
//! 5. Combine: score each path by the average embedding similarity of its nodes.
//! 6. Print ranked results (hybrid score = path_score × embedding_cohesion).
//!
//! All data is synthetic; no network or files required.

use oxirs_graphrag::community_detector::{CommunityDetector, CommunityGraph};
use oxirs_graphrag::graph_embedder::{Graph, GraphEmbedder, WalkConfig};
use oxirs_graphrag::path_finder::{KnowledgeEdge, PathFinder, PathFinderConfig};

// Node labels for the synthetic graph (index = node id in graph_embedder::Graph)
const NODES: &[&str] = &[
    "Alice",    // 0
    "Bob",      // 1
    "Carol",    // 2
    "ACME",     // 3
    "AI-Lab",   // 4
    "TechCorp", // 5
    "Berlin",   // 6
    "Paris",    // 7
    "Research", // 8
    "DevTeam",  // 9
];

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

fn main() {
    // ── 1. Build structural graph for embeddings ──────────────────────────────
    let mut emb_graph = Graph::new(NODES.len());
    // Edges (undirected structural, indexed by NODES array positions)
    let structural_edges: &[(usize, usize)] = &[
        (0, 3), // Alice – ACME
        (1, 3), // Bob – ACME
        (2, 4), // Carol – AI-Lab
        (3, 5), // ACME – TechCorp
        (3, 8), // ACME – Research
        (4, 5), // AI-Lab – TechCorp
        (5, 6), // TechCorp – Berlin
        (5, 7), // TechCorp – Paris
        (8, 9), // Research – DevTeam
        (0, 1), // Alice – Bob (colleagues)
        (1, 2), // Bob – Carol
    ];
    for &(a, b) in structural_edges {
        emb_graph.add_edge(a, b, 1.0);
        emb_graph.add_edge(b, a, 1.0); // undirected
    }

    // ── 2. Compute node embeddings ────────────────────────────────────────────
    let walk_config = WalkConfig {
        walk_length: 8,
        walks_per_node: 10,
        return_param_p: 1.0,
        in_out_param_q: 1.0,
    };
    // structural_embedding returns Vec<NodeEmbedding> directly
    let node_embeddings = GraphEmbedder::structural_embedding(&emb_graph, 8);

    println!("=== Step 1: Structural node embeddings (dim=8) ===");
    println!("  Nodes embedded: {}", node_embeddings.len());

    // Build a lookup: node_index → embedding vector
    let mut node_emb: Vec<Option<Vec<f32>>> = vec![None; NODES.len()];
    for emb in &node_embeddings {
        if emb.node_id < NODES.len() {
            node_emb[emb.node_id] = Some(emb.vector.clone());
        }
    }

    // Also generate random walks for display
    let walks = GraphEmbedder::random_walks(&emb_graph, &walk_config);
    println!("  Random walks generated: {}", walks.len());

    // ── 3. Community detection ────────────────────────────────────────────────
    let mut cg = CommunityGraph::new();
    for (id, label) in NODES.iter().enumerate() {
        cg.add_node(id as u64, label);
    }
    for &(a, b) in structural_edges {
        cg.add_edge(a as u64, b as u64, 1.0);
    }
    let detector = CommunityDetector::new(2, 50);
    let detection = detector.detect(&mut cg);

    println!("\n=== Step 2: Community detection ===");
    println!(
        "  Communities: {}  |  Modularity: {:.4}",
        detection.communities.len(),
        detection.modularity
    );
    for community in &detection.communities {
        let labels: Vec<&str> = community
            .members
            .iter()
            .filter_map(|&id| NODES.get(id as usize).copied())
            .collect();
        println!("  Community {:>2}: [{}]", community.id, labels.join(", "));
    }

    // ── 4. Graph traversal (path retrieval) ──────────────────────────────────
    let kg_edges = vec![
        KnowledgeEdge::new("Alice", "works_at", "ACME"),
        KnowledgeEdge::new("Bob", "works_at", "ACME"),
        KnowledgeEdge::new("Carol", "member_of", "AI-Lab"),
        KnowledgeEdge::new("ACME", "partner_of", "TechCorp"),
        KnowledgeEdge::new("ACME", "has_division", "Research"),
        KnowledgeEdge::new("AI-Lab", "collaborates_with", "TechCorp"),
        KnowledgeEdge::new("TechCorp", "office_in", "Berlin"),
        KnowledgeEdge::new("TechCorp", "office_in", "Paris"),
        KnowledgeEdge::new("Research", "team", "DevTeam"),
        KnowledgeEdge::new("Bob", "knows", "Carol"),
    ];

    let path_config = PathFinderConfig {
        max_depth: 4,
        max_paths: 10,
        ..Default::default()
    };
    let finder = PathFinder::new(kg_edges, path_config);

    let query_source = "Alice";
    let query_target = "Berlin";
    let paths = finder.bfs_paths(query_source, query_target, 4);

    println!("\n=== Step 3: Graph paths [{query_source}] → [{query_target}] ===");

    // ── 5. Hybrid scoring: path score × embedding cohesion ────────────────────
    // For each path, compute average cosine similarity of consecutive node embeddings.
    let node_to_idx: std::collections::HashMap<&str, usize> =
        NODES.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let mut scored_paths: Vec<(f32, String)> = paths
        .iter()
        .map(|path| {
            let narrative = path.narrative();
            // Compute embedding cohesion: mean cosine sim of adjacent node pairs
            let cohesion = if path.nodes.len() < 2 {
                1.0f32
            } else {
                let sims: Vec<f32> = path
                    .nodes
                    .windows(2)
                    .filter_map(|pair| {
                        let a_idx = node_to_idx.get(pair[0].as_str())?;
                        let b_idx = node_to_idx.get(pair[1].as_str())?;
                        let a_emb = node_emb[*a_idx].as_deref()?;
                        let b_emb = node_emb[*b_idx].as_deref()?;
                        Some(cosine_similarity(a_emb, b_emb))
                    })
                    .collect();
                if sims.is_empty() {
                    0.5f32
                } else {
                    sims.iter().sum::<f32>() / sims.len() as f32
                }
            };

            // Hybrid score: graph path score normalised by hop count, boosted by embedding cohesion
            let graph_score = if path.hop_count > 0 {
                path.score as f32 / path.hop_count as f32
            } else {
                path.score as f32
            };
            let hybrid_score = (graph_score * 0.5 + cohesion * 0.5).clamp(0.0, 1.0);
            (hybrid_score, narrative)
        })
        .collect();

    scored_paths.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    if scored_paths.is_empty() {
        println!("  (no path found within 4 hops)");
    } else {
        println!("  Ranked by hybrid score (graph-traversal + embedding cohesion):");
        for (rank, (score, narrative)) in scored_paths.iter().enumerate() {
            println!("  #{}: [hybrid={:.3}] {}", rank + 1, score, narrative);
        }
    }

    println!("\n=== Summary ===");
    println!("  Nodes in graph:       {}", NODES.len());
    println!("  Communities detected: {}", detection.communities.len());
    println!("  Paths found:          {}", paths.len());
    println!("\nHybrid retrieval example completed successfully.");
}
