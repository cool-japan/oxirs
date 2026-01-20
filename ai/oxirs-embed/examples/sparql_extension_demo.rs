//! # SPARQL Extension Demonstration
//!
//! This example demonstrates the advanced SPARQL extension capabilities of oxirs-embed,
//! including vector similarity operators, semantic query expansion, and fuzzy matching.
//!
//! ## Features Demonstrated
//!
//! 1. **Vector Similarity**: Compute similarity between entities
//! 2. **Nearest Neighbors**: Find k-nearest neighbors using embeddings
//! 3. **Similar Entities**: Find all entities above a similarity threshold
//! 4. **Semantic Query Expansion**: Automatically expand queries with similar concepts
//! 5. **Fuzzy Matching**: Match entities even with typos
//! 6. **Query Statistics**: Monitor query performance
//!
//! ## Running this example
//!
//! ```bash
//! cargo run --example sparql_extension_demo --features basic-models
//! ```

use anyhow::Result;
use oxirs_embed::{
    EmbeddingModel, ModelConfig, NamedNode, SparqlExtension, SparqlExtensionConfig, TransE, Triple,
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("=== SPARQL Extension Demonstration ===\n");

    // Create and train a knowledge graph embedding model
    let model = create_knowledge_graph().await?;

    // Create SPARQL extension with custom configuration
    let config = SparqlExtensionConfig {
        default_similarity_threshold: 0.6,
        max_expansions_per_element: 5,
        enable_query_rewriting: true,
        enable_semantic_caching: true,
        semantic_cache_size: 100,
        enable_fuzzy_matching: true,
        min_expansion_confidence: 0.5,
        enable_parallel_processing: true,
    };

    let sparql_ext = SparqlExtension::with_config(Box::new(model), config);

    info!("\n1. Vector Similarity Computation");
    info!("===================================");
    demo_vector_similarity(&sparql_ext).await?;

    info!("\n2. Nearest Neighbor Search");
    info!("===================================");
    demo_nearest_neighbors(&sparql_ext).await?;

    info!("\n3. Similar Entities Discovery");
    info!("===================================");
    demo_similar_entities(&sparql_ext).await?;

    info!("\n4. Similar Relations Discovery");
    info!("===================================");
    demo_similar_relations(&sparql_ext).await?;

    info!("\n5. Semantic Query Expansion");
    info!("===================================");
    demo_semantic_expansion(&sparql_ext).await?;

    info!("\n6. Fuzzy Entity Matching");
    info!("===================================");
    demo_fuzzy_matching(&sparql_ext).await?;

    info!("\n7. Query Performance Statistics");
    info!("===================================");
    demo_query_statistics(&sparql_ext).await?;

    info!("\n8. Practical SPARQL Query Examples");
    info!("===================================");
    demo_practical_queries(&sparql_ext).await?;

    info!("\n=== Demonstration Complete ===");

    Ok(())
}

/// Create and train a sample knowledge graph
async fn create_knowledge_graph() -> Result<TransE> {
    info!("Creating knowledge graph with biomedical and research entities...");

    let config = ModelConfig::default()
        .with_dimensions(128)
        .with_learning_rate(0.01)
        .with_max_epochs(100)
        .with_batch_size(32);

    let mut model = TransE::new(config);

    // Add biomedical knowledge
    let biomedical_triples = vec![
        // Diseases and symptoms
        ("alzheimers_disease", "has_symptom", "memory_loss"),
        ("alzheimers_disease", "has_symptom", "cognitive_decline"),
        ("parkinsons_disease", "has_symptom", "tremor"),
        ("parkinsons_disease", "has_symptom", "motor_impairment"),
        ("dementia", "has_symptom", "memory_loss"),
        ("dementia", "has_symptom", "confusion"),
        // Genes and diseases
        ("APOE4_gene", "associated_with", "alzheimers_disease"),
        ("LRRK2_gene", "associated_with", "parkinsons_disease"),
        ("APP_gene", "associated_with", "alzheimers_disease"),
        // Proteins and functions
        ("amyloid_beta", "accumulates_in", "alzheimers_disease"),
        ("alpha_synuclein", "accumulates_in", "parkinsons_disease"),
        ("tau_protein", "forms_tangles_in", "alzheimers_disease"),
        // Drugs and targets
        ("donepezil", "treats", "alzheimers_disease"),
        ("levodopa", "treats", "parkinsons_disease"),
        ("memantine", "treats", "alzheimers_disease"),
        ("donepezil", "inhibits", "acetylcholinesterase"),
        ("levodopa", "converts_to", "dopamine"),
        // Research relationships
        ("alzheimers_research", "studies", "alzheimers_disease"),
        ("parkinsons_research", "studies", "parkinsons_disease"),
        ("neurology_dept", "conducts", "alzheimers_research"),
        ("neurology_dept", "conducts", "parkinsons_research"),
    ];

    // Add research network knowledge
    let research_triples = [
        ("dr_smith", "researches", "alzheimers_disease"),
        ("dr_jones", "researches", "parkinsons_disease"),
        ("dr_smith", "collaborates_with", "dr_jones"),
        ("dr_smith", "affiliated_with", "neurology_dept"),
        ("dr_jones", "affiliated_with", "neurology_dept"),
        ("dr_wang", "researches", "dementia"),
        ("dr_wang", "collaborates_with", "dr_smith"),
    ];

    // Add all triples
    for (s, p, o) in biomedical_triples.iter().chain(research_triples.iter()) {
        let triple = Triple::new(
            NamedNode::new(&format!("http://bio.example.org/{s}"))?,
            NamedNode::new(&format!("http://bio.example.org/{p}"))?,
            NamedNode::new(&format!("http://bio.example.org/{o}"))?,
        );
        model.add_triple(triple)?;
    }

    info!("Training knowledge graph embeddings...");
    let stats = model.train(Some(100)).await?;
    info!(
        "Training completed: {} epochs, final loss: {:.6}",
        stats.epochs_completed, stats.final_loss
    );

    Ok(model)
}

/// Demonstrate vector similarity computation
async fn demo_vector_similarity(sparql_ext: &SparqlExtension) -> Result<()> {
    let entity_pairs = vec![
        (
            "http://bio.example.org/alzheimers_disease",
            "http://bio.example.org/dementia",
        ),
        (
            "http://bio.example.org/alzheimers_disease",
            "http://bio.example.org/parkinsons_disease",
        ),
        (
            "http://bio.example.org/dr_smith",
            "http://bio.example.org/dr_jones",
        ),
    ];

    for (e1, e2) in entity_pairs {
        let similarity = sparql_ext.vec_similarity(e1, e2).await?;
        info!(
            "Similarity between '{}' and '{}': {:.4}",
            extract_name(e1),
            extract_name(e2),
            similarity
        );
    }

    Ok(())
}

/// Demonstrate nearest neighbor search
async fn demo_nearest_neighbors(sparql_ext: &SparqlExtension) -> Result<()> {
    let target_entity = "http://bio.example.org/alzheimers_disease";
    let k = 5;

    info!(
        "Finding {} nearest neighbors for '{}':",
        k,
        extract_name(target_entity)
    );

    let neighbors = sparql_ext.vec_nearest(target_entity, k, Some(0.3)).await?;

    for (i, (entity, similarity)) in neighbors.iter().enumerate() {
        info!(
            "  {}. {} (similarity: {:.4})",
            i + 1,
            extract_name(entity),
            similarity
        );
    }

    Ok(())
}

/// Demonstrate similar entities discovery
async fn demo_similar_entities(sparql_ext: &SparqlExtension) -> Result<()> {
    let target_entity = "http://bio.example.org/memory_loss";
    let threshold = 0.4;

    info!(
        "Finding entities similar to '{}' (threshold: {}):",
        extract_name(target_entity),
        threshold
    );

    let similar = sparql_ext
        .vec_similar_entities(target_entity, threshold)
        .await?;

    if similar.is_empty() {
        info!("  No similar entities found above threshold");
    } else {
        for (entity, similarity) in similar.iter().take(5) {
            info!(
                "  - {} (similarity: {:.4})",
                extract_name(entity),
                similarity
            );
        }
    }

    Ok(())
}

/// Demonstrate similar relations discovery
async fn demo_similar_relations(sparql_ext: &SparqlExtension) -> Result<()> {
    let target_relation = "http://bio.example.org/treats";
    let threshold = 0.3;

    info!(
        "Finding relations similar to '{}' (threshold: {}):",
        extract_name(target_relation),
        threshold
    );

    let similar = sparql_ext
        .vec_similar_relations(target_relation, threshold)
        .await?;

    if similar.is_empty() {
        info!("  No similar relations found above threshold");
    } else {
        for (relation, similarity) in similar.iter().take(5) {
            info!(
                "  - {} (similarity: {:.4})",
                extract_name(relation),
                similarity
            );
        }
    }

    Ok(())
}

/// Demonstrate semantic query expansion
async fn demo_semantic_expansion(sparql_ext: &SparqlExtension) -> Result<()> {
    let original_query = r#"
        PREFIX bio: <http://bio.example.org/>
        SELECT ?disease ?symptom WHERE {
            ?disease bio:has_symptom ?symptom .
            FILTER(?disease = bio:alzheimers_disease)
        }
    "#;

    info!("Original SPARQL query:");
    info!("{}", original_query);

    let expanded = sparql_ext.expand_query_semantically(original_query).await?;

    info!("\nExpansion statistics:");
    info!("  Total expansions: {}", expanded.expansion_count);
    info!("  Entity expansions: {}", expanded.entity_expansions.len());
    info!(
        "  Relation expansions: {}",
        expanded.relation_expansions.len()
    );

    if !expanded.entity_expansions.is_empty() {
        info!("\nEntity expansions:");
        for (original, expansions) in expanded.entity_expansions.iter().take(3) {
            info!("  '{}' can expand to:", extract_name(original));
            for exp in expansions.iter().take(3) {
                info!(
                    "    - {} (confidence: {:.2})",
                    extract_name(&exp.expanded),
                    exp.confidence
                );
            }
        }
    }

    Ok(())
}

/// Demonstrate fuzzy entity matching
async fn demo_fuzzy_matching(sparql_ext: &SparqlExtension) -> Result<()> {
    let fuzzy_queries = vec![
        ("alzhemers", "alzheimers_disease (typo)"),
        ("memoryloss", "memory_loss (no underscore)"),
        ("parkinsons", "parkinsons_disease (partial)"),
    ];

    for (query, description) in fuzzy_queries {
        info!("Fuzzy matching '{}':", description);

        let matches = sparql_ext.fuzzy_match_entity(query, 3).await?;

        if matches.is_empty() {
            info!("  No fuzzy matches found");
        } else {
            for (entity, score) in matches {
                info!("  - {} (score: {:.4})", extract_name(&entity), score);
            }
        }
    }

    Ok(())
}

/// Demonstrate query statistics
async fn demo_query_statistics(sparql_ext: &SparqlExtension) -> Result<()> {
    let stats = sparql_ext.get_statistics().await;

    info!("Query Performance Statistics:");
    info!(
        "  Similarity computations: {}",
        stats.similarity_computations
    );
    info!(
        "  Nearest neighbor queries: {}",
        stats.nearest_neighbor_queries
    );
    info!("  Query expansions: {}", stats.query_expansions);
    info!("  Fuzzy matches: {}", stats.fuzzy_matches);
    info!("  Cache hits: {}", stats.cache_hits);
    info!("  Cache misses: {}", stats.cache_misses);

    if stats.similarity_computations > 0 {
        let cache_hit_rate = if stats.cache_hits + stats.cache_misses > 0 {
            (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64) * 100.0
        } else {
            0.0
        };
        info!("  Cache hit rate: {:.2}%", cache_hit_rate);
    }

    Ok(())
}

/// Demonstrate practical SPARQL query patterns
async fn demo_practical_queries(_sparql_ext: &SparqlExtension) -> Result<()> {
    info!("\nExample 1: Finding similar diseases");
    info!("-----------------------------------");
    let query1 = r#"
        PREFIX bio: <http://bio.example.org/>
        PREFIX vec: <http://oxirs.ai/vec#>

        SELECT ?disease ?similarity WHERE {
            ?disease vec:similarTo bio:alzheimers_disease .
            BIND(vec:similarity(bio:alzheimers_disease, ?disease) AS ?similarity)
            FILTER(?similarity > 0.5)
        }
        ORDER BY DESC(?similarity)
    "#;
    info!("{}", query1);

    info!("\nExample 2: Finding nearest research collaborators");
    info!("---------------------------------------------------");
    let query2 = r#"
        PREFIX bio: <http://bio.example.org/>
        PREFIX vec: <http://oxirs.ai/vec#>

        SELECT ?researcher ?distance WHERE {
            ?researcher vec:nearestTo bio:dr_smith .
            BIND(vec:distance(bio:dr_smith, ?researcher) AS ?distance)
        }
        LIMIT 5
    "#;
    info!("{}", query2);

    info!("\nExample 3: Semantic search for treatments");
    info!("-------------------------------------------");
    let query3 = r#"
        PREFIX bio: <http://bio.example.org/>
        PREFIX vec: <http://oxirs.ai/vec#>

        SELECT ?drug ?disease WHERE {
            ?drug ?relation ?disease .
            FILTER(vec:semanticMatch(?relation, bio:treats, 0.7))
        }
    "#;
    info!("{}", query3);

    Ok(())
}

/// Extract entity name from full URI
fn extract_name(uri: &str) -> &str {
    uri.rsplit('/').next().unwrap_or(uri)
}
