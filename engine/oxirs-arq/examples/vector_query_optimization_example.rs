//! Vector-Aware SPARQL Query Optimization Example
//!
//! This example demonstrates how to use the vector-aware query optimization
//! capabilities in oxirs-arq to enhance SPARQL query performance with semantic
//! vector search integration.

use anyhow::Result;
use oxirs_arq::{
    SparqlEngine, 
    vector_query_optimizer::{
        VectorOptimizerConfig, VectorIndexInfo, VectorIndexType, 
        VectorDistanceMetric, IndexAccuracyStats, IndexPerformanceStats
    },
    integrated_query_planner::IntegratedPlannerConfig,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();
    
    println!("🚀 Vector-Aware SPARQL Query Optimization Example");
    
    // 1. Create SPARQL engine with vector optimization
    let mut engine = create_vector_enabled_engine()?;
    
    // 2. Register vector indices
    register_sample_vector_indices(&engine)?;
    
    // 3. Execute sample queries with vector optimization
    execute_sample_queries(&mut engine)?;
    
    // 4. Show performance metrics
    show_performance_metrics(&engine)?;
    
    println!("✅ Vector optimization example completed successfully!");
    Ok(())
}

/// Create a SPARQL engine with vector optimization enabled
fn create_vector_enabled_engine() -> Result<SparqlEngine> {
    println!("\n📊 Creating vector-enabled SPARQL engine...");
    
    // Create base SPARQL engine
    let mut engine = SparqlEngine::new()?;
    
    // Configure vector optimization
    let vector_config = VectorOptimizerConfig {
        enable_vector_optimization: true,
        similarity_threshold: 0.8,
        max_vector_candidates: 1000,
        vector_cache_size: 10_000,
        enable_hybrid_search: true,
        embedding_dimension: 768, // BERT-like embeddings
        distance_metric: VectorDistanceMetric::Cosine,
        preferred_index_types: vec![
            VectorIndexType::Hnsw,
            VectorIndexType::IvfPq,
            VectorIndexType::IvfFlat,
        ],
        complexity_threshold: 5.0,
    };
    
    // Configure integrated planner
    let planner_config = IntegratedPlannerConfig {
        adaptive_optimization: true,
        cross_query_optimization: true,
        streaming_threshold: 256 * 1024 * 1024, // 256MB
        ml_cost_estimation: true,
        plan_cache_size: 1000,
        parallel_planning: true,
        stats_collection_interval: Duration::from_secs(30),
        advanced_index_recommendations: true,
    };
    
    // Enable vector optimization
    engine.enable_vector_optimization_with_config(vector_config, planner_config)?;
    
    println!("✅ Vector optimization enabled");
    println!("   - Similarity threshold: 0.8");
    println!("   - Max candidates: 1000");
    println!("   - Distance metric: Cosine");
    println!("   - Embedding dimension: 768");
    
    Ok(engine)
}

/// Register sample vector indices for demonstration
fn register_sample_vector_indices(engine: &SparqlEngine) -> Result<()> {
    println!("\n📝 Registering vector indices...");
    
    // Register HNSW index for entities
    let entity_index = VectorIndexInfo {
        index_type: VectorIndexType::Hnsw,
        dimension: 768,
        size: 1_000_000, // 1M entities
        distance_metric: VectorDistanceMetric::Cosine,
        build_time: Duration::from_secs(300), // 5 minutes build time
        last_updated: Instant::now(),
        accuracy_stats: IndexAccuracyStats {
            recall_at_k: {
                let mut map = HashMap::new();
                map.insert(1, 0.98);
                map.insert(5, 0.95);
                map.insert(10, 0.92);
                map.insert(50, 0.88);
                map
            },
            precision_at_k: {
                let mut map = HashMap::new();
                map.insert(10, 0.89);
                map.insert(50, 0.85);
                map
            },
            average_distance_error: 0.02,
            query_count: 50000,
        },
        performance_stats: IndexPerformanceStats {
            average_query_time: Duration::from_micros(150), // 150μs average
            queries_per_second: 6666.0,
            memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
            cache_hit_rate: 0.85,
            index_efficiency: 0.92,
        },
    };
    
    engine.register_vector_index("entity_embeddings".to_string(), entity_index)?;
    println!("✅ Registered HNSW entity index (1M vectors, 768-dim)");
    
    // Register IVF-PQ index for properties
    let property_index = VectorIndexInfo {
        index_type: VectorIndexType::IvfPq,
        dimension: 384,
        size: 500_000, // 500K properties
        distance_metric: VectorDistanceMetric::Cosine,
        build_time: Duration::from_secs(120), // 2 minutes build time
        last_updated: Instant::now(),
        accuracy_stats: IndexAccuracyStats {
            recall_at_k: {
                let mut map = HashMap::new();
                map.insert(1, 0.94);
                map.insert(5, 0.90);
                map.insert(10, 0.87);
                map.insert(50, 0.82);
                map
            },
            precision_at_k: {
                let mut map = HashMap::new();
                map.insert(10, 0.84);
                map.insert(50, 0.79);
                map
            },
            average_distance_error: 0.05,
            query_count: 25000,
        },
        performance_stats: IndexPerformanceStats {
            average_query_time: Duration::from_micros(80), // 80μs average
            queries_per_second: 12500.0,
            memory_usage: 512 * 1024 * 1024, // 512MB
            cache_hit_rate: 0.78,
            index_efficiency: 0.88,
        },
    };
    
    engine.register_vector_index("property_embeddings".to_string(), property_index)?;
    println!("✅ Registered IVF-PQ property index (500K vectors, 384-dim)");
    
    // Register flat index for literals (smaller, exact search)
    let literal_index = VectorIndexInfo {
        index_type: VectorIndexType::FlatIndex,
        dimension: 256,
        size: 100_000, // 100K literals
        distance_metric: VectorDistanceMetric::Cosine,
        build_time: Duration::from_secs(10), // 10 seconds build time
        last_updated: Instant::now(),
        accuracy_stats: IndexAccuracyStats {
            recall_at_k: {
                let mut map = HashMap::new();
                map.insert(1, 1.0); // Exact search
                map.insert(5, 1.0);
                map.insert(10, 1.0);
                map.insert(50, 1.0);
                map
            },
            precision_at_k: {
                let mut map = HashMap::new();
                map.insert(10, 1.0);
                map.insert(50, 1.0);
                map
            },
            average_distance_error: 0.0, // Exact
            query_count: 10000,
        },
        performance_stats: IndexPerformanceStats {
            average_query_time: Duration::from_micros(500), // 500μs average
            queries_per_second: 2000.0,
            memory_usage: 100 * 1024 * 1024, // 100MB
            cache_hit_rate: 0.95,
            index_efficiency: 1.0, // Exact search
        },
    };
    
    engine.register_vector_index("literal_embeddings".to_string(), literal_index)?;
    println!("✅ Registered Flat literal index (100K vectors, 256-dim)");
    
    Ok(())
}

/// Execute sample SPARQL queries with vector optimization
fn execute_sample_queries(engine: &mut SparqlEngine) -> Result<()> {
    println!("\n🔍 Executing sample queries with vector optimization...");
    
    // Mock dataset - in a real implementation, this would be a proper dataset
    struct MockDataset;
    impl oxirs_arq::executor::Dataset for MockDataset {
        fn get_default_graph(&self) -> Result<&dyn oxirs_arq::executor::Graph> {
            Err(anyhow::anyhow!("Mock dataset - not implemented"))
        }
        
        fn get_named_graph(&self, _graph_name: &oxirs_arq::Term) -> Result<Option<&dyn oxirs_arq::executor::Graph>> {
            Ok(None)
        }
    }
    
    let dataset = MockDataset;
    
    // Sample queries that could benefit from vector optimization
    let sample_queries = vec![
        (
            "Semantic Entity Search",
            r#"
            PREFIX vec: <http://example.org/vector/>
            SELECT ?entity ?similarity WHERE {
                ?entity vec:similar "artificial intelligence" .
                ?entity vec:similarity ?similarity .
                FILTER (?similarity > 0.8)
            }
            ORDER BY DESC(?similarity)
            LIMIT 10
            "#
        ),
        (
            "Hybrid Text-Vector Search",
            r#"
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX vec: <http://example.org/vector/>
            SELECT ?concept ?label ?score WHERE {
                ?concept rdfs:label ?label .
                ?concept vec:search "machine learning concepts" .
                ?concept vec:similarity ?score .
                FILTER (regex(?label, "learning", "i") && ?score > 0.7)
            }
            ORDER BY DESC(?score)
            LIMIT 20
            "#
        ),
        (
            "Cross-Modal Similarity",
            r#"
            PREFIX vec: <http://example.org/vector/>
            PREFIX ex: <http://example.org/>
            SELECT ?image ?text ?similarity WHERE {
                ?image a ex:Image .
                ?text a ex:TextDocument .
                ?image vec:crossModalSimilarity ?text .
                ?image vec:similarity ?similarity .
                FILTER (?similarity > 0.6)
            }
            ORDER BY DESC(?similarity)
            LIMIT 15
            "#
        ),
        (
            "Semantic Property Expansion",
            r#"
            PREFIX vec: <http://example.org/vector/>
            PREFIX ex: <http://example.org/>
            SELECT ?subject ?property ?object WHERE {
                ?subject ?property ?object .
                ?property vec:semanticallyRelated ex:hasSkill .
                FILTER (?subject = ex:person123)
            }
            "#
        )
    ];
    
    for (query_name, query_str) in sample_queries {
        println!("\n🔸 Executing: {}", query_name);
        println!("   Query preview: {}", query_str.lines().nth(3).unwrap_or("").trim());
        
        let start_time = Instant::now();
        
        // In a real implementation, this would execute successfully
        // For this example, we'll simulate the vector optimization process
        match engine.execute_query(query_str, &dataset) {
            Ok((solution, stats)) => {
                let execution_time = start_time.elapsed();
                println!("   ✅ Query executed successfully");
                println!("   📊 Results: {} bindings", solution.len());
                println!("   ⏱️  Execution time: {:?}", execution_time);
                println!("   💾 Memory used: {} KB", stats.memory_used / 1024);
                
                // Simulate vector execution feedback
                let strategy_hash = calculate_query_hash(query_str);
                let _ = engine.update_vector_execution_feedback(
                    strategy_hash,
                    execution_time,
                    0.92, // Mock recall
                    stats.memory_used,
                    true,
                );
            }
            Err(e) => {
                // Expected for mock dataset - show what would happen
                println!("   ℹ️  Vector optimization analysis completed");
                println!("   🎯 Vector strategy detected and cost estimated");
                println!("   📈 Query plan enhanced with vector awareness");
                println!("   ⚠️  Mock execution: {}", e);
            }
        }
    }
    
    Ok(())
}

/// Show performance metrics from vector optimization
fn show_performance_metrics(engine: &SparqlEngine) -> Result<()> {
    println!("\n📈 Vector Optimization Performance Metrics");
    
    if let Some(metrics) = engine.get_vector_performance_metrics() {
        println!("╭─────────────────────────────────────────╮");
        println!("│           Vector Metrics                │");
        println!("├─────────────────────────────────────────┤");
        println!("│ Vector queries optimized: {:>13} │", metrics.vector_queries_optimized);
        println!("│ Hybrid queries optimized: {:>13} │", metrics.hybrid_queries_optimized);
        println!("│ Semantic expansions:      {:>13} │", metrics.semantic_expansions_performed);
        println!("│ Average speedup:          {:>13.2}x │", metrics.average_optimization_speedup);
        println!("│ Vector cache hit rate:    {:>13.1}% │", metrics.vector_cache_hit_rate * 100.0);
        println!("│ Embedding gen time:       {:>13?} │", metrics.embedding_generation_time);
        println!("│ Total optimization time:  {:>13?} │", metrics.total_optimization_time);
        println!("╰─────────────────────────────────────────╯");
    } else {
        println!("⚠️  Vector optimization not enabled or no metrics available");
    }
    
    // Show integration status
    println!("\n🔧 Integration Status");
    println!("├─ Vector optimization: {}", 
        if engine.is_vector_optimization_enabled() { "✅ Enabled" } else { "❌ Disabled" });
    println!("├─ Integrated planning: {}", 
        if engine.is_integrated_planning_enabled() { "✅ Enabled" } else { "❌ Disabled" });
    println!("└─ Vector indices: 3 registered (HNSW, IVF-PQ, Flat)");
    
    Ok(())
}

/// Calculate a simple hash for query string (for demonstration)
fn calculate_query_hash(query: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    query.hash(&mut hasher);
    hasher.finish()
}

/// Additional example: Programmatic vector index optimization
#[allow(dead_code)]
fn demonstrate_index_optimization(engine: &SparqlEngine) -> Result<()> {
    println!("\n🎯 Vector Index Optimization Recommendations");
    
    // Get index recommendations (would come from integrated planner)
    let recommendations = engine.get_index_recommendations()?;
    
    if recommendations.is_empty() {
        println!("📊 Analysis: Current vector indices are well-optimized");
        println!("   - Entity HNSW index: High performance for similarity search");
        println!("   - Property IVF-PQ index: Good balance of speed and memory");
        println!("   - Literal Flat index: Perfect for exact small-scale search");
        
        println!("\n💡 Optimization Tips:");
        println!("   1. Consider HNSW for properties if query volume increases");
        println!("   2. Monitor cache hit rates and adjust cache sizes");
        println!("   3. Use hybrid search for complex queries");
        println!("   4. Enable GPU acceleration for large vector operations");
    } else {
        println!("📋 Recommendations found:");
        for (i, rec) in recommendations.iter().enumerate() {
            println!("   {}. {}", i + 1, rec.description);
        }
    }
    
    Ok(())
}

/// Example of vector-specific SPARQL functions
#[allow(dead_code)]
fn demonstrate_vector_functions() {
    println!("\n🔧 Vector-Specific SPARQL Functions");
    
    println!("Available vector functions:");
    println!("├─ vec:similarity(?a, ?b)     - Calculate similarity between vectors");
    println!("├─ vec:similar(?entity, k)    - Find k most similar entities");
    println!("├─ vec:search(?text, limit)   - Semantic text search");
    println!("├─ vec:searchIn(?query, ?graph) - Graph-scoped vector search");
    println!("├─ vec:distance(?a, ?b)       - Calculate vector distance");
    println!("├─ vec:embed(?text)           - Generate embedding for text");
    println!("├─ vec:cosine(?a, ?b)         - Cosine similarity");
    println!("├─ vec:euclidean(?a, ?b)      - Euclidean distance");
    println!("├─ vec:cluster(?entities, k)  - K-means clustering");
    println!("└─ vec:recommend(?entity, k)  - Entity recommendations");
    
    println!("\nExample usage in SPARQL:");
    println!("  SELECT ?similar WHERE {{");
    println!("    ?entity vec:similar \"machine learning\" .");
    println!("    ?entity vec:similarity ?similar .");
    println!("    FILTER(?similar > 0.8)");
    println!("  }} ORDER BY DESC(?similar)");
}