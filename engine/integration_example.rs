//! Integration Example
//!
//! This example demonstrates the enhanced OxiRS integration capabilities
//! with realistic SPARQL, reasoning, validation, and vector search functionality.

use anyhow::Result;
use std::time::Duration;

mod oxirs_integration;
use oxirs_integration::{
    OxirsOrchestrator, IntegrationConfig, IntegratedQuery,
    VectorQuery, HybridQuerySpec, HybridQueryType, IntegrationStrategy
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    println!("🚀 OxiRS Integration Example");
    println!("===========================");
    
    // Create enhanced integration configuration
    let config = IntegrationConfig {
        enable_reasoning: true,
        enable_validation: true,
        enable_vector_search: true,
        enable_neural_symbolic: true,
        max_query_time: Duration::from_secs(30),
        cache_size: 10000,
        parallel_execution: true,
    };
    
    // Create orchestrator
    let orchestrator = OxirsOrchestrator::new(config);
    
    // Example 1: SPARQL Query with Enhancements
    println!("\n📊 Example 1: Enhanced SPARQL Query");
    println!("-----------------------------------");
    
    let sparql_query = IntegratedQuery {
        sparql_query: Some(r#"
            SELECT ?person ?name ?type WHERE {
                ?person rdf:type ?type .
                ?person foaf:name ?name .
                FILTER(?type = foaf:Person)
            }
        "#.to_string()),
        reasoning_rules: None,
        validation_shapes: None,
        vector_query: None,
        hybrid_query: None,
    };
    
    let result = orchestrator.execute_integrated_query(sparql_query).await?;
    
    if let Some(sparql_results) = &result.sparql_results {
        println!("✅ SPARQL Results:");
        println!("   - Query Time: {:?}", sparql_results.query_time);
        println!("   - Bindings: {} results", sparql_results.bindings.len());
        println!("   - Optimizations: {:?}", sparql_results.optimizations_applied);
        
        for (i, binding) in sparql_results.bindings.iter().take(2).enumerate() {
            println!("   - Result {}: {:?}", i + 1, binding);
        }
    }
    
    // Example 2: Reasoning with Rule Types
    println!("\n🧠 Example 2: Enhanced Reasoning");
    println!("-------------------------------");
    
    let reasoning_query = IntegratedQuery {
        sparql_query: None,
        reasoning_rules: Some(vec![
            "rdfs:subClassOf Person Human".to_string(),
            "owl:transitiveProperty knows".to_string(),
            "custom:skillInference AI_Researcher hasSkill MachineLearning".to_string(),
        ]),
        validation_shapes: None,
        vector_query: None,
        hybrid_query: None,
    };
    
    let result = orchestrator.execute_integrated_query(reasoning_query).await?;
    
    if let Some(reasoning_results) = &result.reasoning_results {
        println!("✅ Reasoning Results:");
        println!("   - Inferred Facts: {}", reasoning_results.inferred_facts.len());
        println!("   - Rules Fired: {:?}", reasoning_results.rules_fired);
        println!("   - Average Confidence: {:.2}", 
                 reasoning_results.confidence_scores.iter().sum::<f32>() / reasoning_results.confidence_scores.len() as f32);
        
        for (i, fact) in reasoning_results.inferred_facts.iter().take(2).enumerate() {
            println!("   - Fact {}: {}", i + 1, fact);
        }
    }
    
    // Example 3: SHACL Validation with Complexity Analysis
    println!("\n🔍 Example 3: Enhanced SHACL Validation");
    println!("-------------------------------------");
    
    let validation_query = IntegratedQuery {
        sparql_query: None,
        reasoning_rules: None,
        validation_shapes: Some(vec![
            r#"
            ex:PersonShape a sh:NodeShape ;
                sh:targetClass ex:Person ;
                sh:property [
                    sh:path ex:name ;
                    sh:datatype xsd:string ;
                    sh:minCount 1 ;
                    sh:maxCount 1 ;
                ] ;
                sh:property [
                    sh:path ex:age ;
                    sh:datatype xsd:integer ;
                    sh:minInclusive 0 ;
                    sh:maxInclusive 150 ;
                ] .
            "#.to_string(),
            r#"
            ex:ComplexShape a sh:NodeShape ;
                sh:targetClass ex:Document ;
                sh:property [
                    sh:path ex:content ;
                    sh:datatype xsd:string ;
                    sh:pattern "^[A-Za-z0-9\\s]+$" ;
                    sh:minLength 10 ;
                    sh:maxLength 1000 ;
                ] ;
                sh:sparql [
                    sh:message "Custom SPARQL constraint failed" ;
                    sh:select "SELECT $this WHERE { $this ex:hasValidFormat true }" ;
                ] .
            "#.to_string(),
        ]),
        vector_query: None,
        hybrid_query: None,
    };
    
    let result = orchestrator.execute_integrated_query(validation_query).await?;
    
    if let Some(validation_results) = &result.validation_results {
        println!("✅ Validation Results:");
        println!("   - Conforms: {}", validation_results.conforms);
        println!("   - Validation Time: {:?}", validation_results.validation_time);
        println!("   - Shapes Evaluated: {}", validation_results.shapes_evaluated.len());
        println!("   - Violations: {}", validation_results.violations.len());
        
        for (i, violation) in validation_results.violations.iter().take(2).enumerate() {
            println!("   - Violation {}: {} ({})", i + 1, violation.message, violation.severity);
        }
    }
    
    // Example 4: Vector Search with Advanced Features
    println!("\n🔍 Example 4: Enhanced Vector Search");
    println!("----------------------------------");
    
    let vector_query = IntegratedQuery {
        sparql_query: None,
        reasoning_rules: None,
        validation_shapes: None,
        vector_query: Some(VectorQuery {
            text: "machine learning artificial intelligence algorithms".to_string(),
            similarity_threshold: 0.7,
            max_results: 5,
            embedding_strategy: "sentence-transformers".to_string(),
        }),
        hybrid_query: None,
    };
    
    let result = orchestrator.execute_integrated_query(vector_query).await?;
    
    if let Some(vector_results) = &result.vector_results {
        println!("✅ Vector Search Results:");
        println!("   - Search Time: {:?}", vector_results.search_time);
        println!("   - Matches Found: {}", vector_results.matches.len());
        println!("   - Total Candidates: {}", vector_results.total_candidates);
        println!("   - Embedding Strategy: {}", vector_results.embedding_strategy);
        
        for (i, match_result) in vector_results.matches.iter().take(2).enumerate() {
            println!("   - Match {}: {} (score: {:.3})", 
                     i + 1, match_result.resource, match_result.similarity_score);
            if let Some(snippet) = &match_result.content_snippet {
                println!("     Snippet: {}...", &snippet[..snippet.len().min(80)]);
            }
        }
    }
    
    // Example 5: Hybrid Neural-Symbolic Query
    println!("\n🧬 Example 5: Hybrid Neural-Symbolic Query");
    println!("----------------------------------------");
    
    let hybrid_query = IntegratedQuery {
        sparql_query: None,
        reasoning_rules: None,
        validation_shapes: None,
        vector_query: None,
        hybrid_query: Some(HybridQuerySpec {
            query_type: HybridQueryType::SemanticSparql,
            symbolic_component: "SELECT ?doc WHERE { ?doc rdf:type ex:AIDocument }".to_string(),
            vector_component: "artificial intelligence research papers".to_string(),
            integration_strategy: IntegrationStrategy::Pipeline,
        }),
    };
    
    let result = orchestrator.execute_integrated_query(hybrid_query).await?;
    
    if let Some(hybrid_results) = &result.hybrid_results {
        println!("✅ Hybrid Query Results:");
        println!("   - Combined Results: {}", hybrid_results.combined_results.len());
        println!("   - Confidence Score: {:.3}", hybrid_results.confidence_score);
        println!("   - Integration Strategy: {:?}", hybrid_results.integration_strategy);
        
        for (i, result_item) in hybrid_results.combined_results.iter().take(2).enumerate() {
            println!("   - Result {}: {}", i + 1, result_item);
        }
    }
    
    // Display performance metrics
    println!("\n📊 Performance Metrics");
    println!("=====================");
    
    let metrics = orchestrator.get_performance_metrics();
    println!("   - Total Operations: {}", metrics.total_operations);
    println!("   - Success Rate: {:.1}%", 
             (metrics.successful_operations as f32 / metrics.total_operations as f32) * 100.0);
    println!("   - Average Execution Time: {:?}", metrics.average_execution_time);
    
    for (component, stats) in &metrics.component_performance {
        println!("   - {}: {} executions, {:.1}% success rate", 
                 component, stats.executions, stats.success_rate * 100.0);
    }
    
    println!("\n✅ Integration example completed successfully!");
    
    Ok(())
}