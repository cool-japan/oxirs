//! # Cross-Module Integration Example
//!
//! This example demonstrates how to use oxirs-shacl's integration features
//! with GraphQL, Fuseki, Streaming, and AI modules.
//!
//! ## Features Demonstrated
//!
//! 1. **GraphQL Integration**: Validate GraphQL mutations and queries
//! 2. **Fuseki Integration**: Validate SPARQL UPDATE operations
//! 3. **Stream Integration**: Real-time validation of RDF events
//! 4. **AI Integration**: ML-powered shape suggestions and violation analysis
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example cross_module_integration --all-features
//! ```

use oxirs_core::{RdfStore, Store};
use oxirs_shacl::{
    integration::{
        AIValidatorBuilder, FusekiEndpointContext, FusekiValidatorBuilder, GraphQLFieldContext,
        GraphQLValidatorBuilder, OperationType, SparqlOperation, StreamEvent, StreamEventType,
        StreamValidatorBuilder,
    },
    Result, Shape, ShapeId, Validator, ValidatorBuilder,
};
use std::collections::HashMap;
use std::sync::Arc;

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== OxiRS SHACL Cross-Module Integration Example ===\n");

    // Setup common components
    let store = setup_store()?;
    let shapes = setup_shapes()?;
    let base_validator = setup_validator(&shapes)?;

    // Demonstrate each integration
    demo_graphql_integration(&store, &base_validator)?;
    demo_fuseki_integration(&store, &base_validator)?;
    demo_stream_integration(&store, &base_validator)?;
    demo_ai_integration(&store, &base_validator)?;

    println!("\n=== Integration Examples Complete ===");

    Ok(())
}

/// Setup an in-memory RDF store with sample data
fn setup_store() -> Result<RdfStore> {
    println!("Setting up RDF store with sample data...");

    let store = RdfStore::new()?;

    // In a real application, you would load RDF data here
    // For this example, we'll use an empty store

    println!("  ✓ Store initialized\n");

    Ok(store)
}

/// Setup SHACL shapes for validation
fn setup_shapes() -> Result<Vec<Shape>> {
    println!("Setting up SHACL shapes...");

    let mut shapes = Vec::new();

    // Create a simple node shape
    let person_shape = Shape::node_shape(ShapeId::new("ex:PersonShape"));

    shapes.push(person_shape);

    println!("  ✓ {} shapes created\n", shapes.len());

    Ok(shapes)
}

/// Setup base SHACL validator
fn setup_validator(_shapes: &[Shape]) -> Result<Arc<Validator>> {
    println!("Setting up base SHACL validator...");

    let validator = ValidatorBuilder::new().build();

    println!("  ✓ Validator initialized\n");

    Ok(Arc::new(validator))
}

/// Demonstrate GraphQL integration
fn demo_graphql_integration(store: &dyn Store, validator: &Arc<Validator>) -> Result<()> {
    println!("--- GraphQL Integration Demo ---");

    // Configure GraphQL validator
    let graphql_validator = GraphQLValidatorBuilder::new()
        .validate_mutations(true)
        .validate_queries(false)
        .max_complexity(1000)
        .type_mapping("Person".to_string(), ShapeId::new("ex:PersonShape"))
        .build(Arc::clone(validator));

    println!("  ✓ GraphQL validator configured");

    // Simulate a GraphQL mutation
    let mutation_context = GraphQLFieldContext {
        type_name: "Person".to_string(),
        field_name: "createPerson".to_string(),
        field_path: vec!["mutation".to_string(), "createPerson".to_string()],
        operation_type: OperationType::Mutation,
        complexity: 150,
    };

    println!("  → Validating GraphQL mutation...");

    match graphql_validator.validate_operation(store, &mutation_context) {
        Ok(result) => {
            println!("    Conforms: {}", result.conforms);
            println!("    Violations: {}", result.violations.len());
            println!("    Warnings: {}", result.warnings.len());

            if result.should_block_operation() {
                println!("    ⚠ Operation would be blocked");
            } else {
                println!("    ✓ Operation would be allowed");
            }
        }
        Err(e) => {
            println!("    ✗ Validation error: {}", e);
        }
    }

    // Demonstrate mutation input validation
    println!("  → Validating mutation input data...");

    let input_data: HashMap<String, serde_json::Value> = HashMap::new();

    match graphql_validator.validate_mutation_input(store, "Person", &input_data) {
        Ok(result) => {
            println!(
                "    Input validation: {}",
                if result.conforms {
                    "✓ passed"
                } else {
                    "✗ failed"
                }
            );
        }
        Err(e) => {
            println!("    ✗ Input validation error: {}", e);
        }
    }

    println!();

    Ok(())
}

/// Demonstrate Fuseki integration
fn demo_fuseki_integration(store: &dyn Store, validator: &Arc<Validator>) -> Result<()> {
    println!("--- Fuseki Integration Demo ---");

    // Configure Fuseki validator
    let fuseki_validator = FusekiValidatorBuilder::new()
        .validate_updates(true)
        .validate_construct(false)
        .max_triples(10000)
        .transactional(true)
        .cache_results(true)
        .endpoint_shapes(
            "/dataset/sparql".to_string(),
            vec![ShapeId::new("ex:PersonShape")],
        )
        .build(Arc::clone(validator));

    println!("  ✓ Fuseki validator configured");

    // Simulate a SPARQL UPDATE operation
    let update_context = FusekiEndpointContext {
        endpoint_path: "/dataset/sparql".to_string(),
        operation: SparqlOperation::InsertData,
        user_agent: Some("OxiRS-SHACL/0.1.0".to_string()),
        request_id: uuid::Uuid::new_v4().to_string(),
        triple_count: 50,
    };

    println!("  → Validating SPARQL UPDATE operation...");

    match fuseki_validator.validate_operation(store, &update_context) {
        Ok(result) => {
            println!("    Conforms: {}", result.conforms);
            println!("    Validation time: {} ms", result.validation_time_ms);
            println!("    Triples validated: {}", result.triple_count);
            println!("    Violations: {}", result.violations.len());

            if result.should_proceed {
                println!("    ✓ Operation would proceed");
            } else {
                println!("    ⚠ Operation would be blocked");
            }

            // Get HTTP status code
            println!("    HTTP status: {}", result.http_status_code());
        }
        Err(e) => {
            println!("    ✗ Validation error: {}", e);
        }
    }

    // Show cache statistics
    let cache_stats = fuseki_validator.cache_stats();
    println!("  → Cache statistics:");
    println!("    Size: {}", cache_stats.size);
    println!("    Capacity: {}", cache_stats.capacity);

    println!();

    Ok(())
}

/// Demonstrate Stream integration
fn demo_stream_integration(store: &dyn Store, validator: &Arc<Validator>) -> Result<()> {
    println!("--- Stream Integration Demo ---");

    // Configure Stream validator
    let stream_validator = StreamValidatorBuilder::new()
        .batch_size(100)
        .batch_timeout(1000)
        .enable_backpressure(true)
        .backpressure_threshold(5000)
        .enable_dlq(true)
        .max_retries(3)
        .filter_invalid_events(false)
        .stream_shapes(vec![ShapeId::new("ex:PersonShape")])
        .build(Arc::clone(validator));

    println!("  ✓ Stream validator configured");

    // Simulate a stream event
    let event = StreamEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now(),
        event_type: StreamEventType::Insert,
        data: vec![],
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("source".to_string(), "kafka".to_string());
            meta.insert("topic".to_string(), "rdf-events".to_string());
            meta
        },
        retry_count: 0,
    };

    println!("  → Validating stream event...");
    println!("    Event ID: {}", event.event_id);
    println!("    Event type: {:?}", event.event_type);

    match stream_validator.validate_event(store, &event) {
        Ok(result) => {
            println!("    Conforms: {}", result.conforms);
            println!("    Validation time: {} ms", result.validation_time_ms);
            println!("    Should forward: {}", result.should_forward);
            println!("    Should retry: {}", result.should_retry);

            let routing = result.routing_decision();
            println!("    Routing decision: {:?}", routing);
        }
        Err(e) => {
            println!("    ✗ Validation error: {}", e);
        }
    }

    // Show throughput metrics
    let throughput = stream_validator.get_throughput();
    println!("  → Throughput: {:.2} events/sec", throughput);

    // Show all metrics
    let metrics = stream_validator.get_metrics();
    println!("  → Metrics collected:");
    for (key, value) in metrics {
        println!("    {}: {:.2}", key, value);
    }

    println!();

    Ok(())
}

/// Demonstrate AI integration
fn demo_ai_integration(store: &dyn Store, validator: &Arc<Validator>) -> Result<()> {
    println!("--- AI Integration Demo ---");

    // Configure AI validator
    let mut ai_validator = AIValidatorBuilder::new()
        .enable_suggestions(true)
        .enable_violation_analysis(true)
        .enable_constraint_learning(true)
        .confidence_threshold(0.75)
        .max_suggestions(5)
        .embedding_dim(768)
        .enable_nl_queries(false)
        .build(Arc::clone(validator));

    println!("  ✓ AI validator configured");

    // Demonstrate shape suggestions
    println!("  → Generating AI-powered shape suggestions...");

    match ai_validator.suggest_shapes(store) {
        Ok(suggestions) => {
            println!("    Generated {} suggestions", suggestions.len());

            for (i, suggestion) in suggestions.iter().enumerate() {
                println!("    Suggestion {}:", i + 1);
                println!("      Shape type: {}", suggestion.shape_type);
                println!("      Target class: {}", suggestion.target_class);
                println!("      Confidence: {:.2}%", suggestion.confidence * 100.0);
                println!("      Rationale: {}", suggestion.rationale);
            }
        }
        Err(e) => {
            println!("    ✗ Suggestion error: {}", e);
        }
    }

    // Demonstrate constraint learning
    println!("  → Learning constraints from data...");

    match ai_validator.learn_constraints(store, "ex:Person") {
        Ok(constraints) => {
            println!("    Learned {} constraints", constraints.len());

            for (i, constraint) in constraints.iter().enumerate() {
                println!("    Constraint {}:", i + 1);
                println!("      Type: {}", constraint.constraint_type);
                println!("      Property: {}", constraint.property_path);
                println!("      Value: {}", constraint.value);
                println!("      Confidence: {:.2}%", constraint.confidence * 100.0);
                println!("      Support: {:.2}%", constraint.support * 100.0);
            }
        }
        Err(e) => {
            println!("    ✗ Learning error: {}", e);
        }
    }

    // Demonstrate shape similarity search
    println!("  → Finding similar shapes...");

    let example_shape = Shape::node_shape(ShapeId::new("ex:PersonShape"));

    match ai_validator.find_similar_shapes(&example_shape, 3) {
        Ok(similar) => {
            println!("    Found {} similar shapes", similar.len());

            for (i, sim) in similar.iter().enumerate() {
                println!("    Similar shape {}:", i + 1);
                println!("      Shape ID: {}", sim.shape_id);
                println!("      Similarity: {:.2}%", sim.similarity_score * 100.0);
            }
        }
        Err(e) => {
            println!("    ✗ Similarity search error: {}", e);
        }
    }

    println!();

    Ok(())
}
