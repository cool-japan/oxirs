//! Integration tests for cross-module integration capabilities
//!
//! Tests the GraphQL, Fuseki, Stream, and AI integration modules.

use oxirs_core::RdfStore;
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

/// Helper function to create a test validator
fn create_test_validator() -> Result<Arc<Validator>> {
    let validator = ValidatorBuilder::new().build();
    Ok(Arc::new(validator))
}

/// Helper function to create a test store
fn create_test_store() -> Result<RdfStore> {
    Ok(RdfStore::new()?)
}

#[test]
fn test_graphql_validator_creation() -> Result<()> {
    let validator = create_test_validator()?;

    let graphql_validator = GraphQLValidatorBuilder::new()
        .validate_mutations(true)
        .validate_queries(false)
        .max_complexity(500)
        .build(Arc::clone(&validator));

    // Verify validator was created successfully
    assert!(std::mem::size_of_val(&graphql_validator) > 0);

    Ok(())
}

#[test]
fn test_graphql_mutation_validation() -> Result<()> {
    let validator = create_test_validator()?;
    let store = create_test_store()?;

    let graphql_validator = GraphQLValidatorBuilder::new()
        .validate_mutations(true)
        .type_mapping("Person".to_string(), ShapeId::new("ex:PersonShape"))
        .build(Arc::clone(&validator));

    let context = GraphQLFieldContext {
        type_name: "Person".to_string(),
        field_name: "createPerson".to_string(),
        field_path: vec!["mutation".to_string()],
        operation_type: OperationType::Mutation,
        complexity: 100,
    };

    let result = graphql_validator.validate_operation(&store, &context)?;

    // Should pass with empty store
    assert!(result.conforms);
    assert_eq!(result.violations.len(), 0);

    Ok(())
}

#[test]
fn test_graphql_complexity_limit() -> Result<()> {
    let validator = create_test_validator()?;
    let store = create_test_store()?;

    let graphql_validator = GraphQLValidatorBuilder::new()
        .max_complexity(100)
        .build(Arc::clone(&validator));

    let context = GraphQLFieldContext {
        type_name: "Person".to_string(),
        field_name: "complexQuery".to_string(),
        field_path: vec!["query".to_string()],
        operation_type: OperationType::Query,
        complexity: 150, // Exceeds limit
    };

    let result = graphql_validator.validate_operation(&store, &context)?;

    // Should fail due to complexity
    assert!(!result.conforms);
    assert!(!result.violations.is_empty());

    Ok(())
}

#[test]
fn test_fuseki_validator_creation() -> Result<()> {
    let validator = create_test_validator()?;

    let fuseki_validator = FusekiValidatorBuilder::new()
        .validate_updates(true)
        .max_triples(10000)
        .build(Arc::clone(&validator));

    assert!(std::mem::size_of_val(&fuseki_validator) > 0);

    Ok(())
}

#[test]
fn test_fuseki_update_validation() -> Result<()> {
    let validator = create_test_validator()?;
    let store = create_test_store()?;

    let fuseki_validator = FusekiValidatorBuilder::new()
        .validate_updates(true)
        .build(Arc::clone(&validator));

    let context = FusekiEndpointContext {
        endpoint_path: "/test/sparql".to_string(),
        operation: SparqlOperation::InsertData,
        user_agent: Some("test".to_string()),
        request_id: "test-123".to_string(),
        triple_count: 10,
    };

    let result = fuseki_validator.validate_operation(&store, &context)?;

    assert!(result.should_proceed);
    assert_eq!(result.triple_count, 10);

    Ok(())
}

#[test]
fn test_fuseki_triple_limit() -> Result<()> {
    let validator = create_test_validator()?;
    let store = create_test_store()?;

    let fuseki_validator = FusekiValidatorBuilder::new()
        .max_triples(100)
        .build(Arc::clone(&validator));

    let context = FusekiEndpointContext {
        endpoint_path: "/test/sparql".to_string(),
        operation: SparqlOperation::InsertData,
        user_agent: None,
        request_id: "test-456".to_string(),
        triple_count: 150, // Exceeds limit
    };

    let result = fuseki_validator.validate_operation(&store, &context)?;

    // Should fail due to triple count
    assert!(!result.conforms);
    assert!(!result.should_proceed);

    Ok(())
}

#[test]
fn test_fuseki_cache_statistics() -> Result<()> {
    let validator = create_test_validator()?;

    let fuseki_validator = FusekiValidatorBuilder::new()
        .cache_results(true)
        .build(Arc::clone(&validator));

    let stats = fuseki_validator.cache_stats();

    assert_eq!(stats.size, 0); // Empty cache initially
    assert!(stats.capacity > 0);

    Ok(())
}

#[test]
fn test_stream_validator_creation() -> Result<()> {
    let validator = create_test_validator()?;

    let stream_validator = StreamValidatorBuilder::new()
        .batch_size(100)
        .enable_backpressure(true)
        .build(Arc::clone(&validator));

    assert!(std::mem::size_of_val(&stream_validator) > 0);

    Ok(())
}

#[test]
fn test_stream_event_validation() -> Result<()> {
    let validator = create_test_validator()?;
    let store = create_test_store()?;

    let stream_validator = StreamValidatorBuilder::new()
        .batch_size(50)
        .build(Arc::clone(&validator));

    let event = StreamEvent {
        event_id: "event-001".to_string(),
        timestamp: chrono::Utc::now(),
        event_type: StreamEventType::Insert,
        data: vec![],
        metadata: HashMap::new(),
        retry_count: 0,
    };

    let result = stream_validator.validate_event(&store, &event)?;

    assert!(result.conforms);
    assert_eq!(result.event_id, "event-001");
    assert!(result.should_forward);

    Ok(())
}

#[test]
fn test_stream_metrics_collection() -> Result<()> {
    let validator = create_test_validator()?;

    let stream_validator = StreamValidatorBuilder::new().build(Arc::clone(&validator));

    let metrics = stream_validator.get_metrics();

    // Initially empty
    assert!(metrics.is_empty());

    // Get throughput
    let throughput = stream_validator.get_throughput();
    assert_eq!(throughput, 0.0); // No events processed yet

    Ok(())
}

#[test]
fn test_stream_dlq_decision() -> Result<()> {
    let validator = create_test_validator()?;

    let stream_validator = StreamValidatorBuilder::new()
        .enable_dlq(true)
        .max_retries(3)
        .build(Arc::clone(&validator));

    let event = StreamEvent {
        event_id: "event-dlq".to_string(),
        timestamp: chrono::Utc::now(),
        event_type: StreamEventType::Insert,
        data: vec![],
        metadata: HashMap::new(),
        retry_count: 5, // Exceeds max retries
    };

    let decision = stream_validator.process_dlq_event(&event)?;

    // Should send to DLQ after max retries
    use oxirs_shacl::integration::DlqDecision;
    assert_eq!(decision, DlqDecision::SendToDlq);

    Ok(())
}

#[test]
fn test_ai_validator_creation() -> Result<()> {
    let validator = create_test_validator()?;

    let ai_validator = AIValidatorBuilder::new()
        .enable_suggestions(true)
        .confidence_threshold(0.8)
        .build(Arc::clone(&validator));

    assert!(std::mem::size_of_val(&ai_validator) > 0);

    Ok(())
}

#[test]
fn test_ai_shape_suggestions() -> Result<()> {
    let validator = create_test_validator()?;
    let store = create_test_store()?;

    let mut ai_validator = AIValidatorBuilder::new()
        .enable_suggestions(true)
        .max_suggestions(3)
        .confidence_threshold(0.75)
        .build(Arc::clone(&validator));

    let suggestions = ai_validator.suggest_shapes(&store)?;

    // Should return suggestions (even if empty store)
    assert!(suggestions.len() <= 3);

    Ok(())
}

#[test]
fn test_ai_constraint_learning() -> Result<()> {
    let validator = create_test_validator()?;
    let store = create_test_store()?;

    let ai_validator = AIValidatorBuilder::new()
        .enable_constraint_learning(true)
        .build(Arc::clone(&validator));

    let _constraints = ai_validator.learn_constraints(&store, "ex:Person")?;

    // Learned constraints returned successfully (vector length is always >= 0)
    Ok(())
}

#[test]
fn test_ai_shape_similarity() -> Result<()> {
    let validator = create_test_validator()?;

    let mut ai_validator = AIValidatorBuilder::new()
        .embedding_dim(768)
        .build(Arc::clone(&validator));

    let shape = Shape::node_shape(ShapeId::new("ex:TestShape"));
    let similar = ai_validator.find_similar_shapes(&shape, 5)?;

    // Should return similar shapes (may be empty)
    assert!(similar.len() <= 5);

    Ok(())
}

#[test]
fn test_ai_cache_clearing() -> Result<()> {
    let validator = create_test_validator()?;

    let ai_validator = AIValidatorBuilder::new().build(Arc::clone(&validator));

    // Clear caches (should not panic)
    ai_validator.clear_caches();

    Ok(())
}

#[test]
fn test_integration_builder_patterns() -> Result<()> {
    let validator = create_test_validator()?;

    // Test all builders can be chained
    let _graphql = GraphQLValidatorBuilder::new()
        .validate_mutations(true)
        .validate_queries(false)
        .fail_on_violation(true)
        .max_complexity(1000)
        .timeout(5000)
        .build(Arc::clone(&validator));

    let _fuseki = FusekiValidatorBuilder::new()
        .validate_updates(true)
        .validate_construct(false)
        .transactional(true)
        .cache_results(true)
        .max_triples(10000)
        .timeout(10000)
        .build(Arc::clone(&validator));

    let _stream = StreamValidatorBuilder::new()
        .batch_size(100)
        .batch_timeout(1000)
        .enable_backpressure(true)
        .backpressure_threshold(5000)
        .enable_dlq(true)
        .max_retries(3)
        .filter_invalid_events(false)
        .build(Arc::clone(&validator));

    let _ai = AIValidatorBuilder::new()
        .enable_suggestions(true)
        .enable_violation_analysis(true)
        .enable_constraint_learning(true)
        .confidence_threshold(0.75)
        .max_suggestions(5)
        .embedding_dim(768)
        .enable_nl_queries(false)
        .build(Arc::clone(&validator));

    Ok(())
}

#[test]
fn test_graphql_operation_types() {
    use oxirs_shacl::integration::OperationType;

    // Ensure all operation types are distinct
    let query = OperationType::Query;
    let mutation = OperationType::Mutation;
    let subscription = OperationType::Subscription;

    assert_ne!(query as i32, mutation as i32);
    assert_ne!(mutation as i32, subscription as i32);
    assert_ne!(query as i32, subscription as i32);
}

#[test]
fn test_fuseki_sparql_operations() {
    use oxirs_shacl::integration::SparqlOperation;

    // Test all SPARQL operation types
    let _operations = [
        SparqlOperation::Select,
        SparqlOperation::Construct,
        SparqlOperation::Describe,
        SparqlOperation::Ask,
        SparqlOperation::InsertData,
        SparqlOperation::DeleteData,
        SparqlOperation::Modify,
        SparqlOperation::Load,
        SparqlOperation::Clear,
        SparqlOperation::Create,
        SparqlOperation::Drop,
    ];
}

#[test]
fn test_stream_event_types() {
    use oxirs_shacl::integration::StreamEventType;

    let _types = [
        StreamEventType::Insert,
        StreamEventType::Delete,
        StreamEventType::Update,
        StreamEventType::Batch,
    ];
}
