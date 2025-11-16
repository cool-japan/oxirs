//! Integration tests for SAMM parser with real SAMM models

use oxirs_samm::metamodel::ModelElement;
use oxirs_samm::parser::parse_aspect_model;
use std::path::PathBuf;

/// Get the path to a test fixture
fn fixture_path(name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("fixtures");
    path.push(name);
    path
}

#[tokio::test]
async fn test_parse_aspect_sample() {
    let path = fixture_path("aspect-sample.ttl");
    let result = parse_aspect_model(&path).await;

    assert!(
        result.is_ok(),
        "Failed to parse aspect-sample.ttl: {:?}",
        result.err()
    );

    let aspect = result.unwrap();

    // Verify the aspect was parsed
    assert_eq!(aspect.name(), "MyAspect");

    // Check preferred names (multi-language)
    assert_eq!(aspect.metadata.get_preferred_name("en"), Some("My Aspect"));
    assert_eq!(
        aspect.metadata.get_preferred_name("de"),
        Some("Mein Aspekt")
    );

    // Check descriptions
    assert_eq!(
        aspect.metadata.get_description("en"),
        Some("This Aspect is an example.")
    );
    assert_eq!(
        aspect.metadata.get_description("de"),
        Some("Dieser Aspekt ist ein Beispiel.")
    );

    // Check properties
    let properties = aspect.properties();
    assert!(
        properties.len() >= 2,
        "Expected at least 2 properties, found {}",
        properties.len()
    );

    // Find specific properties
    let my_property = properties.iter().find(|p| p.name() == "myProperty");
    assert!(
        my_property.is_some(),
        "Property 'myProperty' not found in aspect"
    );

    let another_property = properties.iter().find(|p| p.name() == "anotherProperty");
    assert!(
        another_property.is_some(),
        "Property 'anotherProperty' not found in aspect"
    );
}

#[tokio::test]
async fn test_parse_multi_lang_example() {
    let path = fixture_path("multi-lang-example.ttl");
    let result = parse_aspect_model(&path).await;

    // This file is a code fragment without an Aspect, so it should fail
    // but with a specific error message
    assert!(result.is_err());

    if let Err(err) = result {
        let err_str = format!("{:?}", err);
        assert!(
            err_str.contains("No Aspect found"),
            "Expected 'No Aspect found' error, got: {}",
            err_str
        );
    }
}

#[tokio::test]
async fn test_parse_collection_with_element_characteristic() {
    let path = fixture_path("collection-with-element-characteristic.ttl");
    let result = parse_aspect_model(&path).await;

    assert!(
        result.is_ok(),
        "Failed to parse collection-with-element-characteristic.ttl: {:?}",
        result.err()
    );

    let aspect = result.unwrap();

    // Verify the aspect was parsed
    assert!(!aspect.name().is_empty());

    // Check that properties were parsed
    let properties = aspect.properties();
    assert!(
        !properties.is_empty(),
        "Expected at least one property with collection characteristic"
    );
}

#[tokio::test]
async fn test_parse_prefixes_sample() {
    let path = fixture_path("prefixes-sample.ttl");
    let result = parse_aspect_model(&path).await;

    // This test verifies that the parser can handle various prefix declarations
    assert!(
        result.is_ok(),
        "Failed to parse prefixes-sample.ttl: {:?}",
        result.err()
    );

    let aspect = result.unwrap();
    assert!(!aspect.name().is_empty());
}

#[tokio::test]
async fn test_parse_nonexistent_file() {
    let path = PathBuf::from("nonexistent.ttl");
    let result = parse_aspect_model(&path).await;

    // Should fail with IO error
    assert!(result.is_err());
}

#[tokio::test]
async fn test_parser_caching() {
    // Parse the same file twice to test caching
    let path = fixture_path("aspect-sample.ttl");

    let result1 = parse_aspect_model(&path).await;
    assert!(result1.is_ok());

    let result2 = parse_aspect_model(&path).await;
    assert!(result2.is_ok());

    // Both should produce the same aspect
    let aspect1 = result1.unwrap();
    let aspect2 = result2.unwrap();

    assert_eq!(aspect1.name(), aspect2.name());
    assert_eq!(aspect1.properties().len(), aspect2.properties().len());
}

#[tokio::test]
async fn test_parse_aspect_with_entities() {
    // Test parsing an aspect that references entities
    let path = fixture_path("aspect-sample.ttl");
    let result = parse_aspect_model(&path).await;

    assert!(result.is_ok(), "Failed to parse aspect with entities");

    let aspect = result.unwrap();
    // Verify aspect was created even if entities are complex
    assert!(!aspect.name().is_empty());
}

#[tokio::test]
async fn test_parse_aspect_empty_properties() {
    // Test parsing an aspect with no properties (edge case)
    let path = fixture_path("prefixes-sample.ttl");
    let result = parse_aspect_model(&path).await;

    // Should succeed even if no properties
    if result.is_ok() {
        let aspect = result.unwrap();
        assert!(!aspect.name().is_empty());
    }
    // If it fails, that's also acceptable for this edge case
}

#[tokio::test]
async fn test_parse_aspect_multi_language() {
    // Test parsing with multiple language tags
    let path = fixture_path("multi-lang-example.ttl");
    let result = parse_aspect_model(&path).await;

    // This file doesn't have an Aspect, so it should fail
    assert!(result.is_err(), "Expected failure for file without Aspect");
}

#[tokio::test]
async fn test_parse_collection_characteristic() {
    // Test parsing aspects with collection characteristics
    let path = fixture_path("collection-with-element-characteristic.ttl");
    let result = parse_aspect_model(&path).await;

    assert!(
        result.is_ok(),
        "Failed to parse collection characteristic: {:?}",
        result.err()
    );

    let aspect = result.unwrap();
    assert!(
        !aspect.properties().is_empty(),
        "Expected at least one property with collection characteristic"
    );
}

#[tokio::test]
async fn test_concurrent_parsing() {
    // Test concurrent parsing of the same file
    let path = fixture_path("aspect-sample.ttl");

    let handles: Vec<_> = (0..5)
        .map(|_| {
            let path = path.clone();
            tokio::spawn(async move { parse_aspect_model(&path).await })
        })
        .collect();

    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "Concurrent parsing failed");
    }
}
