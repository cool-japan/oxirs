//! Integration tests for complete model lifecycle workflows
//!
//! These tests demonstrate how Query, Comparison, and Transformation modules
//! work together in real-world scenarios.

use oxirs_samm::{
    comparison::ModelComparison,
    metamodel::{Aspect, Characteristic, CharacteristicKind, Constraint, ModelElement, Property},
    query::ModelQuery,
    transformation::ModelTransformation,
};

/// Helper function to create a sample aspect for testing
fn create_sample_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Test Aspect".to_string());
    aspect
        .metadata
        .add_description("en".to_string(), "A test aspect".to_string());

    // Add required property
    let mut prop1 = Property::new("urn:samm:org.example:1.0.0#requiredProp".to_string());
    prop1
        .metadata
        .add_preferred_name("en".to_string(), "Required Property".to_string());
    prop1.characteristic = Some(Characteristic {
        metadata: oxirs_samm::metamodel::ElementMetadata::new(
            "urn:samm:org.example:1.0.0#TextCharacteristic".to_string(),
        ),
        kind: CharacteristicKind::Trait,
        data_type: Some("xsd:string".to_string()),
        constraints: Vec::new(),
    });
    prop1.optional = false;

    // Add optional property
    let mut prop2 = Property::new("urn:samm:org.example:1.0.0#optionalProp".to_string());
    prop2
        .metadata
        .add_preferred_name("en".to_string(), "Optional Property".to_string());
    prop2.characteristic = Some(Characteristic {
        metadata: oxirs_samm::metamodel::ElementMetadata::new(
            "urn:samm:org.example:1.0.0#IntCharacteristic".to_string(),
        ),
        kind: CharacteristicKind::Trait,
        data_type: Some("xsd:int".to_string()),
        constraints: Vec::new(),
    });
    prop2.optional = true;

    // Add collection property
    let mut prop3 = Property::new("urn:samm:org.example:1.0.0#itemsList".to_string());
    prop3
        .metadata
        .add_preferred_name("en".to_string(), "Items List".to_string());
    prop3.characteristic = Some(Characteristic {
        metadata: oxirs_samm::metamodel::ElementMetadata::new(
            "urn:samm:org.example:1.0.0#Collection".to_string(),
        ),
        kind: CharacteristicKind::Collection {
            element_characteristic: None,
        },
        data_type: Some("xsd:string".to_string()),
        constraints: Vec::new(),
    });
    prop3.is_collection = true;

    aspect.add_property(prop1);
    aspect.add_property(prop2);
    aspect.add_property(prop3);

    aspect
}

#[test]
fn test_query_api_basic_usage() {
    // Demonstrates basic query API usage
    let aspect = create_sample_aspect();
    let query = ModelQuery::new(&aspect);

    // Find properties by type
    let required = query.find_required_properties();
    assert_eq!(required.len(), 2); // requiredProp + itemsList

    let optional = query.find_optional_properties();
    assert_eq!(optional.len(), 1); // optionalProp

    let collections = query.find_properties_with_collection_characteristic();
    assert_eq!(collections.len(), 1); // itemsList

    // Calculate complexity
    let metrics = query.complexity_metrics();
    assert_eq!(metrics.total_properties, 3);
    // max_nesting_depth is a valid metric (always >= 0 for usize)
}

#[test]
fn test_transformation_api_basic_usage() {
    // Demonstrates basic transformation API usage
    let mut aspect = create_sample_aspect();

    // Apply a simple transformation
    let mut transformation = ModelTransformation::new(&mut aspect);
    transformation.update_preferred_name("en", "Modified Aspect");
    let result = transformation.apply();

    assert!(result.failed_transformations.is_empty());
    assert!(result.transformations_applied > 0);

    // Verify the change
    assert_eq!(
        aspect.metadata().get_preferred_name("en"),
        Some("Modified Aspect")
    );
}

#[test]
fn test_comparison_api_basic_usage() {
    // Demonstrates basic comparison API usage
    let original = create_sample_aspect();
    let mut modified = original.clone();

    // Modify the aspect
    modified
        .metadata
        .add_preferred_name("en".to_string(), "Modified Aspect".to_string());

    // Compare
    let comparison = ModelComparison::compare(&original, &modified);

    assert!(comparison.has_changes());
    assert!(!comparison.metadata_changes.is_empty());
}

#[test]
fn test_query_then_transform_workflow() {
    // Query model, then transform based on findings
    let mut aspect = create_sample_aspect();

    // Step 1: Query to find optional properties
    let query = ModelQuery::new(&aspect);
    let optional_count_before = query.find_optional_properties().len();
    assert_eq!(optional_count_before, 1);

    // Step 2: Transform - make requiredProp optional
    let mut transformation = ModelTransformation::new(&mut aspect);
    transformation.make_property_optional("requiredProp");
    let _ = transformation.apply();

    // Step 3: Query again to verify change
    let query_after = ModelQuery::new(&aspect);
    let optional_count_after = query_after.find_optional_properties().len();
    assert_eq!(optional_count_after, 2); // Now both requiredProp and optionalProp are optional
}

#[test]
fn test_transform_then_compare_workflow() {
    // Transform a model then compare with original
    let original = create_sample_aspect();
    let mut modified = original.clone();

    // Transform the modified version
    let mut transformation = ModelTransformation::new(&mut modified);
    transformation.rename_property("requiredProp", "mandatoryField");
    let result = transformation.apply();

    assert!(result.failed_transformations.is_empty());

    // Compare the two versions
    let comparison = ModelComparison::compare(&original, &modified);

    // Should detect changes (property was renamed)
    assert!(comparison.has_changes());
    // Note: Rename detection may vary - the key is that changes were detected
    assert!(!comparison.properties_removed.is_empty() || !comparison.properties_added.is_empty());
}

#[test]
fn test_complexity_analysis_workflow() {
    // Analyze model complexity
    let mut aspect = create_sample_aspect();

    // Add more properties
    for i in 0..5 {
        let mut prop = Property::new(format!("urn:samm:org.example:1.0.0#prop{}", i));
        prop.characteristic = Some(Characteristic {
            metadata: oxirs_samm::metamodel::ElementMetadata::new(format!(
                "urn:samm:org.example:1.0.0#Char{}",
                i
            )),
            kind: CharacteristicKind::Trait,
            data_type: Some("xsd:string".to_string()),
            constraints: Vec::new(),
        });
        aspect.add_property(prop);
    }

    // Analyze complexity
    let query = ModelQuery::new(&aspect);
    let metrics = query.complexity_metrics();

    assert_eq!(metrics.total_properties, 8); // 3 original + 5 new
    assert_eq!(metrics.total_operations, 0);
}

#[test]
fn test_dependency_graph_construction() {
    // Build and analyze dependency graph
    let aspect = create_sample_aspect();
    let query = ModelQuery::new(&aspect);

    let dependencies = query.build_dependency_graph();
    assert!(!dependencies.is_empty());

    // Check for circular dependencies (should be none)
    let circular = query.detect_circular_dependencies();
    assert!(circular.is_empty());
}

#[test]
fn test_property_grouping_by_characteristic() {
    // Group properties by characteristic type
    let aspect = create_sample_aspect();
    let query = ModelQuery::new(&aspect);

    let grouped = query.group_properties_by_characteristic_type();

    assert!(grouped.contains_key("Trait"));
    assert!(grouped.contains_key("Collection"));

    let trait_props = &grouped["Trait"];
    assert_eq!(trait_props.len(), 2); // requiredProp + optionalProp
}
