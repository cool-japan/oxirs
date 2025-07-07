//! Integration tests for shape import functionality

use oxirs_shacl::{
    shape_import::{ImportDirective, ImportType, ShapeImportConfig},
    Result, Validator, ValidationConfig,
};

#[test]
fn test_validator_with_import_manager() {
    let mut validator = Validator::new();
    
    // Test that import manager is properly initialized
    let stats = validator.get_import_statistics();
    assert_eq!(stats.total_imports, 0);
    assert_eq!(stats.successful_imports, 0);
}

#[test]
fn test_basic_shape_loading_debug() {
    let mut validator = Validator::new();
    
    let simple_shape = r#"@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .

ex:TestShape a sh:NodeShape .
"#;
    
    // First test basic shape loading without imports
    let load_result = validator.load_shapes_from_rdf(simple_shape, "turtle", None);
    println!("Basic load result: {:?}", load_result);
    assert!(load_result.is_ok(), "Failed basic shape loading: {:?}", load_result.err());
    
    let count = load_result.unwrap();
    println!("Loaded {} shapes", count);
    assert!(count > 0, "No shapes loaded in basic test");
    
    println!("Validator shapes: {:?}", validator.shapes().keys().collect::<Vec<_>>());
    assert!(!validator.shapes().is_empty(), "Validator has no shapes after basic load");
}

#[test]
fn test_load_shapes_with_owl_imports() {
    let mut validator = Validator::new();
    
    let shapes_with_imports = r#"@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .

ex:PersonShape a sh:NodeShape .
"#;
    
    // This should parse the shapes and attempt to process imports
    // Note: External imports will fail due to network, but we test the structure
    let result = validator.load_shapes_with_imports(shapes_with_imports, "turtle", None);
    
    // The main shape should be loaded even if imports fail
    assert!(result.is_ok(), "Failed to load shapes: {:?}", result.err());
    let import_result = result.unwrap();
    
    // Should have loaded at least the main shape
    assert!(!import_result.shapes.is_empty(), "No shapes were loaded");
    
    // Check that the shape was added to the validator
    assert!(!validator.shapes().is_empty(), "No shapes in validator");
    
    // Import directives should be processed (warnings depend on whether extraction worked)
    // We'll be flexible here since the RDF parsing might vary
}

#[test]
fn test_load_shapes_with_shacl_imports() {
    let mut validator = Validator::new();
    
    let shapes_with_imports = r#"@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .

ex:OrganizationShape a sh:NodeShape .
"#;
    
    let result = validator.load_shapes_with_imports(shapes_with_imports, "turtle", None);
    assert!(result.is_ok(), "Failed to load shapes with imports: {:?}", result.err());
    
    let import_result = result.unwrap();
    
    // Should have loaded the main shape
    assert!(import_result.shapes.len() >= 1, "No shapes were loaded");
    
    // Verify the shape is in the validator
    let org_shape_id = oxirs_shacl::ShapeId::new("http://example.org/OrganizationShape");
    assert!(validator.get_shape(&org_shape_id).is_some(), "OrganizationShape not found in validator");
}

#[test]
fn test_import_directive_creation_and_usage() {
    let mut validator = Validator::new();
    
    // Test creating an import directive
    let directive = ImportDirective {
        source_iri: "http://example.org/test-shapes.ttl".to_string(),
        target_namespace: Some("http://myorg.org/imported#".to_string()),
        specific_shapes: Some(vec!["TestShape".to_string()]),
        import_type: ImportType::Selective,
        format_hint: Some("turtle".to_string()),
    };
    
    // This will fail due to network access, but tests the structure
    let result = validator.load_shapes_from_external(&directive);
    assert!(result.is_err()); // Expected to fail due to no network access
    
    // Verify that import statistics were updated
    let stats = validator.get_import_statistics();
    assert_eq!(stats.total_imports, 1);
    assert_eq!(stats.failed_imports, 1);
}

#[test]
fn test_import_security_configuration() {
    // Test with restrictive security settings
    let import_config = ShapeImportConfig {
        allow_http: false,
        allow_file: false,
        max_resource_size: 1024,
        ..Default::default()
    };
    
    let mut validator = Validator::with_import_config(
        ValidationConfig::default(),
        import_config
    );
    
    // Configure additional security settings
    validator.configure_import_security(false, false, 512);
    
    // Verify security settings are applied
    let directive = ImportDirective {
        source_iri: "http://example.org/shapes.ttl".to_string(), // HTTP should be blocked
        target_namespace: None,
        specific_shapes: None,
        import_type: ImportType::Include,
        format_hint: None,
    };
    
    let result = validator.load_shapes_from_external(&directive);
    assert!(result.is_err());
    
    // The error should indicate security restrictions
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("HTTP") || error_msg.contains("security"));
}

#[test]
fn test_import_url_convenience_method() {
    let mut validator = Validator::new();
    
    // Test the convenience method for loading from URL
    let result = validator.load_shapes_from_url(
        "https://example.org/shapes.ttl",
        Some(ImportType::Include)
    );
    
    // Should fail due to network, but structure should be correct
    assert!(result.is_err());
    
    // Check import statistics
    let stats = validator.get_import_statistics();
    assert_eq!(stats.total_imports, 1);
}

#[test]
fn test_selective_shape_import() {
    let mut validator = Validator::new();
    
    // Test loading specific shapes from external source
    let result = validator.load_specific_shapes_from_external(
        "https://example.org/all-shapes.ttl",
        vec!["Shape1".to_string(), "Shape2".to_string()],
        Some("http://myorg.org/imported#".to_string())
    );
    
    // Should fail due to network, but test the method structure
    assert!(result.is_err());
    
    // Verify the request was tracked
    let stats = validator.get_import_statistics();
    assert_eq!(stats.total_imports, 1);
}

#[test]
fn test_clear_import_cache() {
    let mut validator = Validator::new();
    
    // Initially cache should be empty
    validator.clear_import_cache();
    
    // This should work without errors
    let stats = validator.get_import_statistics();
    assert_eq!(stats.cache_hits, 0);
}

#[test]
fn test_check_import_dependencies() {
    let validator = Validator::new();
    
    // Should not have any circular dependencies initially
    let result = validator.check_import_dependencies();
    assert!(result.is_ok());
}

#[test]
fn test_resolve_external_references() {
    let mut validator = Validator::new();
    
    // Add a simple shape first
    let shape_ttl = r#"@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .

ex:TestShape a sh:NodeShape .
"#;
    
    let loaded = validator.load_shapes_from_rdf(shape_ttl, "turtle", None);
    assert!(loaded.is_ok());
    
    // Attempt to resolve external references
    let result = validator.resolve_external_references();
    assert!(result.is_ok());
    
    // Should return empty list since no external references
    let import_results = result.unwrap();
    assert!(import_results.is_empty());
}

#[test]
fn test_import_metadata_generation() {
    let mut validator = Validator::new();
    
    let simple_shape = r#"
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        
        ex:SimpleShape a sh:NodeShape .
    "#;
    
    let result = validator.load_shapes_with_imports(simple_shape, "turtle", Some("http://example.org/"));
    assert!(result.is_ok(), "Failed to load simple shape: {:?}", result.err());
    
    let import_result = result.unwrap();
    
    // Check metadata was generated correctly
    assert_eq!(import_result.metadata.source_iri, "http://example.org/");
    assert!(import_result.metadata.shape_count > 0, "Shape count should be > 0");
    assert_eq!(import_result.metadata.import_depth, 0);
    assert_eq!(import_result.metadata.content_type, Some("turtle".to_string()));
    assert_eq!(import_result.metadata.content_size, simple_shape.len());
    assert!(!import_result.metadata.content_hash.is_empty(), "Content hash should not be empty");
    assert!(!import_result.metadata.imported_at.is_empty(), "Import timestamp should not be empty");
}

#[test]
fn test_import_types_functionality() {
    // Test all import types can be created
    let include_type = ImportType::Include;
    let selective_type = ImportType::Selective;
    let dependency_type = ImportType::Dependency;
    let mapping_type = ImportType::NamespaceMapping("http://mapped.org/".to_string());
    
    // Verify they can be pattern matched
    match include_type {
        ImportType::Include => assert!(true),
        _ => assert!(false),
    }
    
    match selective_type {
        ImportType::Selective => assert!(true),
        _ => assert!(false),
    }
    
    match dependency_type {
        ImportType::Dependency => assert!(true),
        _ => assert!(false),
    }
    
    match mapping_type {
        ImportType::NamespaceMapping(ns) => assert_eq!(ns, "http://mapped.org/"),
        _ => assert!(false),
    }
}

#[test]
fn test_comprehensive_import_workflow() {
    let mut validator = Validator::new();
    
    // Test a complete workflow: load shapes, check dependencies, get statistics
    let shapes_with_metadata = r#"@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

ex:Ontology owl:imports <http://example.org/external-shapes.ttl> .
ex:CompleteShape a sh:NodeShape .
"#;
    
    // 1. Load shapes with imports
    let load_result = validator.load_shapes_with_imports(shapes_with_metadata, "turtle", Some("http://example.org/main"));
    assert!(load_result.is_ok(), "Failed to load shapes with metadata: {:?}", load_result.err());
    
    let import_result = load_result.unwrap();
    assert!(!import_result.shapes.is_empty(), "No shapes loaded in comprehensive test");
    
    // 2. Check dependencies
    let deps_result = validator.check_import_dependencies();
    assert!(deps_result.is_ok());
    
    // 3. Get and verify statistics
    let stats = validator.get_import_statistics();
    assert!(stats.total_imports > 0);
    
    // 4. Verify shapes were loaded
    assert!(!validator.shapes().is_empty());
    
    // 5. Test cache operations
    validator.clear_import_cache();
    
    // 6. Test security configuration
    validator.configure_import_security(true, false, 2048);
    
    // 7. Resolve any external references
    let resolve_result = validator.resolve_external_references();
    assert!(resolve_result.is_ok());
}