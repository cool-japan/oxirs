//! Test to reproduce the language tag bug

use oxirs_shacl::Validator;

#[test]
fn test_shape_with_language_tagged_name() {
    let shapes = r#"
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix wgs84: <http://www.w3.org/2003/01/geo/wgs84_pos#> .
@prefix shapes: <urn:continuum:shapes/> .

shapes:LatitudeProperty a sh:PropertyShape ;
    sh:path wgs84:lat ;
    sh:name "latitude"@en ;
    sh:datatype xsd:double ;
    sh:severity sh:Violation .
"#;

    let mut validator = Validator::new();
    let result = validator.load_shapes_from_rdf(shapes, "turtle", None);

    match &result {
        Ok(count) => println!("SUCCESS: Loaded {} shapes", count),
        Err(e) => println!("ERROR: {:?}", e),
    }

    assert!(
        result.is_ok(),
        "Should load shapes with language-tagged sh:name"
    );
}

#[test]
fn test_shape_with_language_tagged_message() {
    let shapes = r#"
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix shapes: <urn:continuum:shapes/> .

shapes:TestShape a sh:NodeShape ;
    sh:targetClass shapes:TestClass ;
    sh:message "Validation failed"@en .
"#;

    let mut validator = Validator::new();
    let result = validator.load_shapes_from_rdf(shapes, "turtle", None);

    match &result {
        Ok(count) => println!("SUCCESS: Loaded {} shapes", count),
        Err(e) => println!("ERROR: {:?}", e),
    }

    assert!(
        result.is_ok(),
        "Should load shapes with language-tagged sh:message"
    );
}

#[test]
fn test_shape_with_plain_message() {
    let shapes = r#"
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix shapes: <urn:continuum:shapes/> .

shapes:TestShape a sh:NodeShape ;
    sh:targetClass shapes:TestClass ;
    sh:message "Validation failed" .
"#;

    let mut validator = Validator::new();
    let result = validator.load_shapes_from_rdf(shapes, "turtle", None);

    match &result {
        Ok(count) => println!("SUCCESS: Loaded {} shapes", count),
        Err(e) => println!("ERROR: {:?}", e),
    }

    assert!(result.is_ok(), "Should load shapes with plain sh:message");
}
