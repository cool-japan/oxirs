use serde_json::Value;

use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

use super::openapi_generator::OpenApiGenerator;
use super::openapi_serializer::to_kebab_case;
use super::openapi_serializer::xsd_to_openapi_type;
use super::openapi_types::{OpenApiOptions, OpenApiVersion, PaginationConfig};

fn movement_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Movement".to_string());
    aspect
        .metadata
        .add_description("en".to_string(), "Describes movement telemetry".to_string());

    let char = Characteristic::new(
        "urn:samm:org.example:1.0.0#SpeedChar".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:kilometrePerHour".to_string(),
        },
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());

    let prop =
        Property::new("urn:samm:org.example:1.0.0#speed".to_string()).with_characteristic(char);

    aspect.add_property(prop);
    aspect
}

#[test]
fn test_generate_openapi_version() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.2.3", "/api/v1/aspects");
    let spec = gen.generate(&aspect).expect("generation should succeed");
    assert_eq!(spec["openapi"], "3.0.3");
    assert_eq!(spec["info"]["version"], "1.2.3");
}

#[test]
fn test_path_is_present() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths should be object");
    assert!(!paths.is_empty(), "paths should not be empty");
    let path_key = paths.keys().next().expect("at least one path");
    assert!(
        path_key.contains("movement"),
        "path should contain aspect name"
    );
}

#[test]
fn test_get_operation_present_by_default() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    assert!(
        path_item.get("get").is_some(),
        "GET operation should be present"
    );
}

#[test]
fn test_post_not_included_by_default() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    assert!(
        path_item.get("post").is_none(),
        "POST should not be present by default"
    );
}

#[test]
fn test_post_included_when_enabled() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects").with_post();
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    assert!(path_item.get("post").is_some(), "POST should be present");
}

#[test]
fn test_components_schemas_contains_aspect() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let schemas = spec["components"]["schemas"].as_object().expect("schemas");
    assert!(
        schemas.contains_key("Movement"),
        "schemas should include Movement"
    );
}

#[test]
fn test_aspect_schema_has_properties() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let schemas = gen.build_schemas(&aspect).expect("build_schemas");
    assert!(schemas["Movement"]["properties"]["speed"].is_object());
}

#[test]
fn test_aspect_schema_required_field() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let schemas = gen.build_schemas(&aspect).expect("build_schemas");
    let required = schemas["Movement"]["required"]
        .as_array()
        .expect("required should be array");
    assert!(required.iter().any(|v| v == "speed"));
}

#[test]
fn test_measurement_type_is_number() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let schemas = gen.build_schemas(&aspect).expect("build_schemas");
    assert_eq!(schemas["Movement"]["properties"]["speed"]["type"], "number");
}

#[test]
fn test_info_description_present() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let spec = gen.generate(&aspect).expect("generation should succeed");
    assert!(spec["info"]["description"].is_string());
}

#[test]
fn test_response_contains_success_code() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    let get_op = &path_item["get"];
    assert!(get_op["responses"]["200"].is_object());
}

#[test]
fn test_delete_responds_204() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects").with_delete();
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    let del_op = &path_item["delete"];
    assert!(del_op["responses"]["204"].is_object());
}

#[test]
fn test_enumeration_generates_enum_schema() {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#TestAspect".to_string());
    let char = Characteristic::new(
        "urn:samm:org.example:1.0.0#StatusEnum".to_string(),
        CharacteristicKind::Enumeration {
            values: vec!["Active".to_string(), "Inactive".to_string()],
        },
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
    let prop =
        Property::new("urn:samm:org.example:1.0.0#status".to_string()).with_characteristic(char);
    aspect.add_property(prop);

    let gen = OpenApiGenerator::new("1.0.0", "/api");
    let schemas = gen.build_schemas(&aspect).expect("build_schemas");
    let status = &schemas["TestAspect"]["properties"]["status"];
    assert!(status["enum"].is_array());
}

#[test]
fn test_to_kebab_case() {
    assert_eq!(to_kebab_case("Movement"), "movement");
    assert_eq!(to_kebab_case("MyAspect"), "my-aspect");
    assert_eq!(to_kebab_case("speed"), "speed");
}

#[test]
fn test_xsd_to_openapi_type_mapping() {
    assert_eq!(
        xsd_to_openapi_type("http://www.w3.org/2001/XMLSchema#boolean"),
        "boolean"
    );
    assert_eq!(
        xsd_to_openapi_type("http://www.w3.org/2001/XMLSchema#int"),
        "integer"
    );
    assert_eq!(
        xsd_to_openapi_type("http://www.w3.org/2001/XMLSchema#float"),
        "number"
    );
    assert_eq!(
        xsd_to_openapi_type("http://www.w3.org/2001/XMLSchema#string"),
        "string"
    );
}

fn v31_gen() -> OpenApiGenerator {
    let options = OpenApiOptions {
        version: OpenApiVersion::V31,
        ..OpenApiOptions::default()
    };
    OpenApiGenerator::with_options(options)
}

fn aspect_with_optional_property() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#SensorReading".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "SensorReading".to_string());

    let char_mandatory = Characteristic::new(
        "urn:samm:org.example:1.0.0#ValueChar".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:metre".to_string(),
        },
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());
    let prop_mandatory = Property::new("urn:samm:org.example:1.0.0#value".to_string())
        .with_characteristic(char_mandatory);
    aspect.add_property(prop_mandatory);

    let char_optional = Characteristic::new(
        "urn:samm:org.example:1.0.0#LabelChar".to_string(),
        CharacteristicKind::Trait,
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
    let prop_optional = Property::new("urn:samm:org.example:1.0.0#label".to_string())
        .with_characteristic(char_optional)
        .as_optional();
    aspect.add_property(prop_optional);

    aspect
}

#[test]
fn test_openapi31_version_string() {
    let aspect = movement_aspect();
    let gen = v31_gen();
    let spec = gen
        .generate(&aspect)
        .expect("3.1 generation should succeed");
    assert_eq!(spec["openapi"], "3.1.0", "should emit openapi 3.1.0");
}

#[test]
fn test_openapi31_json_schema_dialect() {
    let aspect = movement_aspect();
    let gen = v31_gen();
    let spec = gen
        .generate(&aspect)
        .expect("3.1 generation should succeed");
    let dialect = spec["jsonSchemaDialect"].as_str().unwrap_or("");
    assert!(
        dialect.contains("json-schema.org") || dialect.contains("2020-12"),
        "jsonSchemaDialect should reference JSON Schema 2020-12, got: {}",
        dialect
    );
}

#[test]
fn test_openapi31_no_nullable_keyword() {
    let aspect = aspect_with_optional_property();
    let gen = v31_gen();
    let spec = gen
        .generate(&aspect)
        .expect("3.1 generation should succeed");
    let json = serde_json::to_string(&spec).expect("serialize");
    assert!(
        !json.contains("\"nullable\""),
        "3.1 document must not use the `nullable` keyword"
    );
}

#[test]
fn test_openapi31_type_array_for_optional() {
    let aspect = aspect_with_optional_property();
    let gen = v31_gen();
    let schemas = gen
        .build_schemas_v31(&aspect)
        .expect("build_schemas_v31 should succeed");

    let label_schema = &schemas["SensorReading"]["properties"]["label"];
    let type_val = &label_schema["type"];
    assert!(
        type_val.is_array(),
        "optional property type should be an array, got: {}",
        type_val
    );
    let empty = vec![];
    let types: Vec<&str> = type_val
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(
        types.contains(&"null"),
        "optional property type array should contain 'null', got: {:?}",
        types
    );
}

#[test]
fn test_openapi31_mandatory_property_not_nullable() {
    let aspect = aspect_with_optional_property();
    let gen = v31_gen();
    let schemas = gen
        .build_schemas_v31(&aspect)
        .expect("build_schemas_v31 should succeed");

    let value_schema = &schemas["SensorReading"]["properties"]["value"];
    let type_val = &value_schema["type"];
    if let Some(arr) = type_val.as_array() {
        let types: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(
            !types.contains(&"null"),
            "mandatory property should not include 'null' type, got: {:?}",
            types
        );
    }
}

#[test]
fn test_openapi31_const_for_single_value_enum() {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#ConstAspect".to_string());
    let char = Characteristic::new(
        "urn:samm:org.example:1.0.0#FixedVal".to_string(),
        CharacteristicKind::Enumeration {
            values: vec!["FIXED".to_string()],
        },
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
    let prop = Property::new("urn:samm:org.example:1.0.0#fixedField".to_string())
        .with_characteristic(char);
    aspect.add_property(prop);

    let gen = v31_gen();
    let schemas = gen
        .build_schemas_v31(&aspect)
        .expect("build_schemas_v31 should succeed");
    let field = &schemas["ConstAspect"]["properties"]["fixedField"];
    assert!(
        field["const"].is_string(),
        "single-value enum should use `const` in 3.1, got: {}",
        field
    );
    assert!(
        field.get("enum").is_none(),
        "single-value `const` should not also emit `enum`"
    );
}

#[test]
fn test_openapi31_multi_value_enum_uses_enum_keyword() {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#EnumAspect".to_string());
    let char = Characteristic::new(
        "urn:samm:org.example:1.0.0#StatusEnum".to_string(),
        CharacteristicKind::Enumeration {
            values: vec!["Active".to_string(), "Inactive".to_string()],
        },
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#string".to_string());
    let prop =
        Property::new("urn:samm:org.example:1.0.0#status".to_string()).with_characteristic(char);
    aspect.add_property(prop);

    let gen = v31_gen();
    let schemas = gen
        .build_schemas_v31(&aspect)
        .expect("build_schemas_v31 should succeed");
    let status = &schemas["EnumAspect"]["properties"]["status"];
    assert!(
        status["enum"].is_array(),
        "multi-value enum should still use `enum` keyword in 3.1"
    );
}

fn collection_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Readings".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Readings".to_string());
    aspect
        .metadata
        .add_description("en".to_string(), "A list of sensor readings".to_string());

    let inner_char = Characteristic::new(
        "urn:samm:org.example:1.0.0#ReadingValue".to_string(),
        CharacteristicKind::Trait,
    )
    .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());

    let coll_char = Characteristic::new(
        "urn:samm:org.example:1.0.0#ReadingList".to_string(),
        CharacteristicKind::Collection {
            element_characteristic: Some(Box::new(inner_char)),
        },
    );
    let prop = Property::new("urn:samm:org.example:1.0.0#readings".to_string())
        .with_characteristic(coll_char);
    aspect.add_property(prop);
    aspect
}

#[test]
fn test_pagination_collection_aspect_emits_extension() {
    let aspect = collection_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects")
        .with_pagination(PaginationConfig::default());
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    let schema = &path_item["get"]["responses"]["200"]["content"]["application/json"]["schema"];
    assert!(
        schema.get("x-samm-pagination").is_some(),
        "collection aspect GET response must include x-samm-pagination, got: {}",
        schema
    );
}

#[test]
fn test_pagination_non_collection_aspect_no_extension() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects")
        .with_pagination(PaginationConfig::default());
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    let schema = &path_item["get"]["responses"]["200"]["content"]["application/json"]["schema"];
    assert!(
        schema.get("x-samm-pagination").is_none(),
        "non-collection aspect must NOT include x-samm-pagination"
    );
}

#[test]
fn test_pagination_cursor_based_true() {
    let aspect = collection_aspect();
    let cfg = PaginationConfig {
        page_size: 50,
        cursor_based: true,
        total_count_header: None,
    };
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects").with_pagination(cfg);
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    let schema = &path_item["get"]["responses"]["200"]["content"]["application/json"]["schema"];
    let pag = schema
        .get("x-samm-pagination")
        .expect("x-samm-pagination must be present");
    assert_eq!(
        pag["cursorBased"],
        Value::Bool(true),
        "cursorBased should be true"
    );
}

#[test]
fn test_pagination_page_size_matches_config() {
    let aspect = collection_aspect();
    let cfg = PaginationConfig {
        page_size: 42,
        cursor_based: false,
        total_count_header: None,
    };
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects").with_pagination(cfg);
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    let schema = &path_item["get"]["responses"]["200"]["content"]["application/json"]["schema"];
    let pag = schema
        .get("x-samm-pagination")
        .expect("x-samm-pagination must be present");
    assert_eq!(
        pag["pageSize"].as_u64(),
        Some(42),
        "pageSize should equal the configured value"
    );
}

#[test]
fn test_pagination_block_nesting_level() {
    let aspect = collection_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects")
        .with_pagination(PaginationConfig::default());
    let spec = gen.generate(&aspect).expect("generation should succeed");
    let paths = spec["paths"].as_object().expect("paths");
    let path_item = paths.values().next().expect("path item");
    let schema_obj = &path_item["get"]["responses"]["200"]["content"]["application/json"]["schema"];
    assert!(
        schema_obj.get("$ref").is_some(),
        "schema should have $ref sibling"
    );
    assert!(
        schema_obj.get("x-samm-pagination").is_some(),
        "x-samm-pagination must be sibling of $ref"
    );
}

#[test]
fn test_pagination_builder_sets_config() {
    let cfg = PaginationConfig {
        page_size: 99,
        cursor_based: true,
        total_count_header: Some("X-Total-Count".to_string()),
    };
    let gen = OpenApiGenerator::new("1.0.0", "/api").with_pagination(cfg.clone());
    assert_eq!(
        gen.options
            .pagination
            .as_ref()
            .expect("pagination should be set"),
        &cfg,
        "with_pagination() must store the provided config"
    );
}

#[test]
fn test_openapi30_still_works() {
    let aspect = movement_aspect();
    let gen = OpenApiGenerator::new("1.0.0", "/api/v1/aspects");
    let spec = gen
        .generate(&aspect)
        .expect("3.0 generation should succeed");
    assert_eq!(
        spec["openapi"], "3.0.3",
        "default generator should still emit 3.0.3"
    );
    assert!(
        spec.get("jsonSchemaDialect").is_none(),
        "3.0 document should not have jsonSchemaDialect field"
    );
    let schemas = gen.build_schemas(&aspect).expect("build_schemas");
    assert_eq!(
        schemas["Movement"]["properties"]["speed"]["type"], "number",
        "3.0 schema type should be a plain string"
    );
}

#[test]
fn test_openapi31_ref_optional_uses_one_of() {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#RefAspect".to_string());
    let char = Characteristic::new(
        "urn:samm:org.example:1.0.0#EntityRef".to_string(),
        CharacteristicKind::SingleEntity {
            entity_type: "urn:samm:org.example:1.0.0#MyEntity".to_string(),
        },
    );
    let prop = Property::new("urn:samm:org.example:1.0.0#entity".to_string())
        .with_characteristic(char)
        .as_optional();
    aspect.add_property(prop);

    let gen = v31_gen();
    let schemas = gen
        .build_schemas_v31(&aspect)
        .expect("build_schemas_v31 should succeed");
    let entity_prop = &schemas["RefAspect"]["properties"]["entity"];
    assert!(
        entity_prop["oneOf"].is_array(),
        "optional $ref property should be wrapped with oneOf, got: {}",
        entity_prop
    );
}
