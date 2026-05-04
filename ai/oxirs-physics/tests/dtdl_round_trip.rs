//! DTDL v3 round-trip and integration tests
//!
//! Tests cover: parsing fixtures → validation → RDF mapping → spot-checks.
//! Each fixture exercises a different subset of DTDL v3 features.

use oxirs_physics::dtdl::{
    interface_to_rdf, is_valid, parse_dtdl_interface, validate, DtdlContent, DtdlValidationError,
    Dtmi, RDF_TYPE,
};

const THERMOSTAT_JSON: &str = include_str!("fixtures/dtdl/thermostat.json");
const ENERGY_METER_JSON: &str = include_str!("fixtures/dtdl/energy_meter.json");
const BUILDING_JSON: &str = include_str!("fixtures/dtdl/building_complex.json");

// ─────────────────────────────────────────────────────────────────────────────
// Parsing — thermostat fixture
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parse_thermostat_dtdl() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("should parse thermostat DTDL");
    assert_eq!(iface.id.0, "dtmi:example:Thermostat;1");
    assert_eq!(
        oxirs_physics::dtdl::primary_type(&iface.element_type),
        Some("Interface")
    );
    let contents = iface.contents.as_ref().expect("should have contents");
    assert_eq!(contents.len(), 3, "Thermostat should have 3 content items");
}

#[test]
fn thermostat_has_telemetry_property_command() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let contents = iface.contents.as_ref().expect("contents");

    let telemetry_count = contents
        .iter()
        .filter(|c| matches!(c, DtdlContent::Telemetry(_)))
        .count();
    let property_count = contents
        .iter()
        .filter(|c| matches!(c, DtdlContent::Property(_)))
        .count();
    let command_count = contents
        .iter()
        .filter(|c| matches!(c, DtdlContent::Command(_)))
        .count();

    assert_eq!(telemetry_count, 1, "one telemetry element");
    assert_eq!(property_count, 1, "one property element");
    assert_eq!(command_count, 1, "one command element");
}

#[test]
fn thermostat_telemetry_schema_and_unit() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let contents = iface.contents.as_ref().expect("contents");
    if let DtdlContent::Telemetry(t) = &contents[0] {
        assert_eq!(t.name, "temperature");
        assert_eq!(t.schema.0, "double");
        assert_eq!(t.unit.as_deref(), Some("Celsius"));
    } else {
        panic!("expected Telemetry as first content");
    }
}

#[test]
fn thermostat_property_writable() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let contents = iface.contents.as_ref().expect("contents");
    if let DtdlContent::Property(p) = &contents[1] {
        assert_eq!(p.name, "targetTemperature");
        assert_eq!(p.writable, Some(true));
    } else {
        panic!("expected Property as second content");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsing — energy meter fixture
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parse_energy_meter_dtdl() {
    let iface = parse_dtdl_interface(ENERGY_METER_JSON).expect("should parse energy meter DTDL");
    assert_eq!(iface.id.0, "dtmi:example:EnergyMeter;2");
    assert_eq!(iface.id.version(), Some(2));
}

#[test]
fn energy_meter_has_two_telemetry_one_property() {
    let iface = parse_dtdl_interface(ENERGY_METER_JSON).expect("parse");
    let contents = iface.contents.as_ref().expect("contents");

    let telemetry_count = contents
        .iter()
        .filter(|c| matches!(c, DtdlContent::Telemetry(_)))
        .count();
    let property_count = contents
        .iter()
        .filter(|c| matches!(c, DtdlContent::Property(_)))
        .count();

    assert_eq!(telemetry_count, 2);
    assert_eq!(property_count, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsing — building fixture (all 5 element types)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parse_building_all_element_types() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse building");
    let contents = iface.contents.as_ref().expect("contents");

    let has_telemetry = contents
        .iter()
        .any(|c| matches!(c, DtdlContent::Telemetry(_)));
    let has_property = contents
        .iter()
        .any(|c| matches!(c, DtdlContent::Property(_)));
    let has_command = contents
        .iter()
        .any(|c| matches!(c, DtdlContent::Command(_)));
    let has_component = contents
        .iter()
        .any(|c| matches!(c, DtdlContent::Component(_)));
    let has_relationship = contents
        .iter()
        .any(|c| matches!(c, DtdlContent::Relationship(_)));

    assert!(has_telemetry, "no Telemetry element");
    assert!(has_property, "no Property element");
    assert!(has_command, "no Command element");
    assert!(has_component, "no Component element");
    assert!(has_relationship, "no Relationship element");
}

#[test]
fn building_language_map_display_name() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse");
    // displayName is { "en": "Building", "de": "Gebäude" }
    let dn = iface.display_name.expect("display_name");
    assert!(dn.is_object(), "display_name should be a language map");
}

#[test]
fn building_component_schema_is_dtmi() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse");
    let contents = iface.contents.as_ref().expect("contents");
    let component = contents
        .iter()
        .find(|c| matches!(c, DtdlContent::Component(_)))
        .expect("component");
    if let DtdlContent::Component(comp) = component {
        assert_eq!(comp.schema.0, "dtmi:example:HvacUnit;1");
        assert!(
            comp.schema.validate().is_ok(),
            "component schema DTMI valid"
        );
    }
}

#[test]
fn building_relationship_has_target() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse");
    let contents = iface.contents.as_ref().expect("contents");
    let rel = contents
        .iter()
        .find(|c| matches!(c, DtdlContent::Relationship(_)))
        .expect("relationship");
    if let DtdlContent::Relationship(r) = rel {
        let target = r.target.as_ref().expect("target");
        assert_eq!(target.0, "dtmi:example:Floor;1");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DTMI validation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dtmi_validation_valid() {
    let valid = Dtmi("dtmi:example:Foo;1".into());
    assert!(valid.validate().is_ok());
}

#[test]
fn dtmi_validation_multi_segment_valid() {
    let d = Dtmi("dtmi:com:example:devices:Thermostat;3".into());
    assert!(d.validate().is_ok());
}

#[test]
fn dtmi_validation_missing_version() {
    let invalid = Dtmi("dtmi:example:Foo".into());
    assert!(invalid.validate().is_err());
}

#[test]
fn dtmi_validation_wrong_prefix() {
    let invalid = Dtmi("http://example.org/Foo;1".into());
    assert!(invalid.validate().is_err());
}

#[test]
fn dtmi_validation_non_integer_version() {
    let invalid = Dtmi("dtmi:example:Foo;abc".into());
    assert!(invalid.validate().is_err());
}

#[test]
fn dtmi_version_accessor() {
    let d = Dtmi("dtmi:example:EnergyMeter;2".into());
    assert_eq!(d.version(), Some(2));
}

#[test]
fn dtmi_path_accessor() {
    let d = Dtmi("dtmi:example:Thermostat;1".into());
    assert_eq!(d.path(), Some("example:Thermostat"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Semantic validation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn validate_thermostat_no_errors() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let errors = validate(&iface);
    assert!(
        errors.is_empty(),
        "thermostat should be valid; errors: {errors:?}"
    );
}

#[test]
fn validate_energy_meter_no_errors() {
    let iface = parse_dtdl_interface(ENERGY_METER_JSON).expect("parse");
    let errors = validate(&iface);
    assert!(
        errors.is_empty(),
        "energy meter should be valid; errors: {errors:?}"
    );
}

#[test]
fn validate_building_no_errors() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse");
    let errors = validate(&iface);
    assert!(
        errors.is_empty(),
        "building should be valid; errors: {errors:?}"
    );
}

#[test]
fn validate_invalid_dtmi_detected() {
    use oxirs_physics::dtdl::DtdlInterface;
    let iface = DtdlInterface {
        context: serde_json::json!("dtmi:dtdl:context;3"),
        element_type: serde_json::json!("Interface"),
        id: Dtmi("bad-id".into()),
        display_name: None,
        description: None,
        comment: None,
        contents: None,
        schemas: None,
        extends: None,
    };
    let errors = validate(&iface);
    assert!(!errors.is_empty(), "bad DTMI should produce errors");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, DtdlValidationError::InvalidDtmi { .. })),
        "expected InvalidDtmi error"
    );
}

#[test]
fn is_valid_true_for_thermostat() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    assert!(is_valid(&iface));
}

// ─────────────────────────────────────────────────────────────────────────────
// RDF mapping — thermostat
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn thermostat_to_rdf_non_empty() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    assert!(!triples.is_empty(), "should produce at least one triple");
}

#[test]
fn thermostat_rdf_has_interface_type() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);

    let has_type = triples
        .iter()
        .any(|t| t.subject == "dtmi:example:Thermostat;1" && t.predicate == RDF_TYPE);
    assert!(has_type, "Interface should have rdf:type triple");
}

#[test]
fn thermostat_rdf_has_label() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    let has_label = triples
        .iter()
        .any(|t| t.predicate.contains("label") && t.object.contains("Thermostat"));
    assert!(has_label, "should have rdfs:label triple");
}

#[test]
fn thermostat_rdf_has_telemetry_link() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    assert!(
        triples.iter().any(|t| t.predicate.contains("hasTelemetry")),
        "should have hasTelemetry triple"
    );
}

#[test]
fn thermostat_rdf_celsius_maps_to_qudt_deg_c() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    let celsius_triple = triples.iter().find(|t| t.object.contains("DEG_C"));
    assert!(
        celsius_triple.is_some(),
        "Celsius unit should map to QUDT DEG_C; triples: {triples:?}"
    );
}

#[test]
fn thermostat_rdf_property_writable_triple() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    assert!(
        triples.iter().any(|t| t.predicate.contains("writable")),
        "writable property should produce a triple"
    );
}

#[test]
fn thermostat_rdf_command_link() {
    let iface = parse_dtdl_interface(THERMOSTAT_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    assert!(
        triples.iter().any(|t| t.predicate.contains("hasCommand")),
        "should have hasCommand triple"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// RDF mapping — energy meter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn energy_meter_rdf_power_unit_qudt_watt() {
    let iface = parse_dtdl_interface(ENERGY_METER_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    let watt_triple = triples.iter().find(|t| t.object.contains("/W"));
    assert!(watt_triple.is_some(), "watt unit should map to QUDT");
}

#[test]
fn energy_meter_rdf_kwh_unit_mapped() {
    let iface = parse_dtdl_interface(ENERGY_METER_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    let kwh_triple = triples
        .iter()
        .find(|t| t.object.contains("KiloW-HR") || t.object.contains("kilowattHour"));
    assert!(kwh_triple.is_some(), "kilowattHour should be mapped");
}

// ─────────────────────────────────────────────────────────────────────────────
// RDF mapping — building (all element types)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn building_rdf_all_element_types_present() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);

    assert!(triples.iter().any(|t| t.predicate.contains("hasTelemetry")));
    assert!(triples.iter().any(|t| t.predicate.contains("hasProperty")));
    assert!(triples.iter().any(|t| t.predicate.contains("hasCommand")));
    assert!(triples.iter().any(|t| t.predicate.contains("hasComponent")));
    assert!(triples
        .iter()
        .any(|t| t.predicate.contains("hasRelationship")));
}

#[test]
fn building_rdf_component_schema_is_dtmi() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);

    // Component schema triple: object should be a DTMI (not a literal)
    let schema_triple = triples
        .iter()
        .find(|t| t.predicate.contains("schema") && t.object.starts_with("dtmi:"));
    assert!(
        schema_triple.is_some(),
        "component schema triple should have DTMI object"
    );
}

#[test]
fn building_rdf_relationship_target() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    assert!(triples
        .iter()
        .any(|t| t.predicate.contains("target") && t.object == "dtmi:example:Floor;1"));
}

#[test]
fn building_rdf_semantic_type_humidity_annotated() {
    let iface = parse_dtdl_interface(BUILDING_JSON).expect("parse");
    let triples = interface_to_rdf(&iface);
    // "relativeHumidity" telemetry has @type ["Telemetry","Humidity"]
    let sem_triple = triples
        .iter()
        .find(|t| t.predicate.contains("semanticType") && t.object.contains("Humidity"));
    assert!(
        sem_triple.is_some(),
        "Humidity semantic type annotation should be a triple"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// QUDT unit mapping table
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn qudt_mapping_celsius() {
    use oxirs_physics::dtdl::dtdl_unit_to_qudt;
    assert_eq!(
        dtdl_unit_to_qudt("Celsius"),
        Some("http://qudt.org/vocab/unit/DEG_C")
    );
}

#[test]
fn qudt_mapping_kelvin() {
    use oxirs_physics::dtdl::dtdl_unit_to_qudt;
    assert_eq!(
        dtdl_unit_to_qudt("Kelvin"),
        Some("http://qudt.org/vocab/unit/K")
    );
}

#[test]
fn qudt_mapping_watt() {
    use oxirs_physics::dtdl::dtdl_unit_to_qudt;
    assert_eq!(
        dtdl_unit_to_qudt("watt"),
        Some("http://qudt.org/vocab/unit/W")
    );
}

#[test]
fn qudt_mapping_pascal() {
    use oxirs_physics::dtdl::dtdl_unit_to_qudt;
    assert_eq!(
        dtdl_unit_to_qudt("pascal"),
        Some("http://qudt.org/vocab/unit/PA")
    );
}

#[test]
fn qudt_mapping_kilogram() {
    use oxirs_physics::dtdl::dtdl_unit_to_qudt;
    assert_eq!(
        dtdl_unit_to_qudt("kilogram"),
        Some("http://qudt.org/vocab/unit/KiloGM")
    );
}

#[test]
fn qudt_mapping_unknown_none() {
    use oxirs_physics::dtdl::dtdl_unit_to_qudt;
    assert!(dtdl_unit_to_qudt("furlongPerFortnight").is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// Error handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parse_invalid_json_errors() {
    let result = parse_dtdl_interface("{ not valid }");
    assert!(result.is_err());
}

#[test]
fn parse_missing_id_errors() {
    let json = r#"{"@type": "Interface"}"#;
    let result = parse_dtdl_interface(json);
    assert!(result.is_err());
}

#[test]
fn parse_wrong_type_errors() {
    let json = r#"{"@type": "Telemetry", "@id": "dtmi:x:y;1", "name": "x", "schema": "double"}"#;
    let result = parse_dtdl_interface(json);
    assert!(result.is_err());
}

#[test]
fn parse_v2_context_accepted() {
    // v2 documents should parse without error
    let json = r#"{
        "@context": "dtmi:dtdl:context;2",
        "@type": "Interface",
        "@id": "dtmi:com:example:Thermostat;1",
        "displayName": "Thermostat"
    }"#;
    let iface = parse_dtdl_interface(json).expect("v2 context should be accepted");
    assert_eq!(iface.id.version(), Some(1));
}
