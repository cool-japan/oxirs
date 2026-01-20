//! Round-trip integration tests for DTDL conversion
//!
//! Tests that verify SAMM → DTDL → SAMM conversion maintains model integrity.

use oxirs_samm::dtdl_parser::parse_dtdl_interface;
use oxirs_samm::generators::dtdl::generate_dtdl;
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Event, ModelElement, Operation, Property,
};

#[test]
fn test_roundtrip_simple_aspect() {
    // Create a simple SAMM Aspect
    let mut original = Aspect::new("urn:samm:com.example:1.0.0#Movement".to_string());
    original
        .metadata
        .add_preferred_name("en".to_string(), "Movement".to_string());
    original
        .metadata
        .add_description("en".to_string(), "Vehicle movement tracking".to_string());

    // Convert to DTDL
    let dtdl = generate_dtdl(&original).expect("DTDL generation failed");

    // Parse back to SAMM
    let parsed = parse_dtdl_interface(&dtdl).expect("DTDL parsing failed");

    // Verify round-trip
    assert_eq!(parsed.name(), original.name());
    assert_eq!(
        parsed.metadata.get_preferred_name("en"),
        original.metadata.get_preferred_name("en")
    );
    assert_eq!(
        parsed.metadata.get_description("en"),
        original.metadata.get_description("en")
    );
}

#[test]
fn test_roundtrip_aspect_with_properties() {
    // Create SAMM Aspect with properties
    let mut original = Aspect::new("urn:samm:com.example:1.0.0#Sensor".to_string());
    original
        .metadata
        .add_preferred_name("en".to_string(), "Sensor".to_string());

    // Add temperature property
    let mut temp = Property::new("urn:samm:com.example:1.0.0#temperature".to_string());
    temp.metadata
        .add_preferred_name("en".to_string(), "temperature".to_string());
    temp.metadata
        .add_description("en".to_string(), "Temperature reading".to_string());
    let mut temp_char = Characteristic::new(
        "urn:samm:com.example:1.0.0#TempChar".to_string(),
        CharacteristicKind::Trait,
    );
    temp_char.data_type = Some("xsd:double".to_string());
    temp.characteristic = Some(temp_char);
    original.add_property(temp);

    // Add optional humidity property
    let mut humidity = Property::new("urn:samm:com.example:1.0.0#humidity".to_string());
    humidity.optional = true;
    let mut hum_char = Characteristic::new(
        "urn:samm:com.example:1.0.0#HumChar".to_string(),
        CharacteristicKind::Trait,
    );
    hum_char.data_type = Some("xsd:float".to_string());
    humidity.characteristic = Some(hum_char);
    original.add_property(humidity);

    // Convert to DTDL
    let dtdl = generate_dtdl(&original).expect("DTDL generation failed");

    // Parse back to SAMM
    let parsed = parse_dtdl_interface(&dtdl).expect("DTDL parsing failed");

    // Verify round-trip
    assert_eq!(parsed.name(), original.name());
    assert_eq!(parsed.properties().len(), original.properties().len());

    // Verify properties
    let parsed_props: std::collections::HashMap<_, _> =
        parsed.properties().iter().map(|p| (p.name(), p)).collect();

    assert!(parsed_props.contains_key("temperature"));
    assert!(parsed_props.contains_key("humidity"));

    // Check data types
    let temp_prop = parsed_props.get("temperature").unwrap();
    assert_eq!(
        temp_prop
            .characteristic
            .as_ref()
            .and_then(|c| c.data_type.as_ref()),
        Some(&"xsd:double".to_string())
    );

    let hum_prop = parsed_props.get("humidity").unwrap();
    assert!(hum_prop.optional);
}

#[test]
fn test_roundtrip_aspect_with_operations() {
    // Create SAMM Aspect with operations
    let mut original = Aspect::new("urn:samm:com.example:1.0.0#Control".to_string());

    let mut reset = Operation::new("urn:samm:com.example:1.0.0#reset".to_string());
    reset
        .metadata
        .add_preferred_name("en".to_string(), "reset".to_string());
    reset
        .metadata
        .add_description("en".to_string(), "Reset to defaults".to_string());
    original.add_operation(reset);

    let mut shutdown = Operation::new("urn:samm:com.example:1.0.0#shutdown".to_string());
    shutdown
        .metadata
        .add_preferred_name("en".to_string(), "shutdown".to_string());
    original.add_operation(shutdown);

    // Convert to DTDL
    let dtdl = generate_dtdl(&original).expect("DTDL generation failed");

    // Parse back to SAMM
    let parsed = parse_dtdl_interface(&dtdl).expect("DTDL parsing failed");

    // Verify round-trip
    assert_eq!(parsed.name(), original.name());
    assert_eq!(parsed.operations().len(), original.operations().len());

    let parsed_ops: std::collections::HashMap<_, _> =
        parsed.operations().iter().map(|o| (o.name(), o)).collect();

    assert!(parsed_ops.contains_key("reset"));
    assert!(parsed_ops.contains_key("shutdown"));
}

#[test]
fn test_roundtrip_aspect_with_events() {
    // Create SAMM Aspect with events
    let mut original = Aspect::new("urn:samm:com.example:1.0.0#Alerts".to_string());

    let mut alert = Event::new("urn:samm:com.example:1.0.0#temperatureAlert".to_string());
    alert
        .metadata
        .add_preferred_name("en".to_string(), "temperatureAlert".to_string());
    alert
        .metadata
        .add_description("en".to_string(), "Alert on high temperature".to_string());
    original.add_event(alert);

    // Convert to DTDL (events become Telemetry)
    let dtdl = generate_dtdl(&original).expect("DTDL generation failed");

    // Parse back to SAMM
    // Note: DTDL Telemetry for events will be parsed as optional properties
    // This is expected behavior as DTDL doesn't distinguish between events and telemetry
    let parsed = parse_dtdl_interface(&dtdl).expect("DTDL parsing failed");

    assert_eq!(parsed.name(), original.name());
    // Events in DTDL are represented as Telemetry, which parses back as properties
    // This is expected lossy conversion
}

#[test]
fn test_roundtrip_complex_aspect() {
    // Create complex SAMM Aspect
    let mut original = Aspect::new("urn:samm:com.example.vehicle:1.0.0#Vehicle".to_string());
    original
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle".to_string());
    original
        .metadata
        .add_description("en".to_string(), "Complete vehicle model".to_string());

    // Add properties of various types
    let mut speed = Property::new("urn:samm:com.example.vehicle:1.0.0#speed".to_string());
    let mut speed_char = Characteristic::new(
        "urn:samm:com.example.vehicle:1.0.0#SpeedChar".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:kilometrePerHour".to_string(),
        },
    );
    speed_char.data_type = Some("xsd:float".to_string());
    speed.characteristic = Some(speed_char);
    original.add_property(speed);

    let mut is_moving = Property::new("urn:samm:com.example.vehicle:1.0.0#isMoving".to_string());
    let mut bool_char = Characteristic::new(
        "urn:samm:com.example.vehicle:1.0.0#BoolChar".to_string(),
        CharacteristicKind::Trait,
    );
    bool_char.data_type = Some("xsd:boolean".to_string());
    is_moving.characteristic = Some(bool_char);
    original.add_property(is_moving);

    // Add operation
    let mut stop = Operation::new("urn:samm:com.example.vehicle:1.0.0#emergencyStop".to_string());
    stop.metadata
        .add_preferred_name("en".to_string(), "emergencyStop".to_string());
    original.add_operation(stop);

    // Convert to DTDL
    let dtdl = generate_dtdl(&original).expect("DTDL generation failed");

    // Parse back to SAMM
    let parsed = parse_dtdl_interface(&dtdl).expect("DTDL parsing failed");

    // Verify round-trip
    assert_eq!(parsed.name(), original.name());
    assert_eq!(parsed.properties().len(), original.properties().len());
    assert_eq!(parsed.operations().len(), original.operations().len());
}

#[test]
fn test_dtdl_json_validity() {
    // Create SAMM Aspect
    let mut aspect = Aspect::new("urn:samm:io.test:1.0.0#TestAspect".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Test Aspect".to_string());

    let mut prop = Property::new("urn:samm:io.test:1.0.0#testProp".to_string());
    let mut char = Characteristic::new(
        "urn:samm:io.test:1.0.0#TestChar".to_string(),
        CharacteristicKind::Trait,
    );
    char.data_type = Some("xsd:int".to_string());
    prop.characteristic = Some(char);
    aspect.add_property(prop);

    // Generate DTDL
    let dtdl = generate_dtdl(&aspect).expect("DTDL generation failed");

    // Verify it's valid JSON
    let _: serde_json::Value =
        serde_json::from_str(&dtdl).expect("Generated DTDL is not valid JSON");

    // Verify it can be parsed back
    let parsed = parse_dtdl_interface(&dtdl).expect("DTDL parsing failed");
    assert_eq!(parsed.name(), aspect.name());
}
