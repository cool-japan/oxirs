//! DTDL Generation Example
//!
//! Demonstrates how to generate DTDL (Digital Twins Definition Language) from SAMM Aspect models.
//! DTDL is used by Azure Digital Twins for modeling IoT devices and digital twins.
//!
//! # Run this example
//!
//! ```bash
//! cargo run --example dtdl_generation
//! ```

use oxirs_samm::generators::dtdl::{generate_dtdl, generate_dtdl_with_options, DtdlOptions};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Event, Operation, Property,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DTDL Generation Example ===\n");

    // Example 1: Create a simple vehicle movement aspect
    let movement_aspect = create_movement_aspect();

    println!("Example 1: Basic DTDL Generation");
    println!("----------------------------------");
    let dtdl = generate_dtdl(&movement_aspect)?;
    println!("{}\n", dtdl);

    // Example 2: Generate with custom options (compact)
    println!("Example 2: Compact DTDL Output");
    println!("-------------------------------");
    let compact_options = DtdlOptions {
        compact: true,
        ..Default::default()
    };
    let compact_dtdl = generate_dtdl_with_options(&movement_aspect, compact_options)?;
    println!("{}\n", compact_dtdl);

    // Example 3: Generate without descriptions
    println!("Example 3: DTDL Without Descriptions");
    println!("-------------------------------------");
    let no_desc_options = DtdlOptions {
        include_descriptions: false,
        ..Default::default()
    };
    let no_desc_dtdl = generate_dtdl_with_options(&movement_aspect, no_desc_options)?;
    println!("{}\n", no_desc_dtdl);

    // Example 4: IoT sensor aspect
    println!("Example 4: IoT Sensor DTDL");
    println!("---------------------------");
    let sensor_aspect = create_sensor_aspect();
    let sensor_dtdl = generate_dtdl(&sensor_aspect)?;
    println!("{}\n", sensor_dtdl);

    println!("✓ All examples completed successfully!");
    println!("\nUse these DTDL files with Azure Digital Twins:");
    println!("  az dt model create --from-file <model.json>");

    Ok(())
}

/// Create a vehicle movement aspect
fn create_movement_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:com.example.vehicle:1.0.0#Movement".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Movement".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "Represents the movement characteristics of a vehicle".to_string(),
    );

    // Add speed property
    let mut speed = Property::new("urn:samm:com.example.vehicle:1.0.0#speed".to_string());
    speed
        .metadata
        .add_preferred_name("en".to_string(), "speed".to_string());
    speed
        .metadata
        .add_description("en".to_string(), "Current speed in km/h".to_string());
    let mut speed_char = Characteristic::new(
        "urn:samm:com.example.vehicle:1.0.0#SpeedCharacteristic".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:kilometrePerHour".to_string(),
        },
    );
    speed_char.data_type = Some("xsd:float".to_string());
    speed.characteristic = Some(speed_char);
    aspect.add_property(speed);

    // Add position property
    let mut position = Property::new("urn:samm:com.example.vehicle:1.0.0#position".to_string());
    position
        .metadata
        .add_preferred_name("en".to_string(), "position".to_string());
    position
        .metadata
        .add_description("en".to_string(), "GPS coordinates".to_string());
    let mut position_char = Characteristic::new(
        "urn:samm:com.example.vehicle:1.0.0#PositionCharacteristic".to_string(),
        CharacteristicKind::Trait,
    );
    position_char.data_type = Some("xsd:string".to_string());
    position.characteristic = Some(position_char);
    aspect.add_property(position);

    // Add isMoving property
    let mut is_moving = Property::new("urn:samm:com.example.vehicle:1.0.0#isMoving".to_string());
    is_moving
        .metadata
        .add_preferred_name("en".to_string(), "isMoving".to_string());
    is_moving.metadata.add_description(
        "en".to_string(),
        "Whether the vehicle is currently moving".to_string(),
    );
    let mut moving_char = Characteristic::new(
        "urn:samm:com.example.vehicle:1.0.0#MovingCharacteristic".to_string(),
        CharacteristicKind::Trait,
    );
    moving_char.data_type = Some("xsd:boolean".to_string());
    is_moving.characteristic = Some(moving_char);
    aspect.add_property(is_moving);

    // Add emergency stop operation
    let mut emergency_stop =
        Operation::new("urn:samm:com.example.vehicle:1.0.0#emergencyStop".to_string());
    emergency_stop
        .metadata
        .add_preferred_name("en".to_string(), "emergencyStop".to_string());
    emergency_stop
        .metadata
        .add_description("en".to_string(), "Emergency stop command".to_string());
    aspect.add_operation(emergency_stop);

    // Add speed limit exceeded event
    let mut speed_event =
        Event::new("urn:samm:com.example.vehicle:1.0.0#speedLimitExceeded".to_string());
    speed_event
        .metadata
        .add_preferred_name("en".to_string(), "speedLimitExceeded".to_string());
    speed_event.metadata.add_description(
        "en".to_string(),
        "Event emitted when speed limit is exceeded".to_string(),
    );
    aspect.add_event(speed_event);

    aspect
}

/// Create an IoT sensor aspect
fn create_sensor_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:io.oxirs.sensor:1.0.0#TemperatureSensor".to_string());
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Temperature Sensor".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "IoT temperature sensor with alert capabilities".to_string(),
    );

    // Temperature property
    let mut temp = Property::new("urn:samm:io.oxirs.sensor:1.0.0#temperature".to_string());
    temp.metadata
        .add_preferred_name("en".to_string(), "temperature".to_string());
    temp.metadata.add_description(
        "en".to_string(),
        "Current temperature reading in Celsius".to_string(),
    );
    let mut temp_char = Characteristic::new(
        "urn:samm:io.oxirs.sensor:1.0.0#TempCharacteristic".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:degreeCelsius".to_string(),
        },
    );
    temp_char.data_type = Some("xsd:double".to_string());
    temp.characteristic = Some(temp_char);
    aspect.add_property(temp);

    // Humidity property
    let mut humidity = Property::new("urn:samm:io.oxirs.sensor:1.0.0#humidity".to_string());
    humidity
        .metadata
        .add_preferred_name("en".to_string(), "humidity".to_string());
    humidity
        .metadata
        .add_description("en".to_string(), "Relative humidity percentage".to_string());
    let mut humidity_char = Characteristic::new(
        "urn:samm:io.oxirs.sensor:1.0.0#HumidityCharacteristic".to_string(),
        CharacteristicKind::Trait,
    );
    humidity_char.data_type = Some("xsd:float".to_string());
    humidity.characteristic = Some(humidity_char);
    aspect.add_property(humidity);

    // Reset command
    let mut reset = Operation::new("urn:samm:io.oxirs.sensor:1.0.0#reset".to_string());
    reset
        .metadata
        .add_preferred_name("en".to_string(), "reset".to_string());
    reset
        .metadata
        .add_description("en".to_string(), "Reset sensor to defaults".to_string());
    aspect.add_operation(reset);

    // Alert event
    let mut alert = Event::new("urn:samm:io.oxirs.sensor:1.0.0#temperatureAlert".to_string());
    alert
        .metadata
        .add_preferred_name("en".to_string(), "temperatureAlert".to_string());
    alert.metadata.add_description(
        "en".to_string(),
        "Alert when temperature exceeds threshold".to_string(),
    );
    aspect.add_event(alert);

    aspect
}
