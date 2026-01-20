//! End-to-End CAN to RDF Integration Example
//!
//! Demonstrates the complete pipeline from CAN bus frames to RDF triples,
//! including:
//!
//! - DBC file parsing for signal definitions
//! - CAN frame decoding using DBC signals
//! - RDF triple generation with W3C PROV-O provenance
//! - QUDT unit mappings for semantic measurements
//!
//! # Architecture
//!
//! ```text
//! CAN Frame → DBC Decoder → Signal Values → RDF Mapper → RDF Triples
//!                                               ↓
//!                                        PROV-O Provenance
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example rdf_integration
//! ```

use oxirs_canbus::{
    parse_dbc, AutomotiveUnits, CanFrame, CanId, CanRdfMapper, CanbusResult, DbcDatabase,
    RdfMappingConfig,
};
use oxirs_core::model::{Object, Predicate, Subject};

fn main() -> CanbusResult<()> {
    println!("=== OxiRS CAN → RDF Integration Example ===\n");

    // Parse DBC file
    let db = create_vehicle_dbc()?;
    println!("DBC Database loaded:");
    println!("  Messages: {}", db.messages.len());
    println!(
        "  Signals: {}",
        db.messages.iter().map(|m| m.signals.len()).sum::<usize>()
    );
    println!();

    // Configure RDF mapping
    let config = RdfMappingConfig {
        device_id: "vehicle_001".to_string(),
        base_iri: "http://automotive.example.com/vehicle".to_string(),
        graph_iri: "urn:automotive:can-data".to_string(),
    };

    println!("RDF Configuration:");
    println!("  Device ID: {}", config.device_id);
    println!("  Base IRI: {}", config.base_iri);
    println!("  Graph IRI: {}", config.graph_iri);
    println!();

    // Create RDF mapper with deadband filtering
    let mut mapper = CanRdfMapper::new(db.clone(), config).with_deadband(0.01);

    // Simulate CAN frames and generate RDF
    println!("Processing CAN Frames and Generating RDF:");
    println!("-----------------------------------------\n");

    let frames = create_simulated_frames()?;

    for (name, frame) in frames {
        println!("--- {} (CAN ID: 0x{:03X}) ---", name, frame.id.as_raw());

        match mapper.map_frame(&frame) {
            Ok(triples) => {
                if triples.is_empty() {
                    println!("  (No significant changes - filtered by deadband)\n");
                    continue;
                }

                for generated in &triples {
                    println!("  Signal: {}", generated.signal_name);
                    println!("  Message: {}", generated.message_name);

                    // Display the main triple
                    println!("  Triple:");
                    println!(
                        "    Subject:   <{}>",
                        format_subject(generated.triple.subject())
                    );
                    println!(
                        "    Predicate: <{}>",
                        format_predicate(generated.triple.predicate())
                    );
                    println!(
                        "    Object:    {}",
                        format_object(generated.triple.object())
                    );

                    // Display unit if available
                    if let Some(unit) = &generated.unit {
                        println!("  Unit: {}", unit);
                    }

                    // Display provenance triples
                    let prov_triples = generated.provenance_triples();
                    if !prov_triples.is_empty() {
                        println!("  Provenance ({} triples):", prov_triples.len());
                        for prov in &prov_triples {
                            println!(
                                "    {} {} {}",
                                format_subject_short(prov.subject()),
                                format_predicate_short(prov.predicate()),
                                format_object_short(prov.object())
                            );
                        }
                    }
                    println!();
                }
            }
            Err(e) => {
                println!("  Error: {}\n", e);
            }
        }
    }

    // Display mapper statistics
    print_mapper_statistics(&mapper);

    // Demonstrate QUDT unit mappings
    print_qudt_units();

    // Show complete Turtle output
    print_turtle_example(&db)?;

    println!("\n=== RDF Integration Complete ===");

    Ok(())
}

fn create_vehicle_dbc() -> CanbusResult<DbcDatabase> {
    let dbc_content = r#"
VERSION ""

NS_ :

BS_:

BU_: Engine Dashboard

BO_ 256 EngineData: 8 Engine
 SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
 SG_ EngineLoad : 16|8@1+ (0.4,0) [0|100] "%" Dashboard
 SG_ CoolantTemp : 24|8@1+ (1,-40) [-40|215] "degC" Dashboard
 SG_ ThrottlePos : 32|8@1+ (0.4,0) [0|100] "%" Dashboard

BO_ 512 VehicleSpeed: 8 Engine
 SG_ WheelSpeed : 0|16@1+ (0.01,0) [0|655.35] "km/h" Dashboard
 SG_ Odometer : 16|24@1+ (0.1,0) [0|1677721.5] "km" Dashboard

BO_ 768 FuelSystem: 8 Engine
 SG_ FuelLevel : 0|8@1+ (0.5,0) [0|100] "%" Dashboard
 SG_ FuelRate : 8|16@1+ (0.05,0) [0|3276.75] "L/h" Dashboard
 SG_ FuelEconomy : 24|16@1+ (0.01,0) [0|655.35] "km/L" Dashboard

CM_ SG_ 256 EngineSpeed "Engine rotational speed";
CM_ SG_ 256 CoolantTemp "Engine coolant temperature";
CM_ SG_ 512 WheelSpeed "Vehicle speed from wheel sensors";
CM_ SG_ 768 FuelLevel "Fuel tank level percentage";
"#;

    parse_dbc(dbc_content)
}

fn create_simulated_frames() -> CanbusResult<Vec<(&'static str, CanFrame)>> {
    let mut frames = Vec::new();

    // EngineData: 2500 rpm, 60% load, 90 degC coolant, 40% throttle
    // EngineSpeed: 2500 rpm -> raw = 2500 / 0.125 = 20000 = 0x4E20
    // EngineLoad: 60% -> raw = 60 / 0.4 = 150 = 0x96
    // CoolantTemp: 90 degC -> raw = 90 + 40 = 130 = 0x82
    // ThrottlePos: 40% -> raw = 40 / 0.4 = 100 = 0x64
    let engine_data = CanFrame::new(
        CanId::standard(256)?,
        vec![0x20, 0x4E, 0x96, 0x82, 0x64, 0x00, 0x00, 0x00],
    )?;
    frames.push(("EngineData", engine_data));

    // VehicleSpeed: 85.5 km/h, 12345.6 km odometer
    // WheelSpeed: 85.5 km/h -> raw = 8550 = 0x2166
    // Odometer: 12345.6 km -> raw = 123456 = 0x01E240
    let vehicle_speed = CanFrame::new(
        CanId::standard(512)?,
        vec![0x66, 0x21, 0x40, 0xE2, 0x01, 0x00, 0x00, 0x00],
    )?;
    frames.push(("VehicleSpeed", vehicle_speed));

    // FuelSystem: 65% level, 4.5 L/h rate, 15.2 km/L economy
    // FuelLevel: 65% -> raw = 130 = 0x82
    // FuelRate: 4.5 L/h -> raw = 90 = 0x005A
    // FuelEconomy: 15.2 km/L -> raw = 1520 = 0x05F0
    let fuel_system = CanFrame::new(
        CanId::standard(768)?,
        vec![0x82, 0x5A, 0x00, 0xF0, 0x05, 0x00, 0x00, 0x00],
    )?;
    frames.push(("FuelSystem", fuel_system));

    // Send same EngineData again (should be filtered by deadband)
    let engine_data_repeat = CanFrame::new(
        CanId::standard(256)?,
        vec![0x20, 0x4E, 0x96, 0x82, 0x64, 0x00, 0x00, 0x00],
    )?;
    frames.push(("EngineData (repeat)", engine_data_repeat));

    // Send slightly different EngineData (2505 rpm - within deadband)
    // 2505 rpm -> raw = 20040 = 0x4E48
    let engine_data_small_change = CanFrame::new(
        CanId::standard(256)?,
        vec![0x48, 0x4E, 0x96, 0x82, 0x64, 0x00, 0x00, 0x00],
    )?;
    frames.push(("EngineData (small change)", engine_data_small_change));

    Ok(frames)
}

fn format_subject(subject: &Subject) -> String {
    match subject {
        Subject::NamedNode(n) => n.as_str().to_string(),
        Subject::BlankNode(b) => format!("_:{}", b.as_str()),
        Subject::Variable(v) => format!("?{}", v.as_str()),
        Subject::QuotedTriple(_) => "<<...>>".to_string(),
    }
}

fn format_predicate(predicate: &Predicate) -> String {
    match predicate {
        Predicate::NamedNode(n) => n.as_str().to_string(),
        Predicate::Variable(v) => format!("?{}", v.as_str()),
    }
}

fn format_object(object: &Object) -> String {
    match object {
        Object::NamedNode(n) => format!("<{}>", n.as_str()),
        Object::BlankNode(b) => format!("_:{}", b.as_str()),
        Object::Literal(lit) => format!("\"{}\"", lit.value()),
        Object::Variable(v) => format!("?{}", v.as_str()),
        Object::QuotedTriple(_) => "<<...>>".to_string(),
    }
}

fn format_subject_short(subject: &Subject) -> String {
    match subject {
        Subject::NamedNode(n) => {
            let iri = n.as_str();
            if let Some(suffix) = iri.strip_prefix("http://www.w3.org/ns/prov#") {
                format!("prov:{}", suffix)
            } else if let Some(suffix) = iri.strip_prefix("http://www.w3.org/2001/XMLSchema#") {
                format!("xsd:{}", suffix)
            } else if let Some(suffix) = iri.strip_prefix("http://automotive.example.com/vehicle/")
            {
                format!("vehicle:{}", suffix)
            } else {
                format!("<{}>", iri)
            }
        }
        Subject::BlankNode(b) => format!("_:{}", b.as_str()),
        Subject::Variable(v) => format!("?{}", v.as_str()),
        Subject::QuotedTriple(_) => "<<...>>".to_string(),
    }
}

fn format_predicate_short(predicate: &Predicate) -> String {
    match predicate {
        Predicate::NamedNode(n) => {
            let iri = n.as_str();
            if let Some(suffix) = iri.strip_prefix("http://www.w3.org/ns/prov#") {
                format!("prov:{}", suffix)
            } else if let Some(suffix) = iri.strip_prefix("http://www.w3.org/2001/XMLSchema#") {
                format!("xsd:{}", suffix)
            } else if let Some(suffix) = iri.strip_prefix("http://qudt.org/schema/qudt/") {
                format!("qudt:{}", suffix)
            } else {
                format!("<{}>", iri)
            }
        }
        Predicate::Variable(v) => format!("?{}", v.as_str()),
    }
}

fn format_object_short(object: &Object) -> String {
    match object {
        Object::NamedNode(n) => {
            let iri = n.as_str();
            if let Some(suffix) = iri.strip_prefix("http://qudt.org/vocab/unit/") {
                format!("unit:{}", suffix)
            } else {
                format!("<{}>", iri)
            }
        }
        Object::BlankNode(b) => format!("_:{}", b.as_str()),
        Object::Literal(lit) => {
            let value = lit.value();
            if value.len() > 30 {
                format!("\"{}...\"", &value[..27])
            } else {
                format!("\"{}\"", value)
            }
        }
        Object::Variable(v) => format!("?{}", v.as_str()),
        Object::QuotedTriple(_) => "<<...>>".to_string(),
    }
}

fn print_mapper_statistics(_mapper: &CanRdfMapper) {
    println!("Mapper Configuration:");
    println!("---------------------");
    println!("  Database loaded with message definitions");
    println!("  Deadband filtering enabled (0.01 threshold)");
    println!("  PROV-O provenance triples generated");
    println!("  QUDT unit mappings applied");
    println!();
}

fn print_qudt_units() {
    println!("QUDT Unit Mappings:");
    println!("-------------------");

    let units = [
        ("rpm", AutomotiveUnits::rpm()),
        ("km/h", AutomotiveUnits::kmh()),
        ("degC", AutomotiveUnits::celsius()),
        ("bar", AutomotiveUnits::bar()),
        ("kW", AutomotiveUnits::kilowatt()),
        ("L/100km", AutomotiveUnits::liters_per_100km()),
        ("%", AutomotiveUnits::percent()),
        ("V", AutomotiveUnits::volt()),
        ("A", AutomotiveUnits::ampere()),
        ("Nm", AutomotiveUnits::newton_meter()),
    ];

    for (name, iri) in units {
        println!("  {} → <{}>", name, iri);
    }
    println!();
}

fn print_turtle_example(db: &DbcDatabase) -> CanbusResult<()> {
    println!("Example Turtle Output:");
    println!("----------------------");

    let turtle = r#"@prefix vehicle: <http://automotive.example.com/vehicle/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix qudt: <http://qudt.org/vocab/unit/> .
@prefix sosa: <http://www.w3.org/ns/sosa/> .

# Vehicle observation for engine speed
vehicle:vehicle_001_EngineData_EngineSpeed a sosa:Observation ;
    sosa:observedProperty vehicle:EngineSpeed ;
    sosa:hasSimpleResult "2500.0"^^xsd:double ;
    sosa:resultTime "2025-12-24T10:30:00Z"^^xsd:dateTime ;
    qudt:unit qudt:REV-PER-MIN ;
    prov:wasGeneratedBy vehicle:vehicle_001_can_processor ;
    prov:generatedAtTime "2025-12-24T10:30:00Z"^^xsd:dateTime .

# Vehicle observation for coolant temperature
vehicle:vehicle_001_EngineData_CoolantTemp a sosa:Observation ;
    sosa:observedProperty vehicle:CoolantTemp ;
    sosa:hasSimpleResult "90.0"^^xsd:double ;
    sosa:resultTime "2025-12-24T10:30:00Z"^^xsd:dateTime ;
    qudt:unit qudt:DEG_C ;
    prov:wasGeneratedBy vehicle:vehicle_001_can_processor ;
    prov:generatedAtTime "2025-12-24T10:30:00Z"^^xsd:dateTime .

# Vehicle observation for wheel speed
vehicle:vehicle_001_VehicleSpeed_WheelSpeed a sosa:Observation ;
    sosa:observedProperty vehicle:WheelSpeed ;
    sosa:hasSimpleResult "85.5"^^xsd:double ;
    sosa:resultTime "2025-12-24T10:30:00Z"^^xsd:dateTime ;
    qudt:unit qudt:KiloM-PER-HR ;
    prov:wasGeneratedBy vehicle:vehicle_001_can_processor ;
    prov:generatedAtTime "2025-12-24T10:30:00Z"^^xsd:dateTime .
"#;

    println!("{}", turtle);

    println!(
        "DBC Messages Available: {:?}",
        db.messages.iter().map(|m| &m.name).collect::<Vec<_>>()
    );

    Ok(())
}
