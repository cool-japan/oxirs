//! RDF Integration Example
//!
//! Demonstrates the complete Modbus-to-RDF pipeline:
//! - Register value decoding
//! - RDF triple generation
//! - W3C PROV-O provenance
//! - QUDT unit handling
//! - SPARQL UPDATE generation
//!
//! # Usage
//!
//! ```bash
//! cargo run --example rdf_integration
//! ```

use chrono::Utc;
use oxirs_core::model::{Object, Predicate, RdfTerm, Subject};
use oxirs_modbus::mapping::{ModbusDataType, RegisterMap, RegisterMapping, RegisterType};
use oxirs_modbus::rdf::{
    GeneratedTriple, GraphUpdater, ModbusTripleGenerator, QudtUnit, SparqlEndpointConfig,
};
use oxirs_modbus::ModbusResult;
use std::collections::HashMap;

fn main() -> ModbusResult<()> {
    println!("=== OxiRS Modbus RDF Integration Example ===\n");

    // Demo 1: Basic triple generation
    demo_triple_generation()?;

    // Demo 2: QUDT unit handling
    demo_qudt_units();

    // Demo 3: SPARQL UPDATE generation
    demo_sparql_update()?;

    // Demo 4: Change detection with deadband
    demo_change_detection()?;

    // Demo 5: Full pipeline simulation
    demo_full_pipeline()?;

    println!("\n=== RDF Integration Example Complete ===");
    Ok(())
}

/// Helper to extract subject IRI from a triple
fn get_subject_iri(triple: &GeneratedTriple) -> String {
    match triple.triple.subject() {
        Subject::NamedNode(n) => n.as_str().to_string(),
        Subject::BlankNode(b) => b.as_str().to_string(),
        Subject::Variable(v) => v.as_str().to_string(),
        Subject::QuotedTriple(_) => "<<quoted>>".to_string(),
    }
}

/// Helper to extract predicate IRI from a triple
fn get_predicate_iri(triple: &GeneratedTriple) -> String {
    match triple.triple.predicate() {
        Predicate::NamedNode(n) => n.as_str().to_string(),
        Predicate::Variable(v) => v.as_str().to_string(),
    }
}

/// Helper to extract object value from a triple
fn get_object_value(triple: &GeneratedTriple) -> String {
    match triple.triple.object() {
        Object::NamedNode(n) => format!("<{}>", n.as_str()),
        Object::BlankNode(b) => format!("_:{}", b.as_str()),
        Object::Literal(l) => format!("\"{}\"^^{}", l.as_str(), l.datatype()),
        Object::Variable(v) => format!("?{}", v.as_str()),
        Object::QuotedTriple(_) => "<<quoted>>".to_string(),
    }
}

fn demo_triple_generation() -> ModbusResult<()> {
    println!("Demo 1: Basic Triple Generation");
    println!("--------------------------------");

    // Create register map
    let mut map = RegisterMap::new("plc001", "http://factory.example.com/device");

    map.add_register(
        RegisterMapping::new(
            0,
            ModbusDataType::Float32,
            "http://factory.example.com/property/temperature",
        )
        .with_name("Temperature")
        .with_unit(QudtUnit::celsius()),
    );

    map.add_register(
        RegisterMapping::new(
            2,
            ModbusDataType::Uint16,
            "http://factory.example.com/property/motorSpeed",
        )
        .with_name("Motor Speed")
        .with_unit(QudtUnit::rpm()),
    );

    // Create triple generator
    let mut generator = ModbusTripleGenerator::new(map);

    // Simulate register values
    // Address 0-1: FLOAT32 = 22.5 (0x41B4 0x0000)
    // Address 2: UINT16 = 1500 (0x05DC)
    let mut values: HashMap<u16, Vec<u16>> = HashMap::new();
    values.insert(0, vec![0x41B4, 0x0000]);
    values.insert(2, vec![0x05DC]);

    // Generate triples
    let timestamp = Utc::now();
    let triples = generator.generate_triples(&values, RegisterType::Holding, timestamp)?;

    println!("Generated {} RDF triples:", triples.len());
    println!();

    for triple in &triples {
        let subject = get_subject_iri(triple);
        let predicate = get_predicate_iri(triple);
        let object = get_object_value(triple);

        // Extract short names for display
        let subject_short = subject.split('/').next_back().unwrap_or(&subject);
        let predicate_short = predicate.split('/').next_back().unwrap_or(&predicate);

        println!("Subject:   {}", subject_short);
        println!("Predicate: {}", predicate_short);
        println!("Object:    {}", object);
        if let Some(ref unit) = triple.unit {
            println!("Unit:      {}", unit);
        }
        println!("Timestamp: {}", triple.timestamp);
        println!();
    }

    Ok(())
}

fn demo_qudt_units() {
    println!("Demo 2: QUDT Unit Handling");
    println!("--------------------------");

    println!("Common QUDT units for industrial IoT:");
    println!();

    let units = [
        (
            "Temperature",
            QudtUnit::celsius(),
            "Temperature in degrees Celsius",
        ),
        ("Temperature", QudtUnit::kelvin(), "Temperature in Kelvin"),
        ("Pressure", QudtUnit::bar(), "Pressure in bar"),
        ("Pressure", QudtUnit::pascal(), "Pressure in Pascals"),
        ("Voltage", QudtUnit::volt(), "Electrical voltage"),
        ("Current", QudtUnit::ampere(), "Electrical current"),
        ("Power", QudtUnit::watt(), "Electrical power"),
        ("Power", QudtUnit::kilowatt(), "Electrical power (kW)"),
        ("Frequency", QudtUnit::hertz(), "Frequency in Hz"),
        ("Speed", QudtUnit::rpm(), "Rotational speed"),
    ];

    println!("  {:>12} | {:<50}", "Category", "QUDT IRI");
    println!("  {:-<12}-+-{:-<50}", "", "");

    for (category, unit, _desc) in &units {
        println!("  {:>12} | {}", category, unit);
    }

    println!();
    println!("QUDT URIs follow the pattern:");
    println!("  http://qudt.org/vocab/unit/<UNIT_NAME>");
    println!();
    println!("Example triple with unit annotation:");
    println!("  <sensor1> :temperature \"22.5\"^^xsd:decimal .");
    println!("  <sensor1> qudt:unit <http://qudt.org/vocab/unit/DEG_C> .");
    println!();
}

fn demo_sparql_update() -> ModbusResult<()> {
    println!("Demo 3: SPARQL UPDATE Generation");
    println!("---------------------------------");

    // Create register map
    let mut map = RegisterMap::new("energy_meter", "http://factory.example.com/device");

    map.add_register(
        RegisterMapping::new(
            0,
            ModbusDataType::Float32,
            "http://factory.example.com/property/power",
        )
        .with_name("Active Power")
        .with_unit(QudtUnit::kilowatt()),
    );

    // Create triple generator
    let mut generator = ModbusTripleGenerator::new(map);

    // Simulate register values
    let mut values: HashMap<u16, Vec<u16>> = HashMap::new();
    values.insert(0, vec![0x4248, 0x0000]); // 50.0 kW

    // Generate triples
    let triples = generator.generate_triples(&values, RegisterType::Input, Utc::now())?;

    // Create graph updater
    let config = SparqlEndpointConfig::new("http://localhost:3030/dataset/update");
    let mut updater =
        GraphUpdater::new(config).with_graph("http://factory.example.com/graph/modbus");

    // Generate SPARQL UPDATE query
    let query = updater.insert_generated_local(&triples)?;

    println!("Generated SPARQL UPDATE query:");
    println!();
    println!("{}", query);
    println!();

    println!("Query structure:");
    println!("  1. PREFIX declarations for common namespaces");
    println!("  2. INSERT DATA into named graph");
    println!("  3. Triple for sensor value");
    println!("  4. PROV-O provenance triple (timestamp)");
    println!();

    Ok(())
}

fn demo_change_detection() -> ModbusResult<()> {
    println!("Demo 4: Change Detection with Deadband");
    println!("--------------------------------------");

    // Create register map with deadband
    let mut map = RegisterMap::new("sensor001", "http://example.org/device");

    map.add_register(
        RegisterMapping::new(
            0,
            ModbusDataType::Float32,
            "http://example.org/property/temperature",
        )
        .with_name("Temperature")
        .with_deadband(0.5), // Only report changes > 0.5
    );

    // Create triple generator with change tracking
    let mut generator = ModbusTripleGenerator::new(map);

    println!("Deadband threshold: 0.5 degrees");
    println!();

    // Simulate readings over time
    let readings: [(f32, &str); 6] = [
        (22.0, "Initial reading"),
        (22.3, "Small change (+0.3)"),
        (22.1, "Small change (-0.2)"),
        (22.8, "Significant change (+0.7)"),
        (22.7, "Small change (-0.1)"),
        (23.5, "Significant change (+0.8)"),
    ];

    println!(
        "  {:>10} | {:>25} | {:>12}",
        "Value", "Description", "Triples"
    );
    println!("  {:-<10}-+-{:-<25}-+-{:-<12}", "", "", "");

    for (value, desc) in &readings {
        // Convert float to registers
        let bits = value.to_bits();
        let high = ((bits >> 16) & 0xFFFF) as u16;
        let low = (bits & 0xFFFF) as u16;

        let mut values: HashMap<u16, Vec<u16>> = HashMap::new();
        values.insert(0, vec![high, low]);

        let triples = generator.generate_triples(&values, RegisterType::Holding, Utc::now())?;

        let triple_count = if triples.is_empty() {
            "0 (filtered)"
        } else {
            "1+ (reported)"
        };

        println!("  {:>10.1} | {:>25} | {:>12}", value, desc, triple_count);
    }

    println!();
    println!("Result: 3 updates instead of 6 (50% reduction in RDF updates)");
    println!();

    Ok(())
}

fn demo_full_pipeline() -> ModbusResult<()> {
    println!("Demo 5: Full Pipeline Simulation");
    println!("---------------------------------");

    println!("Simulating complete Modbus-to-RDF pipeline:");
    println!();

    // Step 1: Configure device mapping
    println!("Step 1: Configure register mapping");
    let mut map = RegisterMap::new("sdm630", "http://factory.example.com/device/energy_meter");

    // SDM630 energy meter registers (example)
    map.add_register(
        RegisterMapping::new(
            0,
            ModbusDataType::Float32,
            "http://factory.example.com/property/voltage_L1",
        )
        .with_name("Voltage L1")
        .with_unit(QudtUnit::volt()),
    );
    map.add_register(
        RegisterMapping::new(
            2,
            ModbusDataType::Float32,
            "http://factory.example.com/property/voltage_L2",
        )
        .with_name("Voltage L2")
        .with_unit(QudtUnit::volt()),
    );
    map.add_register(
        RegisterMapping::new(
            4,
            ModbusDataType::Float32,
            "http://factory.example.com/property/voltage_L3",
        )
        .with_name("Voltage L3")
        .with_unit(QudtUnit::volt()),
    );
    map.add_register(
        RegisterMapping::new(
            6,
            ModbusDataType::Float32,
            "http://factory.example.com/property/current_L1",
        )
        .with_name("Current L1")
        .with_unit(QudtUnit::ampere()),
    );
    map.add_register(
        RegisterMapping::new(
            52,
            ModbusDataType::Float32,
            "http://factory.example.com/property/total_power",
        )
        .with_name("Total Power")
        .with_unit(QudtUnit::kilowatt())
        .with_deadband(0.1),
    );

    println!("  Device: SDM630 Energy Meter");
    println!("  Registers mapped: {}", map.registers.len());
    println!();

    // Step 2: Simulate Modbus read
    println!("Step 2: Read Modbus registers");
    let mut raw_values: HashMap<u16, Vec<u16>> = HashMap::new();

    // Simulate values (these would come from actual Modbus reads)
    raw_values.insert(0, float_to_regs(230.5)); // Voltage L1
    raw_values.insert(2, float_to_regs(231.2)); // Voltage L2
    raw_values.insert(4, float_to_regs(229.8)); // Voltage L3
    raw_values.insert(6, float_to_regs(15.2)); // Current L1
    raw_values.insert(52, float_to_regs(10.5)); // Total Power

    println!("  Read {} register groups", raw_values.len());
    println!();

    // Step 3: Generate RDF triples
    println!("Step 3: Generate RDF triples");
    let mut generator = ModbusTripleGenerator::new(map);
    let triples = generator.generate_triples(&raw_values, RegisterType::Input, Utc::now())?;

    println!("  Generated {} triples", triples.len());
    println!();

    // Step 4: Show generated triples
    println!("Step 4: Generated RDF (Turtle format):");
    println!();

    println!("@prefix : <http://factory.example.com/property/> .");
    println!("@prefix device: <http://factory.example.com/device/> .");
    println!("@prefix qudt: <http://qudt.org/vocab/unit/> .");
    println!("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .");
    println!("@prefix prov: <http://www.w3.org/ns/prov#> .");
    println!();

    for triple in &triples {
        let subject_iri = get_subject_iri(triple);
        let predicate_iri = get_predicate_iri(triple);
        let object_val = get_object_value(triple);

        let subject_short = subject_iri.split('/').next_back().unwrap_or(&subject_iri);
        let predicate_short = predicate_iri
            .split('/')
            .next_back()
            .unwrap_or(&predicate_iri);

        println!(
            "device:{} :{} {} .",
            subject_short, predicate_short, object_val
        );
    }

    println!();

    // Step 5: Generate SPARQL UPDATE
    println!("Step 5: Build SPARQL UPDATE query");
    let config = SparqlEndpointConfig::new("http://localhost:3030/energy/update");
    let mut updater =
        GraphUpdater::new(config).with_graph("http://factory.example.com/graph/energy_meter");

    let query = updater.insert_generated_local(&triples)?;
    println!("  Query length: {} bytes", query.len());
    println!("  Target endpoint: http://localhost:3030/energy/update");
    println!("  Named graph: http://factory.example.com/graph/energy_meter");
    println!();

    println!("Pipeline complete!");
    println!();

    Ok(())
}

// Helper function to convert f32 to Modbus registers
fn float_to_regs(value: f32) -> Vec<u16> {
    let bits = value.to_bits();
    let high = ((bits >> 16) & 0xFFFF) as u16;
    let low = (bits & 0xFFFF) as u16;
    vec![high, low]
}
