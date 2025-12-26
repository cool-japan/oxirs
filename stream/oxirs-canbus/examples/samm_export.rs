//! SAMM Aspect Model Export from DBC
//!
//! Demonstrates generating Eclipse ESMF SAMM (Semantic Aspect Meta Model)
//! aspect models from DBC database definitions.
//!
//! SAMM is the standard for Digital Twin semantic models, enabling
//! interoperability between automotive OEMs and suppliers.
//!
//! # SAMM Concepts
//!
//! - **Aspect**: Top-level model container (maps to DBC Message)
//! - **Property**: Individual data element (maps to DBC Signal)
//! - **Characteristic**: Data type and constraints
//! - **Unit**: QUDT measurement unit
//! - **Constraint**: Value range restrictions
//!
//! # Usage
//!
//! ```bash
//! cargo run --example samm_export
//! ```

use oxirs_canbus::{
    parse_dbc, validate_for_samm, CanbusResult, DbcSammGenerator, SammConfig, SAMM_C_PREFIX,
    SAMM_E_PREFIX, SAMM_PREFIX, SAMM_U_PREFIX,
};

fn main() -> CanbusResult<()> {
    println!("=== OxiRS DBC → SAMM Aspect Model Export ===\n");

    // Display SAMM overview
    print_samm_overview();

    // Create sample DBC database
    let db = create_vehicle_dbc()?;

    println!("Source DBC Database:");
    println!("--------------------");
    println!("  Messages: {}", db.messages.len());
    for msg in &db.messages {
        println!(
            "    - {} (ID: 0x{:X}, {} signals)",
            msg.name,
            msg.id,
            msg.signals.len()
        );
    }
    println!();

    // Validate DBC for SAMM compatibility
    println!("SAMM Compatibility Validation:");
    println!("------------------------------");
    let validation = validate_for_samm(&db);
    println!("  Valid: {}", validation.valid);
    println!("  Warnings: {}", validation.warnings.len());
    println!("  Errors: {}", validation.errors.len());

    if !validation.warnings.is_empty() {
        println!("\n  Warnings:");
        for warning in &validation.warnings {
            println!("    - {}", warning);
        }
    }

    if !validation.errors.is_empty() {
        println!("\n  Errors:");
        for error in &validation.errors {
            println!("    - {}", error);
        }
    }
    println!();

    // Configure SAMM generator
    let config = SammConfig {
        version: "2.1.0".to_string(),
        namespace: "com.automotive.vehicle".to_string(),
        include_comments: true,
        generate_constraints: true,
        generate_enumerations: true,
    };

    println!("SAMM Configuration:");
    println!("-------------------");
    println!("  Version: {}", config.version);
    println!("  Namespace: {}", config.namespace);
    println!("  Include Comments: {}", config.include_comments);
    println!("  Generate Constraints: {}", config.generate_constraints);
    println!("  Generate Enumerations: {}", config.generate_enumerations);
    println!();

    // Create SAMM generator
    let generator = DbcSammGenerator::new(config.clone());

    // Generate SAMM for entire database
    println!("Generated SAMM Aspect Models:");
    println!("=============================\n");

    let samm_ttl = generator.generate_from_database(&db);
    println!("{}", samm_ttl);

    // Generate for individual messages
    println!("\n--- Individual Message Export ---\n");

    for msg in &db.messages {
        println!("# Aspect: {}", msg.name);
        println!("# -------{}", "-".repeat(msg.name.len()));
        let aspect_ttl = generator.generate_for_message(msg);
        // Just show first few lines for brevity
        let lines: Vec<&str> = aspect_ttl.lines().take(20).collect();
        for line in lines {
            println!("{}", line);
        }
        if aspect_ttl.lines().count() > 20 {
            println!("  ... ({} more lines)", aspect_ttl.lines().count() - 20);
        }
        println!();
    }

    // Display namespace prefixes
    print_namespace_prefixes();

    println!("\n=== SAMM Export Complete ===");

    Ok(())
}

fn print_samm_overview() {
    println!("SAMM (Semantic Aspect Meta Model) Overview:");
    println!("-------------------------------------------");
    println!("  SAMM is the Eclipse ESMF standard for Digital Twin semantics.");
    println!("  It enables machine-readable API descriptions for IoT/automotive.");
    println!();
    println!("  Key Mappings from DBC to SAMM:");
    println!("    DBC Message  → SAMM Aspect (container)");
    println!("    DBC Signal   → SAMM Property (data element)");
    println!("    DBC Unit     → SAMM Unit (QUDT reference)");
    println!("    DBC Min/Max  → SAMM RangeConstraint");
    println!("    DBC ValueDesc→ SAMM Enumeration");
    println!();
    println!("  Output Format: Turtle (TTL) with SAMM vocabulary");
    println!();
}

fn create_vehicle_dbc() -> CanbusResult<oxirs_canbus::DbcDatabase> {
    let dbc_content = r#"
VERSION ""

NS_ :

BS_:

BU_: Engine Transmission Dashboard

BO_ 256 EngineStatus: 8 Engine
 SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
 SG_ EngineState : 16|3@1+ (1,0) [0|7] "" Dashboard
 SG_ CoolantTemp : 24|8@1+ (1,-40) [-40|215] "degC" Dashboard
 SG_ OilPressure : 32|8@1+ (4,0) [0|1020] "kPa" Dashboard
 SG_ FuelLevel : 40|8@1+ (0.5,0) [0|100] "%" Dashboard

BO_ 512 TransmissionStatus: 8 Transmission
 SG_ GearPosition : 0|4@1+ (1,0) [0|15] "" Dashboard
 SG_ GearMode : 4|3@1+ (1,0) [0|7] "" Dashboard
 SG_ ClutchEngaged : 7|1@1+ (1,0) [0|1] "" Dashboard
 SG_ TransTemp : 8|8@1+ (1,-40) [-40|215] "degC" Dashboard

BO_ 768 VehicleDynamics: 8 Engine
 SG_ VehicleSpeed : 0|16@1+ (0.01,0) [0|655.35] "km/h" Dashboard
 SG_ AccelPedalPos : 16|8@1+ (0.4,0) [0|100] "%" Dashboard
 SG_ BrakePedalPos : 24|8@1+ (0.4,0) [0|100] "%" Dashboard
 SG_ SteeringAngle : 32|16@1- (0.1,0) [-3276.8|3276.7] "deg" Dashboard

CM_ SG_ 256 EngineSpeed "Engine rotational speed in revolutions per minute";
CM_ SG_ 256 EngineState "Current engine operational state";
CM_ SG_ 256 CoolantTemp "Engine coolant temperature";
CM_ SG_ 256 OilPressure "Engine oil pressure";
CM_ SG_ 512 GearPosition "Currently engaged gear";
CM_ SG_ 512 GearMode "Transmission operating mode";
CM_ SG_ 768 VehicleSpeed "Vehicle speed from wheel sensors";
CM_ SG_ 768 SteeringAngle "Steering wheel angle (negative=left, positive=right)";

VAL_ 256 EngineState 0 "Off" 1 "Cranking" 2 "Running" 3 "Idle" 4 "Warmup" 5 "Shutdown" ;
VAL_ 512 GearPosition 0 "Park" 1 "Reverse" 2 "Neutral" 3 "D1" 4 "D2" 5 "D3" 6 "D4" 7 "D5" 8 "D6" ;
VAL_ 512 GearMode 0 "Park" 1 "Reverse" 2 "Neutral" 3 "Drive" 4 "Sport" 5 "Manual" 6 "Eco" ;
"#;

    parse_dbc(dbc_content)
}

fn print_namespace_prefixes() {
    println!("SAMM Namespace Prefixes:");
    println!("------------------------");
    println!("  samm:   {}", SAMM_PREFIX);
    println!("  samm-c: {}", SAMM_C_PREFIX);
    println!("  samm-e: {}", SAMM_E_PREFIX);
    println!("  unit:   {}", SAMM_U_PREFIX);
    println!();
    println!("Additional Standard Prefixes:");
    println!("  xsd:    http://www.w3.org/2001/XMLSchema#");
    println!("  rdfs:   http://www.w3.org/2000/01/rdf-schema#");
    println!("  qudt:   http://qudt.org/vocab/unit/");
    println!();
    println!("Digital Twin Use Cases:");
    println!("  - API Generation: SAMM → OpenAPI/AsyncAPI specifications");
    println!("  - Data Validation: Validate incoming CAN data against model");
    println!("  - Documentation: Auto-generate human-readable docs");
    println!("  - Code Generation: Generate data classes/structs");
    println!("  - Interoperability: Share models between OEMs/suppliers");
}
