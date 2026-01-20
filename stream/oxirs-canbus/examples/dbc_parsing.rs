//! DBC File Parsing Example
//!
//! Demonstrates parsing Vector CANdb++ DBC files and extracting
//! message and signal definitions for CAN data decoding.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example dbc_parsing
//! ```

use oxirs_canbus::{
    parse_dbc, ByteOrder, CanFrame, CanId, CanbusResult, DbcDatabase, SignalDecoder, ValueType,
};

fn main() -> CanbusResult<()> {
    println!("=== OxiRS DBC Parsing Example ===\n");

    // Example DBC content (embedded for demo)
    let dbc_content = r#"
VERSION ""

NS_ :
	NS_DESC_
	CM_
	BA_DEF_
	BA_
	VAL_
	CAT_DEF_
	CAT_
	FILTER
	BA_DEF_DEF_
	EV_DATA_
	ENVVAR_DATA_
	SGTYPE_
	SGTYPE_VAL_
	BA_DEF_SGTYPE_
	BA_SGTYPE_
	SIG_TYPE_REF_
	VAL_TABLE_
	SIG_GROUP_
	SIG_VALTYPE_
	SIGTYPE_VALTYPE_
	BO_TX_BU_
	BA_REL_
	BA_SGTYPE_REL_
	SG_MUL_VAL_

BS_:

BU_: Engine Transmission Dashboard

BO_ 100 EngineData: 8 Engine
 SG_ EngineSpeed : 0|16@1+ (0.125,0) [0|8031.875] "rpm" Dashboard
 SG_ EngineLoad : 16|8@1+ (0.392156862745098,0) [0|100] "%" Dashboard
 SG_ CoolantTemp : 24|8@1+ (1,-40) [-40|215] "degC" Dashboard
 SG_ ThrottlePosition : 32|8@1+ (0.392156862745098,0) [0|100] "%" Dashboard

BO_ 200 TransmissionData: 8 Transmission
 SG_ VehicleSpeed : 0|16@1+ (0.01,0) [0|655.35] "km/h" Dashboard
 SG_ CurrentGear : 16|4@1+ (1,0) [0|15] "" Dashboard
 SG_ GearMode : 20|3@1+ (1,0) [0|7] "" Dashboard

BO_ 300 FuelData: 8 Engine
 SG_ FuelLevel : 0|8@1+ (0.5,0) [0|100] "%" Dashboard
 SG_ FuelConsumption : 8|16@1+ (0.01,0) [0|655.35] "L/100km" Dashboard
 SG_ FuelRange : 24|16@1+ (1,0) [0|65535] "km" Dashboard

CM_ SG_ 100 EngineSpeed "Engine rotational speed in RPM";
CM_ SG_ 100 EngineLoad "Current engine load percentage";
CM_ SG_ 100 CoolantTemp "Engine coolant temperature";
CM_ SG_ 200 VehicleSpeed "Vehicle speed from transmission";
CM_ SG_ 200 CurrentGear "Currently selected gear (0=P, 1=R, 2=N, 3-8=D1-D6)";

VAL_ 200 GearMode 0 "Park" 1 "Reverse" 2 "Neutral" 3 "Drive" 4 "Sport" 5 "Manual" ;
VAL_ 200 CurrentGear 0 "Park" 1 "Reverse" 2 "Neutral" 3 "1st" 4 "2nd" 5 "3rd" 6 "4th" 7 "5th" 8 "6th" ;

BA_DEF_ BO_ "GenMsgCycleTime" INT 0 10000;
BA_DEF_DEF_ "GenMsgCycleTime" 100;
BA_ "GenMsgCycleTime" BO_ 100 100;
BA_ "GenMsgCycleTime" BO_ 200 50;
BA_ "GenMsgCycleTime" BO_ 300 1000;
"#;

    // Parse the DBC content
    let db = parse_dbc(dbc_content)?;

    // Display database overview
    print_database_overview(&db);

    // Display message details
    print_message_details(&db);

    // Demonstrate signal decoding
    demo_signal_decoding(&db)?;

    println!("\n=== DBC Parsing Complete ===");

    Ok(())
}

fn print_database_overview(db: &DbcDatabase) {
    println!("Database Overview:");
    println!("------------------");
    println!("  Version: {:?}", db.version);
    println!(
        "  Nodes: {:?}",
        db.nodes.iter().map(|n| &n.name).collect::<Vec<_>>()
    );
    println!("  Messages: {}", db.messages.len());
    println!(
        "  Total Signals: {}",
        db.messages.iter().map(|m| m.signals.len()).sum::<usize>()
    );
    println!(
        "  Attribute Definitions: {}",
        db.attribute_definitions.len()
    );
    println!();
}

fn print_message_details(db: &DbcDatabase) {
    println!("Message Definitions:");
    println!("--------------------");

    for msg in &db.messages {
        println!(
            "\nBO_ {} {}: {} {}",
            msg.id, msg.name, msg.dlc, msg.transmitter
        );

        for sig in &msg.signals {
            let _byte_order = match sig.byte_order {
                ByteOrder::LittleEndian => "Intel",
                ByteOrder::BigEndian => "Motorola",
            };
            let value_type = match sig.value_type {
                ValueType::Unsigned => "+",
                ValueType::Signed => "-",
            };

            println!(
                "  SG_ {} : {}|{}@1{} ({},{}) [{}|{}] \"{}\" {}",
                sig.name,
                sig.start_bit,
                sig.bit_length,
                value_type,
                sig.factor,
                sig.offset,
                sig.min,
                sig.max,
                sig.unit,
                sig.receivers.join(",")
            );

            // Print value descriptions if available
            if !sig.value_descriptions.is_empty() {
                print!("      Values: ");
                for (value, desc) in &sig.value_descriptions {
                    print!("{}=\"{}\" ", value, desc);
                }
                println!();
            }
        }
    }
    println!();
}

fn demo_signal_decoding(db: &DbcDatabase) -> CanbusResult<()> {
    println!("Signal Decoding Demo:");
    println!("---------------------");

    // Create a signal decoder
    let db_static: &'static DbcDatabase = Box::leak(Box::new(db.clone()));
    let decoder = SignalDecoder::new(db_static);

    // Simulate CAN frame for EngineData (ID 100)
    // - EngineSpeed: 4000 rpm (raw = 4000/0.125 = 32000 = 0x7D00)
    // - EngineLoad: 75% (raw = 75/0.392156862745098 = 191 = 0xBF)
    // - CoolantTemp: 90 degC (raw = 90 + 40 = 130 = 0x82)
    // - ThrottlePosition: 50% (raw = 50/0.392156862745098 = 127 = 0x7F)
    let engine_data: Vec<u8> = vec![
        0x00, 0x7D, // EngineSpeed: 32000 little-endian
        0xBF, // EngineLoad: 191
        0x82, // CoolantTemp: 130
        0x7F, // ThrottlePosition: 127
        0x00, 0x00, 0x00,
    ];

    let _engine_frame = CanFrame::new(CanId::standard(100)?, engine_data.clone())?;

    println!("\nDecoding EngineData (CAN ID: 0x064):");
    if let Ok(signals) = decoder.decode_message(100, &engine_data) {
        for (name, signal) in &signals {
            println!(
                "  {}: {:.2} {} (raw: {})",
                name, signal.physical_value, signal.unit, signal.raw_value
            );
            if let Some(desc) = &signal.description {
                println!("    Description: {}", desc);
            }
        }
    }

    // Simulate CAN frame for TransmissionData (ID 200)
    // - VehicleSpeed: 85.5 km/h (raw = 8550 = 0x2166)
    // - CurrentGear: 5th (raw = 7)
    // - GearMode: Drive (raw = 3)
    let transmission_data: Vec<u8> = vec![
        0x66, 0x21, // VehicleSpeed: 8550 little-endian
        0x37, // CurrentGear: 7 (bits 0-3), GearMode: 3 (bits 4-6)
        0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let _transmission_frame = CanFrame::new(CanId::standard(200)?, transmission_data.clone())?;

    println!("\nDecoding TransmissionData (CAN ID: 0x0C8):");
    if let Ok(signals) = decoder.decode_message(200, &transmission_data) {
        for (name, signal) in &signals {
            let value_desc = if name == "CurrentGear" || name == "GearMode" {
                // Look up enumeration value
                db.messages
                    .iter()
                    .find(|m| m.id == 200)
                    .and_then(|m| m.signals.iter().find(|s| &s.name == name))
                    .and_then(|s| {
                        s.value_descriptions
                            .get(&signal.raw_value)
                            .map(|d| format!(" ({})", d))
                    })
                    .unwrap_or_default()
            } else {
                String::new()
            };

            println!(
                "  {}: {:.2} {}{}",
                name, signal.physical_value, signal.unit, value_desc
            );
        }
    }

    // Simulate FuelData (ID 300)
    let fuel_data: Vec<u8> = vec![
        0x8C, // FuelLevel: 70% (raw = 140)
        0x3A, 0x03, // FuelConsumption: 8.26 L/100km (raw = 826)
        0xFA, 0x00, // FuelRange: 250 km
        0x00, 0x00, 0x00,
    ];

    let _fuel_frame = CanFrame::new(CanId::standard(300)?, fuel_data.clone())?;

    println!("\nDecoding FuelData (CAN ID: 0x12C):");
    if let Ok(signals) = decoder.decode_message(300, &fuel_data) {
        for (name, signal) in &signals {
            println!("  {}: {:.2} {}", name, signal.physical_value, signal.unit);
        }
    }

    Ok(())
}
